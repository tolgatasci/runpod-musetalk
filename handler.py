"""
RunPod Serverless Handler for MuseTalk
Fast lip-sync video generation with configurable parameters
"""

import runpod
import base64
import tempfile
import os
import sys
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, '/app/musetalk')


def get_duration(path: str) -> float:
    """Get video/audio duration using ffprobe"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except:
        return 0.0


def convert_audio_to_wav(input_path: str, output_path: str) -> bool:
    """Convert audio to WAV format"""
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le',
            output_path
        ], capture_output=True, timeout=60)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return False


def create_inference_config(image_path: str, audio_path: str, config_path: str, bbox_shift: int = 0) -> bool:
    """Create MuseTalk inference config YAML file"""
    try:
        import yaml
        config = {
            'task_0': {
                'video_path': image_path,
                'audio_path': audio_path,
                'bbox_shift': bbox_shift
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created config: {config}")
        return True
    except Exception as e:
        print(f"Config creation error: {e}")
        return False


def run_musetalk_inference(image_path: str, audio_path: str, output_path: str,
                           bbox_shift: int = 0,
                           extra_margin: int = 10,
                           fps: int = 25,
                           batch_size: int = 8,
                           parsing_mode: str = 'jaw',
                           left_cheek_width: int = 90,
                           right_cheek_width: int = 90,
                           version: str = 'v15') -> bool:
    """Run MuseTalk inference with configurable parameters"""
    try:
        # Convert audio to wav if needed
        wav_path = audio_path
        if not audio_path.lower().endswith('.wav'):
            wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
            if not convert_audio_to_wav(audio_path, wav_path):
                print("Failed to convert audio to WAV")
                return False

        # Create output directory
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Create dynamic config file for MuseTalk
        config_path = os.path.join(os.path.dirname(image_path), 'inference_config.yaml')
        if not create_inference_config(image_path, wav_path, config_path, bbox_shift):
            print("Failed to create inference config")
            return False

        # MuseTalk inference command - uses config file for inputs
        cmd = [
            'python', '-m', 'scripts.inference',
            '--inference_config', config_path,
            '--result_dir', output_dir,
            '--extra_margin', str(extra_margin),
            '--fps', str(fps),
            '--batch_size', str(batch_size),
            '--parsing_mode', parsing_mode,
            '--left_cheek_width', str(left_cheek_width),
            '--right_cheek_width', str(right_cheek_width),
            '--version', version,
            '--use_float16',
        ]

        print(f"Running MuseTalk: {' '.join(cmd)}")

        env = os.environ.copy()
        env['PYTHONPATH'] = '/app/musetalk:' + env.get('PYTHONPATH', '')

        result = subprocess.run(
            cmd,
            cwd='/app/musetalk',
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout (MuseTalk is fast)
        )

        print(f"MuseTalk STDOUT: {result.stdout[-2000:]}")
        if result.returncode != 0:
            print(f"MuseTalk STDERR: {result.stderr[-2000:]}")
            return False

        # Find output video - MuseTalk creates in subdirectories
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith('.mp4'):
                    found_output = os.path.join(root, f)
                    if found_output != output_path:
                        shutil.move(found_output, output_path)
                    return os.path.exists(output_path) and os.path.getsize(output_path) > 10000

        return False

    except subprocess.TimeoutExpired:
        print("MuseTalk timeout!")
        return False
    except Exception as e:
        print(f"MuseTalk error: {e}")
        import traceback
        traceback.print_exc()
        return False


def handler(event):
    """
    RunPod serverless handler for MuseTalk

    Input parameters:
        - image_base64: Base64 encoded source image
        - audio_base64: Base64 encoded audio file
        - bbox_shift: Bounding box offset (-9 to 9, default: 0)
        - extra_margin: Face crop padding (default: 10)
        - fps: Output video FPS (default: 25)
        - batch_size: Processing batch size (default: 8)
        - parsing_mode: Blending mode - 'jaw' or 'face' (default: 'jaw')
        - left_cheek_width: Left cheek region (default: 90)
        - right_cheek_width: Right cheek region (default: 90)
        - version: Model version 'v1' or 'v15' (default: 'v15')
    """

    print("=" * 50)
    print("MuseTalk Handler Started")
    print("=" * 50)

    try:
        job_input = event.get('input', {})

        tmpdir = tempfile.mkdtemp()
        try:
            image_path = os.path.join(tmpdir, 'source.png')
            audio_path = os.path.join(tmpdir, 'audio.mp3')
            output_path = os.path.join(tmpdir, 'output.mp4')

            # Get image
            if 'image_base64' in job_input:
                print("Decoding image from base64...")
                image_data = base64.b64decode(job_input['image_base64'])
                with open(image_path, 'wb') as f:
                    f.write(image_data)
            elif 'image_url' in job_input:
                print("Downloading image from URL...")
                import requests
                r = requests.get(job_input['image_url'], timeout=60)
                with open(image_path, 'wb') as f:
                    f.write(r.content)
            else:
                return {'error': 'No image provided'}

            # Get audio
            if 'audio_base64' in job_input:
                print("Decoding audio from base64...")
                audio_data = base64.b64decode(job_input['audio_base64'])
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
            elif 'audio_url' in job_input:
                print("Downloading audio from URL...")
                import requests
                r = requests.get(job_input['audio_url'], timeout=60)
                with open(audio_path, 'wb') as f:
                    f.write(r.content)
            else:
                return {'error': 'No audio provided'}

            image_size = os.path.getsize(image_path)
            audio_size = os.path.getsize(audio_path)
            audio_duration = get_duration(audio_path)

            print(f"Input: image={image_size}B, audio={audio_size}B ({audio_duration:.1f}s)")

            # Get configurable parameters
            bbox_shift = int(job_input.get('bbox_shift', 0))
            extra_margin = int(job_input.get('extra_margin', 10))
            fps = int(job_input.get('fps', 25))
            batch_size = int(job_input.get('batch_size', 8))
            parsing_mode = job_input.get('parsing_mode', 'jaw')
            left_cheek_width = int(job_input.get('left_cheek_width', 90))
            right_cheek_width = int(job_input.get('right_cheek_width', 90))
            version = job_input.get('version', 'v15')

            print(f"Parameters: bbox_shift={bbox_shift}, margin={extra_margin}, fps={fps}, "
                  f"batch={batch_size}, parsing={parsing_mode}, version={version}")

            # Run MuseTalk
            print("Starting MuseTalk inference...")
            if not run_musetalk_inference(
                image_path, audio_path, output_path,
                bbox_shift=bbox_shift,
                extra_margin=extra_margin,
                fps=fps,
                batch_size=batch_size,
                parsing_mode=parsing_mode,
                left_cheek_width=left_cheek_width,
                right_cheek_width=right_cheek_width,
                version=version
            ):
                return {'error': 'MuseTalk inference failed'}

            if not os.path.exists(output_path):
                return {'error': 'No output video generated'}

            output_size = os.path.getsize(output_path)
            output_duration = get_duration(output_path)

            print(f"Output: {output_size}B ({output_duration:.1f}s)")

            # Encode output
            with open(output_path, 'rb') as f:
                video_base64 = base64.b64encode(f.read()).decode('utf-8')

            print("Success!")
            return {
                'video_base64': video_base64,
                'duration': output_duration,
                'size_bytes': output_size
            }

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        import traceback
        print(f"Handler error: {e}")
        traceback.print_exc()
        return {'error': str(e)}


if __name__ == "__main__":
    runpod.serverless.start({'handler': handler})
