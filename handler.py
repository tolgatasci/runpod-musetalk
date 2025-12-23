"""
RunPod Serverless Handler for MuseTalk
Fast lip-sync video generation
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


def run_musetalk_inference(image_path: str, audio_path: str, output_path: str) -> bool:
    """Run MuseTalk inference"""
    try:
        # Convert audio to wav if needed
        wav_path = audio_path
        if not audio_path.lower().endswith('.wav'):
            wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
            if not convert_audio_to_wav(audio_path, wav_path):
                print("Failed to convert audio to WAV")
                return False

        # MuseTalk inference command
        cmd = [
            'python', '-m', 'scripts.inference',
            '--source_image', image_path,
            '--driving_audio', wav_path,
            '--output', output_path,
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

        if result.returncode != 0:
            print(f"MuseTalk STDOUT: {result.stdout[-2000:]}")
            print(f"MuseTalk STDERR: {result.stderr[-2000:]}")
            return False

        return os.path.exists(output_path) and os.path.getsize(output_path) > 10000

    except subprocess.TimeoutExpired:
        print("MuseTalk timeout!")
        return False
    except Exception as e:
        print(f"MuseTalk error: {e}")
        import traceback
        traceback.print_exc()
        return False


def handler(event):
    """RunPod serverless handler"""

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

            # Run MuseTalk
            print("Starting MuseTalk inference...")
            if not run_musetalk_inference(image_path, audio_path, output_path):
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
