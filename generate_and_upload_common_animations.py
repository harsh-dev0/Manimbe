import os
import json
import time
import logging
import requests
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from upload_to_s3 import upload_file_to_s3
from animation_cache import AnimationCache
from s3_storage import S3Storage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import common animations from preload_animations.py
try:
    from preload_animations import COMMON_ANIMATIONS
except ImportError:
    logger.error("Could not import COMMON_ANIMATIONS from preload_animations.py")
    COMMON_ANIMATIONS = []

# API endpoint (change if your server is running on a different URL)
API_URL = "http://localhost:8000"

def generate_animation(prompt):
    """Generate an animation using the API"""
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={"prompt": prompt},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error generating animation: {e}")
        return None

def check_animation_status(job_id):
    """Check the status of an animation generation job"""
    try:
        response = requests.get(
            f"{API_URL}/status/{job_id}",
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error checking animation status: {e}")
        return None

def download_animation(job_id, output_path):
    """Download an animation"""
    try:
        response = requests.get(
            f"{API_URL}/download/{job_id}",
            stream=True,
            timeout=30
        )
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        logger.error(f"Error downloading animation: {e}")
        return False

def start_server():
    """Start the API server if it's not already running"""
    try:
        # Check if server is already running
        try:
            response = requests.get(f"{API_URL}/", timeout=5)
            if response.status_code == 200:
                logger.info("Server is already running")
                return True
        except:
            logger.info("Server is not running, starting it...")
        
        # Start the server
        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'text': True
        }
        
        # Add Windows-specific flags only if on Windows
        if sys.platform == 'win32':
            popen_kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
            
        process = subprocess.Popen(
            ["python", "main.py"],
            **popen_kwargs
        )
        
        # Wait for server to start
        for _ in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            try:
                response = requests.get(f"{API_URL}/", timeout=5)
                if response.status_code == 200:
                    logger.info("Server started successfully")
                    return True
            except:
                pass
        
        logger.error("Failed to start server")
        return False
    
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False

def main():
    # Create output directory
    output_dir = Path("./outputs/preloaded")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start the server if it's not already running
    if not start_server():
        logger.error("Could not start server, exiting")
        return
    
    # Initialize S3 storage and animation cache
    s3_storage = S3Storage()
    animation_cache = AnimationCache(s3_storage)
    
    if not s3_storage.is_enabled:
        logger.error("S3 storage is not configured. Cannot upload animations.")
        return
    
    # Process each common animation
    results = []
    for animation in COMMON_ANIMATIONS:
        prompt = animation.get("prompt")
        job_id = animation.get("job_id")
        title = animation.get("title")
        s3_key = animation.get("s3_key")
        
        if not all([prompt, job_id, title, s3_key]):
            logger.warning(f"Missing required data for animation: {animation}")
            continue
        
        logger.info(f"Processing animation: {title}")
        
        # Step 1: Generate the animation
        logger.info(f"Generating animation for prompt: {prompt}")
        response = generate_animation(prompt)
        if not response:
            logger.error(f"Failed to generate animation for: {title}")
            continue
        
        # Get the job ID from the response
        api_job_id = response.get("id")
        if not api_job_id:
            logger.error(f"No job ID in response for: {title}")
            continue
        
        # Step 2: Wait for the animation to complete
        logger.info(f"Waiting for animation to complete: {api_job_id}")
        max_wait_time = 300  # 5 minutes
        wait_interval = 5  # 5 seconds
        for _ in range(max_wait_time // wait_interval):
            status_response = check_animation_status(api_job_id)
            if not status_response:
                logger.error(f"Failed to check status for job: {api_job_id}")
                break
            
            status = status_response.get("status")
            if status == "completed":
                logger.info(f"Animation completed: {api_job_id}")
                break
            elif status == "failed":
                logger.error(f"Animation failed: {api_job_id}")
                break
            
            time.sleep(wait_interval)
        else:
            logger.error(f"Timed out waiting for animation: {api_job_id}")
            continue
        
        if status != "completed":
            continue
        
        # Step 3: Download the animation
        output_path = output_dir / f"{job_id}.mp4"
        logger.info(f"Downloading animation to: {output_path}")
        if not download_animation(api_job_id, output_path):
            logger.error(f"Failed to download animation: {api_job_id}")
            continue
        
        # Step 4: Upload to S3
        logger.info(f"Uploading animation to S3: {s3_key}")
        s3_url = upload_file_to_s3(output_path, s3_key, "video/mp4")
        if not s3_url:
            logger.error(f"Failed to upload animation to S3: {job_id}")
            continue
        
        # Step 5: Add to cache
        logger.info(f"Adding animation to cache: {job_id}")
        animation_cache.cache_animation(prompt, job_id, title, s3_url=s3_url)
        
        results.append({
            "job_id": job_id,
            "title": title,
            "prompt": prompt,
            "s3_key": s3_key,
            "s3_url": s3_url,
            "local_path": str(output_path)
        })
        
        logger.info(f"Successfully processed animation: {title}")
    
    # Save results
    with open("generated_animations.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Processed {len(results)} animations out of {len(COMMON_ANIMATIONS)}")
    logger.info("Results saved to generated_animations.json")

if __name__ == "__main__":
    main()
