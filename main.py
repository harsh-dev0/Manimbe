from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from s3_storage import S3Storage

from anthropic import Anthropic
import os
import subprocess
import uuid
import logging
import signal
from pathlib import Path
import shutil
import time
import json
import sys
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="VisuaMath Forge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://manimai.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def handle_exit_signal(signum, frame):
    logger.info(f"Received exit signal {signum}. Cleaning up and shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)

BASE_DIR = Path("./outputs")
CODE_DIR = BASE_DIR / "code"
MEDIA_DIR = BASE_DIR / "media"
JOB_DIR = BASE_DIR / "jobs"
LOG_DIR = BASE_DIR / "logs"

CODE_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

file_handler = logging.FileHandler(LOG_DIR / "manim_api.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

# Initialize S3 storage for cloud deployment
s3_storage = S3Storage()

# Debug logging for S3 storage initialization status only
logger.info(f"S3 storage initialized: {s3_storage.is_enabled}")
if s3_storage.is_enabled:
    logger.info(f"S3 bucket configured: {s3_storage.bucket_name}")
else:
    logger.info("S3 storage not initialized, using local storage only")
    


try:
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        anthropic_client = Anthropic(api_key=anthropic_api_key)
        logger.info("Anthropic client initialized successfully")
    else:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            import groq
            groq_client = groq.Client(api_key=groq_api_key)
            logger.info("Groq client initialized successfully")
            anthropic_client = None
        else:
            logger.warning("No API keys found for LLM services")
            anthropic_client = None
            groq_client = None
except Exception as e:
    logger.error(f"Error initializing LLM clients: {e}")
    anthropic_client = None
    groq_client = None

DEV_MODE = os.environ.get("DEV_MODE", "0") == "1"
if DEV_MODE:
    logger.info("Running in development mode")

class PromptRequest(BaseModel):
    prompt: str

class ManimGenerationResponse(BaseModel):
    id: str
    status: str
    video_url: str = None
    title: str = None
    error: str = None

generation_jobs = {}

def load_jobs():
    try:
        job_files = list(JOB_DIR.glob("*.json"))
        for job_file in job_files:
            try:
                with open(job_file, "r") as f:
                    job_data = json.load(f)
                    job_id = job_file.stem
                    generation_jobs[job_id] = job_data
                    logger.info(f"Loaded job {job_id} from disk")
            except Exception as e:
                logger.error(f"Error loading job file {job_file}: {e}")
    except Exception as e:
        logger.error(f"Error loading jobs: {e}")

def save_job(job_id, job_data):
    try:
        job_file = JOB_DIR / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job_data, f)
        logger.info(f"Saved job {job_id} to disk")
    except Exception as e:
        logger.error(f"Error saving job {job_id}: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to VisuaMath Forge API"}

@app.get("/test-s3-upload")
def test_s3_upload():
    try:
        test_file_path = MEDIA_DIR / "test_file.txt"
        with open(test_file_path, "w") as f:
            f.write("This is a test file for S3 upload")
            
        logger.info(f"Created test file at {test_file_path}")
        logger.info(f"S3 storage enabled: {s3_storage.is_enabled}")
        logger.info(f"S3 bucket name: {s3_storage.bucket_name}")
        
        if s3_storage.is_enabled:
            s3_key = "test/test_upload.txt"
            logger.info(f"Attempting to upload test file to S3 with key: {s3_key}")
            s3_url = s3_storage.upload_file(str(test_file_path), s3_key)
            
            if s3_url:
                return {"success": True, "message": "S3 upload successful", "url": s3_url}
            else:
                return {"success": False, "message": "S3 upload failed"}
        else:
            return {"success": False, "message": "S3 storage not enabled"}
    except Exception as e:
        logger.error(f"Error in test S3 upload: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}

@app.post("/generate", response_model=ManimGenerationResponse)
async def generate_animation(request: PromptRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_data = {
        "status": "processing",
        "created_at": time.time(),
        "prompt": request.prompt
    }
    generation_jobs[job_id] = job_data
    save_job(job_id, job_data)

    try:
        background_tasks.add_task(
            process_animation_request, 
            job_id=job_id, 
            prompt=request.prompt
        )
        return ManimGenerationResponse(
            id=job_id,
            status="processing"
        )
    except Exception as e:
        error_message = f"Failed to start animation task: {str(e)}"
        logger.error(error_message)
        
        job_data = {
            "status": "failed",
            "created_at": time.time(),
            "prompt": request.prompt,
            "error": error_message
        }
        
        generation_jobs[job_id] = job_data
        save_job(job_id, job_data)
        
        return ManimGenerationResponse(
            id=job_id,
            status="failed",
            error=error_message
        )

@app.get("/status/{job_id}", response_model=ManimGenerationResponse)
async def get_job_status(job_id: str):
    try:
        logger.info(f"Status check requested for job: {job_id}")
        logger.info(f"Current jobs in memory: {list(generation_jobs.keys())}")
        
        if job_id not in generation_jobs:
            job_file = JOB_DIR / f"{job_id}.json"
            if job_file.exists():
                try:
                    with open(job_file, "r") as f:
                        generation_jobs[job_id] = json.load(f)
                        logger.info(f"Loaded job {job_id} from disk on demand")
                except Exception as e:
                    logger.error(f"Error loading job {job_id} from disk: {e}")
        
        if job_id not in generation_jobs:
            logger.error(f"Job not found: {job_id}")
            return JSONResponse(
                status_code=404,
                content={"detail": "Job not found", "id": job_id, "status": "not_found"}
            )
        
        job = generation_jobs[job_id]
        
        response = ManimGenerationResponse(
            id=job_id,
            status=job.get("status", "unknown")
        )
        
        if "video_url" in job:
            response.video_url = job["video_url"]
        if "title" in job:
            response.title = job["title"]
        if "error" in job:
            response.error = job["error"]
            
        return response
    except Exception as e:
        logger.error(f"Error processing status request for job {job_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error while checking job status",
                "id": job_id,
                "status": "error",
                "error": str(e)
            }
        )

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    if job_id not in generation_jobs or generation_jobs[job_id].get("status") != "completed":
        raise HTTPException(status_code=404, detail="Video not found or not ready")

    # Check if we have an S3 URL
    if "video_url" in generation_jobs[job_id] and generation_jobs[job_id]["video_url"].startswith("https://"):
        # Redirect to the S3 URL
        return RedirectResponse(url=generation_jobs[job_id]["video_url"])
    
    # Fall back to local file
    video_path = generation_jobs[job_id].get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        path=video_path, 
        filename=f"animation_{job_id}.mp4", 
        media_type="video/mp4"
    )

@app.on_event("startup")
def setup_periodic_cleanup():
    load_jobs()
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Start a background task for periodic cleanup
    import asyncio
    from fastapi.concurrency import run_in_threadpool
    
    async def periodic_cleanup():
        while True:
            await run_in_threadpool(cleanup_old_jobs)
            await asyncio.sleep(3600)  # Run cleanup every hour
    
    asyncio.create_task(periodic_cleanup())

def cleanup_old_jobs():
    """Clean up old jobs from memory and disk to prevent memory accumulation."""
    try:
        current_time = time.time()
        jobs_to_remove = []
        
        # Find jobs older than 24 hours
        for job_id, job_data in generation_jobs.items():
            job_age = current_time - job_data.get("created_at", current_time)
            if job_age > 86400:  # 24 hours in seconds
                jobs_to_remove.append(job_id)
        
        # Remove old jobs from memory
        for job_id in jobs_to_remove:
            if job_id in generation_jobs:
                del generation_jobs[job_id]
                logger.info(f"Removed job {job_id} from memory")
            
            # Remove job file
            job_file = JOB_DIR / f"{job_id}.json"
            if job_file.exists():
                job_file.unlink()
                logger.info(f"Removed job file {job_id}.json")
            
            # Remove code file
            code_file = CODE_DIR / f"{job_id}.py"
            if code_file.exists():
                code_file.unlink()
                logger.info(f"Removed code file {job_id}.py")
            
            # Remove media file if it exists locally
            media_files = list(MEDIA_DIR.glob(f"*{job_id}*.mp4"))
            for media_file in media_files:
                media_file.unlink()
                logger.info(f"Removed media file {media_file}")
        
        logger.info(f"Cleanup completed: removed {len(jobs_to_remove)} old jobs")
    except Exception as e:
        logger.error(f"Error during job cleanup: {str(e)}")

def process_animation_request(job_id: str, prompt: str):
    try:
        logger.info(f"Processing animation request: {job_id}, prompt: {prompt}")
        
        # Update job status to processing
        generation_jobs[job_id] = {
            "status": "processing",
            "prompt": prompt,
            "created_at": time.time()
        }
        save_job(job_id, generation_jobs[job_id])
        
        # Generate Manim code
        code, title = generate_manim_code(prompt)
        if not code:
            raise ValueError("Failed to generate Manim code")
        
        # Save code to file
        code_file_path = CODE_DIR / f"{job_id}.py"
        with open(code_file_path, "w") as f:
            f.write(code)
        
        # Update job status to rendering
        generation_jobs[job_id].update({
            "title": title,
            "status": "rendering"
        })
        save_job(job_id, generation_jobs[job_id])
        
        # Check if we have a direct URL from the code generation step
        if "direct_url" in generation_jobs[job_id]:
            # Use the direct URL instead of creating a video
            direct_url = generation_jobs[job_id]["direct_url"]
            logger.info(f"Using direct URL for job {job_id}: {direct_url}")
            video_result = {"local_path": str(MEDIA_DIR / f"{job_id}.mp4"), "s3_url": direct_url}
        else:
            # Create the animation video normally
            video_result = create_video(job_id, code_file_path)
            logger.info(f"Video creation result: {video_result}")
        
        # Handle video storage (local or S3)
        if video_result and isinstance(video_result, dict) and "local_path" in video_result:
            video_path = video_result["local_path"]
            output_path = Path(video_path)
            
            # Check if S3 URL is already available from create_video
            if "s3_url" in video_result:
                video_url = video_result["s3_url"]
                logger.info(f"Using S3 URL from create_video: {video_url}")
            else:
                # Try to upload to S3 if not already done
                try:
                    # Get AWS credentials from environment variables
                    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
                    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
                    bucket_name = os.environ.get("S3_BUCKET_NAME")
                    aws_region = os.environ.get("AWS_REGION", "us-east-1")
                    
                    if aws_access_key and aws_secret_key and bucket_name and os.path.exists(output_path):
                        logger.info(f"S3 upload: Credentials found, uploading to bucket {bucket_name}")
                        
                        # Initialize S3 client
                        import boto3
                        s3_client = boto3.client(
                            's3',
                            aws_access_key_id=aws_access_key,
                            aws_secret_access_key=aws_secret_key,
                            region_name=aws_region
                        )
                        
                        # Upload to S3
                        s3_key = f"videos/{job_id}.mp4"
                        logger.info(f"S3 upload: Uploading {output_path} to {bucket_name}/{s3_key}")
                        
                        with open(str(output_path), 'rb') as data:
                            s3_client.put_object(
                                Bucket=bucket_name,
                                Key=s3_key,
                                Body=data,
                                ContentType='video/mp4'
                            )
                        logger.info(f"S3 upload: Direct put_object completed successfully")
                        
                        # Generate the URL
                        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
                        logger.info(f"S3 upload: Success! URL: {s3_url}")
                        video_url = s3_url
                    else:
                        logger.warning(f"S3 upload: AWS credentials not found or file doesn't exist, using local URL")
                        video_url = f"/media/{job_id}.mp4"
                except Exception as e:
                    logger.error(f"S3 upload error: {str(e)}")
                    import traceback
                    logger.error(f"S3 upload error details:\n{traceback.format_exc()}")
                    video_url = f"/media/{job_id}.mp4"
        else:
            logger.error(f"Invalid video result: {video_result}")
            video_url = f"/media/{job_id}.mp4"
            video_path = str(MEDIA_DIR / f"{job_id}.mp4")
            # Create an empty file as a last resort
            Path(video_path).touch()
        
        logger.info(f"Final video URL: {video_url}")
        generation_jobs[job_id].update({
            "status": "completed",
            "video_url": video_url,
            "video_path": video_path,
            "completed_at": time.time()
        })
        save_job(job_id, generation_jobs[job_id])
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing animation request: {error_message}")
        
        # Make sure we have a title variable defined
        if 'title' not in locals():
            title = None
            
        # Update job with error status
        if job_id in generation_jobs:
            generation_jobs[job_id].update({
                "status": "failed",
                "error": error_message,
                "completed_at": time.time()
            })
        else:
            generation_jobs[job_id] = {
                "status": "failed",
                "created_at": time.time(),
                "prompt": prompt,
                "error": error_message,
                "title": title
            }
            
        save_job(job_id, generation_jobs[job_id])
        logger.info(f"Animation generation failed for job {job_id}")

def generate_manim_code(prompt: str):
    """Generate Manim code using AI with fallback to API error demo"""
    try:
        system_prompt = """You are a Manim expert. Generate only Python code for mathematical animations.

Requirements:
1. Start with a comment with title
2. Include 'from manim import *' and 'import numpy as np'
3. Define a class inheriting from Scene
4. Implement construct() with animations
5. Set camera resolution explicitly: config.pixel_height = 720, config.pixel_width = 1280
6. Center all objects properly on screen using .center() or .move_to(ORIGIN)
7. Keep animations under 20 seconds
8. Use Text() instead of MathTex when possible
9. For LaTeX, only use packages: texlive-latex-base, texlive-latex-recommended, texlive-latex-extra, texlive-science, texlive-fonts-recommended
10. Use only 2D animations
11. Add final self.wait(1) to prevent abrupt ending
12. Add config.frame_width = 14 and config.frame_height = 8 at start
13. For fractions, use a/b notation instead of "\frac"
14. Create visually appealing animations with smooth transitions
15. Avoid complex packages and custom LaTeX commands
16. Only Return the python code and nothing else
17. Do not import any other packages
18. Do not import any other modules
19. Keep animation as simple as possible
20. Do not use any complex packages
21. use only texlive-latex-base, texlive-latex-recommended, texlive-latex-extra, texlive-science, texlive-fonts-recommended
Example:
```python
# Dynamic Wave Function Visualization
from manim import *
import numpy as np

class WaveFunction(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": BLUE},
                )
                
                # Create wave function
                def func(x):
                    return np.sin(x)
                
                # Plot the wave
                graph = axes.plot(func, color=YELLOW)
                
                # Add axis labels
                x_label = MathTex("x").next_to(axes.x_axis.get_end(), DOWN)
                y_label = MathTex("y").next_to(axes.y_axis.get_end(), LEFT)
                
                # Simple equation using MathTex
                eq_text = MathTex("y = \\sin(x)", color=GREEN).to_edge(UP)
                
                # Create animation
                self.play(Create(axes))
                self.play(Write(x_label), Write(y_label))
                self.play(Write(eq_text))
                self.wait(0.5)
                self.play(Create(graph))
                self.wait(1)
```
        """

        # Try to use Anthropic Claude API if available
        if 'anthropic_client' in globals() and anthropic_client:
            logger.info("Generating code with Anthropic Claude")
            
            response = anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",  
                    max_tokens=2000,                    
                    temperature=0.1,                    
                    system=system_prompt,               
                    messages=[{
                        "role": "user", 
                        "content": f"Create a Manim animation that demonstrates: {prompt}"
                    }]
            )
            
            # Clean the response
            code = response.content[0].text
            # Remove any markdown code blocks
            code = code.replace("```python", "").replace("```", "").strip()
            # Extract title from first comment
            title = None
            lines = code.split('\n')
            for line in lines:
                if line.strip().startswith('#') and not line.strip().startswith('#!'):
                    title = line.strip('# ').strip()
                    break
                    
            # Validate code structure
            if 'from manim import' not in code or 'class' not in code or 'Scene' not in code:
                logger.warning("Claude generated code missing required elements, trying Groq or fallback")
            else:    
                if code and title:
                    logger.info("Successfully generated valid Manim code with Claude")
                    return code, title
        
        # Try to use Groq API if available and Claude failed or isn't available
        if 'groq_client' in globals() and groq_client:
            logger.info("Generating code with Groq")
            
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a Manim animation that demonstrates: {prompt}"}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            # Clean the response
            code = response.choices[0].message.content
            # Remove any markdown code blocks
            code = code.replace("```python", "").replace("```", "").strip()
            # Extract title from first comment
            title = None
            lines = code.split('\n')
            for line in lines:
                if line.strip().startswith('#') and not line.strip().startswith('#!'):
                    title = line.strip('# ').strip()
                    break
                    
            # Validate code structure
            if 'from manim import' not in code or 'class' not in code or 'Scene' not in code:
                logger.warning("Groq generated code missing required elements, using API error fallback")
            else:
                if code and title:
                    logger.info("Successfully generated valid Manim code with Groq")
                    return code, title
        
        # If we get here, both APIs failed or aren't available
        logger.warning("No working LLM API found, using API error fallback")


        # Fallback to direct URL instead of generating API error demo code
        logger.warning("AI generation failed or not available, using direct API error video URL")
        
        # Instead of generating code, we'll return a dummy code that won't be used
        # The URL will be used directly in the process_animation_request function
        api_error_code = f"""
# Dummy code - not used
from manim import *

class APIErrorDemo(Scene):
    def construct(self):
        # This code won't be executed as we're using a direct URL
        pass
"""
        
        # Update the job to use the direct URL
        job_id = next((k for k, v in generation_jobs.items() if v.get("prompt") == prompt), None)
        if job_id:
            generation_jobs[job_id]["direct_url"] = "https://manim-ai-videos.s3.us-east-1.amazonaws.com/videos/0251cdf9-76dc-4484-85d7-85dd8252ea89.mp4"
            save_job(job_id, generation_jobs[job_id])
            
        return api_error_code, "API Error Demo"

    except Exception as e:
        logger.error(f"Error in code generation: {str(e)}")
        
        # Use direct URL instead of API error demo code
        logger.warning("Exception occurred, using direct API error video URL")
        
        # Create minimal dummy code that won't be used
        api_error_code = f"""
# Dummy code - not used
from manim import *

class APIErrorDemo(Scene):
    def construct(self):
        # This code won't be executed as we're using a direct URL
        pass
"""
        
        # Update the job to use the direct URL if possible
        job_id = next((k for k, v in generation_jobs.items() if v.get("prompt") == prompt), None)
        if job_id:
            generation_jobs[job_id]["direct_url"] = "https://manim-ai-videos.s3.us-east-1.amazonaws.com/videos/0251cdf9-76dc-4484-85d7-85dd8252ea89.mp4"
            save_job(job_id, generation_jobs[job_id])
            
        return api_error_code, "API Error Demo"

# No demo code constants needed as we're using the API error demo code directly in the generate_manim_code function

def detect_scene_class(code_file_path: Path):
    try:
        with open(code_file_path, "r") as f:
            code = f.read()
        
        scene_classes = re.findall(r'class\s+([\w_]+)\s*\(\s*(?:manim\.)?Scene\s*\)', code)
        if scene_classes:
            return scene_classes[0]
            
        lines = code.split("\n")
        for line in lines:
            if "class" in line and "Scene" in line:
                parts = line.split("class ")[1].split("(")
                if len(parts) > 0:
                    class_name = parts[0].strip()
                    return class_name
        
        all_classes = re.findall(r'class\s+([\w_]+)', code)
        if all_classes:
            logger.warning(f"Could not find Scene class, using first class found: {all_classes[0]}")
            return all_classes[0]
            
        return None
    except Exception as e:
        logger.error(f"Error detecting scene class: {str(e)}")
        return None

def create_video(job_id: str, code_file_path: Path):
    try:
        output_path = MEDIA_DIR / f"{job_id}.mp4"
        scene_class = detect_scene_class(code_file_path)
        if not scene_class:
            raise ValueError("Could not detect Scene class in the code")

        # Create a valid module name from the job_id
        module_name = f"manim_scene_{job_id.replace('-', '_')}"
        
        # Copy the original code file with a valid module name
        module_path = code_file_path.parent / f"{module_name}.py"
        shutil.copy(code_file_path, module_path)

        # Create runner script with proper imports
        runner_script = f'''
from manim import *
import sys
import os

# Add the code directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import the scene module
from {module_name} import {scene_class}

# Configure Manim
config.media_dir = "{str(MEDIA_DIR)}"
config.video_dir = "{str(MEDIA_DIR)}"
config.output_file = "{job_id}"
config.frame_rate = 30
config.pixel_height = 720
config.pixel_width = 1280

# Render the scene
scene = {scene_class}()
scene.render()
'''
        runner_path = code_file_path.parent / f"run_{job_id}.py"
        with open(runner_path, "w") as f:
            f.write(runner_script)

        # Run the script in a separate process with timeout
        # Basic process arguments that work on all platforms
        popen_kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'text': True
        }

        # For Windows, add CREATE_NO_WINDOW flag
        # For Linux/Unix, we'll use preexec_fn to handle process group
        if sys.platform == 'win32':
            popen_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        else:
            # On Unix-like systems, start process in new session
            # This ensures clean process termination
            import os
            popen_kwargs['preexec_fn'] = os.setsid
            
        process = subprocess.Popen(
            [sys.executable, str(runner_path)],
            **popen_kwargs
        )

        try:
            stdout, stderr = process.communicate(timeout=120)  # 2 minute timeout
            logger.info(f"Process stdout: {stdout}")
            if stderr:
                logger.error(f"Process stderr: {stderr}")

            if process.returncode != 0:
                raise Exception(f"Render failed: {stderr}")

        except subprocess.TimeoutExpired:
            process.kill()
            raise Exception("Animation render timed out after 120 seconds")

        finally:
            # Cleanup temporary files
            try:
                runner_path.unlink(missing_ok=True)
                module_path.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Error cleaning up temp files: {e}")

        # Check for output video
        video_files = list(MEDIA_DIR.glob(f"*{job_id}*.mp4"))
        if video_files:
            output_path = video_files[0]
            logger.info(f"Found rendered video at {output_path}")
            
            # Debug S3 storage status
            logger.info(f"S3 storage enabled in create_video: {s3_storage.is_enabled}")
            logger.info(f"S3 bucket name in create_video: {s3_storage.bucket_name}")
            
            # If S3 is enabled, upload the video
            if s3_storage.is_enabled:
                logger.info(f"Attempting to upload video to S3: {output_path}")
                s3_key = f"videos/{job_id}.mp4"
                s3_url = s3_storage.upload_file(str(output_path), s3_key)
                logger.info(f"S3 upload result: {s3_url}")
                if s3_url:
                    # Store both local path and S3 URL
                    logger.info(f"Returning S3 URL: {s3_url}")
                    return {"local_path": str(output_path), "s3_url": s3_url}
            else:
                logger.info("S3 storage is not enabled, using local storage only")
            
            return {"local_path": str(output_path)}

        raise FileNotFoundError("No video file was generated")

    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        return create_dummy_video(job_id, str(e))

def create_dummy_video(job_id: str, error_message: str):
    try:
        dummy_video_path = MEDIA_DIR / f"{job_id}.mp4"
        
        # Instead of generating a video, use the provided error video URL
        error_video_url = "https://manim-ai-videos.s3.us-east-1.amazonaws.com/videos/ec808a73-d0e8-4573-9057-5c1580fa1e11.mp4"
        logger.info(f"Using direct error video URL for job {job_id}: {error_video_url}")
        
        # Return the direct S3 URL instead of local path
        return {"local_path": str(dummy_video_path), "s3_url": error_video_url}
    except Exception as e:
        logger.error(f"Failed to create dummy video reference: {str(e)}")
        dummy_video_path = MEDIA_DIR / f"{job_id}.mp4"
        dummy_video_path.touch()
        # Even in case of exception, return the direct URL
        return {"local_path": str(dummy_video_path), "s3_url": "https://manim-ai-videos.s3.us-east-1.amazonaws.com/videos/ec808a73-d0e8-4573-9057-5c1580fa1e11.mp4"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)