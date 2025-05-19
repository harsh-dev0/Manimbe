# VisuaMath Forge API

VisuaMath Forge is a web API service that generates mathematical animations using the Manim library driven by AI. The service takes natural language prompts and converts them into visual mathematical demonstrations using either Claude or Groq AI models.

## Features

- Convert natural language prompts into Manim animations
- AI-powered code generation using Claude and Groq models (with fallback options)
- RESTful API with background task processing
- Support for both local file storage and S3 cloud storage
- Job status tracking and management
- Automatic cleanup of old jobs and resources

## System Requirements

- Python 3.8+
- FastAPI
- Manim animation library
- Anthropic or Groq API access (optional but recommended)
- AWS S3 access for cloud storage (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/visuamath-forge.git
cd visuamath-forge
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install fastapi uvicorn python-dotenv anthropic boto3 groq
pip install manim
```

4. Install additional dependencies required by Manim:
```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-science texlive-fonts-recommended ffmpeg dvisvgm

# MacOS with Homebrew
brew install ffmpeg
brew cask install mactex

# Windows
# Install MiKTeX and FFmpeg manually
```

## Configuration

1. Create a `.env` file in the project root:
```bash
# API Keys (use at least one)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# AWS S3 Configuration (optional)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET_NAME=your_bucket_name
AWS_REGION=us-east-1

# Development mode
DEV_MODE=0  # Set to 1 for development mode

# Server port (optional)
PORT=8000
```

## Directory Structure

The service will automatically create these directories if they don't exist:
- `outputs/code/`: Stores generated Manim code files
- `outputs/media/`: Stores rendered animation videos
- `outputs/jobs/`: Stores job data in JSON format
- `outputs/logs/`: Stores application logs

## Running the Server

Start the server using:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or simply run:
```bash
python main.py
```

## API Endpoints

### Main Endpoints

- `GET /`: Welcome message
- `POST /generate`: Submit a new animation generation request
- `GET /status/{job_id}`: Check the status of a generation job
- `GET /download/{job_id}`: Download or redirect to the generated video
- `GET /test-s3-upload`: Test if S3 upload is working correctly

### Example Usage

To generate a new animation:

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Show the concept of integration as the area under a curve"}'
```

Response:
```json
{
  "id": "12345678-1234-5678-1234-567812345678",
  "status": "processing"
}
```

To check status:
```bash
curl "http://localhost:8000/status/12345678-1234-5678-1234-567812345678"
```

Response (when complete):
```json
{
  "id": "12345678-1234-5678-1234-567812345678",
  "status": "completed",
  "video_url": "https://your-bucket.s3.amazonaws.com/videos/12345678-1234-5678-1234-567812345678.mp4",
  "title": "Integration as Area Under a Curve"
}
```

## Error Handling

The API provides fallback mechanisms when:
- AI code generation fails
- Manim rendering fails
- S3 upload fails

In each case, appropriate error messages are logged, and a default video may be provided as a fallback.

## Deployment

### Docker (recommended)

1. Build the Docker image:
```bash
docker build -t visuamath-forge .
```

2. Run the container:
```bash
docker run -d -p 8000:8000 --env-file .env visuamath-forge
```

### Cloud Deployment

The service is designed to work well with cloud platforms like:
- AWS Elastic Beanstalk
- Google Cloud Run
- Heroku

When deployed to cloud platforms, configure environment variables instead of using a `.env` file.

## Maintenance

- The server automatically cleans up old jobs every 9 minutes (jobs older than 10 minutes)
- For manual cleanup, you can delete files in the `outputs/` directory

## Troubleshooting

- Check logs in `outputs/logs/manim_api.log`
- If videos are not being created:
  - Ensure Manim and its dependencies are correctly installed
  - Check for LaTeX errors in the logs
  - Make sure FFmpeg is installed and accessible


## Credits

- Manim: Mathematical Animation Engine by 3Blue1Brown (Grant Sanderson)
- Anthropic's Claude and Groq's Models for AI-powered code generation