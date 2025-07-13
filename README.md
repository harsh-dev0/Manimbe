# VisuaMath Forge API

VisuaMath Forge is a web API service that generates mathematical animations using the Manim library driven by AI. The service takes natural language prompts and converts them into visual mathematical demonstrations using Gemini (primary) and Claude (fallback) AI models.

## Features

- Convert natural language prompts into Manim animations
- AI-powered code generation using Gemini 1.5 Flash (primary) and Claude models (fallback)
- BYOK (Bring Your Own Key) support for Gemini API
- RESTful API with background task processing
- Support for both local file storage and S3 cloud storage
- Videos uploaded to S3 with prompt-based filenames for better organization
- Job status tracking and management
- Automatic cleanup of old jobs and resources

## System Requirements

- Python 3.8+
- FastAPI
- Manim animation library
- Google Gemini API access (recommended)
- Anthropic Claude API access (fallback)
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
pip install fastapi uvicorn python-dotenv google-generativeai anthropic boto3
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
GEMINI_API_KEY=your_gemini_api_key_here  # Primary AI model
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Fallback AI model

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

To generate a new animation with BYOK (Bring Your Own Key) for Gemini:

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Show the concept of integration as the area under a curve",
    "gemini_api_key": "your_gemini_api_key_here"
  }'
```

Or without BYOK (uses global API key):

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
  "video_url": "https://your-bucket.s3.amazonaws.com/videos/integration_as_area_under_curve_a1b2c3d4.mp4",
  "title": "Integration as Area Under a Curve"
}
```

## AI Model Priority

The service uses the following priority for AI code generation:

1. **Gemini 1.5 Flash** (Primary) - Uses the best available Gemini model
   - Supports BYOK (Bring Your Own Key) from request
   - Falls back to global GEMINI_API_KEY if no BYOK provided
   
2. **Claude 3.5 Haiku** (Fallback) - Used when Gemini is unavailable or fails
   - Uses global ANTHROPIC_API_KEY

3. **Error Demo Video** - Final fallback when both AI models fail

## S3 File Naming

Videos uploaded to S3 use prompt-based filenames for better organization:
- Original prompt: "Show the concept of integration as the area under a curve"
- S3 filename: `integration_as_area_under_curve_a1b2c3d4.mp4`
- Special characters are removed, spaces become underscores
- A hash is added for uniqueness

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
- If Gemini API fails:
  - Check your API key is valid
  - Verify you have sufficient quota
  - Try using BYOK with a different API key

## Credits

- Manim: Mathematical Animation Engine by 3Blue1Brown (Grant Sanderson)
- Google's Gemini models for AI-powered code generation
- Anthropic's Claude models for fallback AI generation