# Manim Backend - Railway Deployment Guide

This guide explains how to deploy the Manim Backend application to Railway.

## Project Overview

This is a FastAPI application that generates mathematical animations using Manim based on user prompts. It uses AI (Anthropic Claude or Groq) to generate Manim code from natural language prompts, then renders the animations using Manim.

## Prerequisites

- A Railway account (https://railway.app)
- Git installed on your local machine
- AWS S3 bucket (optional but recommended for production)
- Anthropic API key or Groq API key

## Deployment Steps

### 1. Push your code to GitHub

First, push your code to a GitHub repository:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Connect Railway to your GitHub repository

1. Log in to Railway (https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Select your repository
5. Railway will automatically detect the Dockerfile and use it for deployment

### 3. Configure Environment Variables

In the Railway dashboard, add the following environment variables:

Required:
- `PORT`: Railway will set this automatically
- `ANTHROPIC_API_KEY` or `GROQ_API_KEY`: At least one is required for the AI code generation

Optional (for S3 storage):
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `S3_BUCKET_NAME`: Your S3 bucket name
- `AWS_REGION`: AWS region (default: us-east-1)

Other:
- `DEV_MODE`: Set to 0 for production

### 4. Deploy the Application

Railway will automatically deploy your application when you push changes to your repository.

### 5. Access Your Application

Once deployed, Railway will provide you with a URL to access your application.

## Troubleshooting

- Check the logs in the Railway dashboard for any errors
- Ensure all required environment variables are set
- If animations are not rendering, check if ffmpeg is installed correctly in the container

## Additional Notes

- The application uses a lot of resources for rendering animations, so you might need to upgrade your Railway plan
- For production use, it's recommended to use S3 for storing the generated videos
