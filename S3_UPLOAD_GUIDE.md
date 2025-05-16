# S3 Upload Guide for Manim Animations

This guide explains how to upload animations to your S3 bucket for the Manim Backend project.

## Prerequisites

1. AWS account with S3 access
2. AWS credentials configured (Access Key ID and Secret Access Key)
3. S3 bucket created
4. Python 3.10+ installed

## Setup

Your AWS credentials should be in your `.env` file:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket_name
AWS_REGION=us-east-1
```

I see you already have these configured in your `.env` file.

## Uploading Files to S3

### Method 1: Using the upload_to_s3.py Script

I've created a script that makes uploading files to S3 easy. Here's how to use it:

#### Upload a Single File

```bash
python upload_to_s3.py path/to/your/file.mp4
```

This will upload the file to your S3 bucket with the key `uploads/file.mp4`.

To specify a custom S3 key (path in the bucket):

```bash
python upload_to_s3.py path/to/your/file.mp4 --key custom/path/file.mp4
```

#### Upload a Directory

```bash
python upload_to_s3.py path/to/your/directory
```

This will upload all files in the directory to your S3 bucket with the prefix `uploads/directory/`.

To specify a custom S3 prefix:

```bash
python upload_to_s3.py path/to/your/directory --key custom/path/
```

### Method 2: Using the Preload Animations Script

The `preload_animations.py` script is designed to register common animations in your cache system. However, it doesn't actually upload the animations to S3. You need to:

1. Create the animations (either manually or by running your application)
2. Upload them to S3 with the correct keys
3. Run the preload script to register them in your cache

Here's the process:

#### Step 1: Generate Common Animations

For each animation in the `COMMON_ANIMATIONS` list in `preload_animations.py`:

1. Run your application to generate the animation:
   ```bash
   curl -X POST "http://localhost:8000/api/generate" -H "Content-Type: application/json" -d '{"prompt": "Show the Pythagorean theorem with a visual proof"}'
   ```

2. Note the job ID from the response.

3. Wait for the animation to complete:
   ```bash
   curl "http://localhost:8000/api/status/{job_id}"
   ```

4. Download the generated video.

#### Step 2: Upload Animations to S3

For each animation, upload it to S3 with the correct key:

```bash
python upload_to_s3.py path/to/animation.mp4 --key preloaded/pythagorean_theorem.mp4
```

Make sure the key matches the `s3_key` in the `COMMON_ANIMATIONS` list.

#### Step 3: Run the Preload Script

After uploading all animations, run the preload script:

```bash
python preload_animations.py
```

This will register all the animations in your cache system.

### Method 3: Using the AWS CLI

If you prefer using the AWS CLI:

1. Install the AWS CLI:
   ```bash
   pip install awscli
   ```

2. Configure AWS credentials:
   ```bash
   aws configure
   ```

3. Upload a file:
   ```bash
   aws s3 cp path/to/your/file.mp4 s3://your-bucket-name/path/in/bucket/file.mp4
   ```

4. Upload a directory:
   ```bash
   aws s3 cp path/to/your/directory s3://your-bucket-name/path/in/bucket/ --recursive
   ```

## Verifying Uploads

To verify that your files were uploaded correctly:

```bash
aws s3 ls s3://your-bucket-name/path/in/bucket/
```

Or check the AWS S3 console: https://s3.console.aws.amazon.com/s3/

## Generating URLs

The format for S3 URLs is:

```
https://{bucket-name}.s3.amazonaws.com/{key}
```

For example:
```
https://manim-ai-videos.s3.amazonaws.com/preloaded/pythagorean_theorem.mp4
```

## Troubleshooting

### Common Issues

1. **Access Denied**: Check your AWS credentials and bucket permissions
2. **File Not Found**: Verify the file path is correct
3. **Bucket Not Found**: Confirm the bucket name is correct
4. **Region Issues**: Make sure you're using the correct AWS region

### Checking Logs

If you encounter issues, check the logs:

```bash
tail -f outputs/logs/manim_api.log
```

## Best Practices

1. Organize your S3 bucket with a clear structure:
   - `preloaded/` for common animations
   - `videos/` for user-generated animations
   - `uploads/` for manually uploaded files

2. Use descriptive filenames that indicate the content

3. Set appropriate content types for your files (the script handles this automatically)

4. Consider setting up lifecycle policies in S3 to automatically delete old files

## Using the Animation Cache

Once animations are uploaded to S3 and registered in the cache, when a user requests a similar animation, the system will:

1. Check the cache for a matching animation
2. Return the S3 URL immediately if found
3. Only generate a new animation if nothing is found in the cache

This saves significant computational resources and Railway credits.
