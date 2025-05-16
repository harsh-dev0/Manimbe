#!/bin/bash

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker and Docker Compose
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-compose

# Add current user to docker group to avoid using sudo with docker commands
sudo usermod -aG docker $USER

# Create swap file if it doesn't exist
if [ ! -f /swapfile ]; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  sudo sysctl vm.swappiness=60
  echo "vm.swappiness=60" | sudo tee -a /etc/sysctl.conf
fi

# Create required directories
mkdir -p outputs/code outputs/media outputs/jobs outputs/logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo "Creating .env file..."
  cat > .env << EOF
# LLM API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key
GROQ_API_KEY=your_groq_api_key

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET_NAME=your_bucket_name
AWS_REGION=us-east-1

# Development mode (0 for production, 1 for development)
DEV_MODE=0
EOF
  echo ".env file created. Please edit it with your actual API keys and configuration."
fi

# Set permissions
chmod +x start.sh

# Start the application
echo "Starting Manim API..."
docker-compose up -d

echo "Manim API is now running!"
echo "You can access it at http://ec2-34-207-193-20.compute-1.amazonaws.com/api/"
