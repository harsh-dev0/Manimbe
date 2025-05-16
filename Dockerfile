FROM python:3.10-slim

# Install system dependencies
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y \
    ffmpeg \
    libcairo2-dev \
    libpango1.0-dev \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-fonts-recommended \
    texlive-science \
    texlive-fonts-extra\
    texlive-xetex \
    texlive-latex-extra \
    lmodern \
    build-essential \
    pkg-config \
    python3-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories and ensure they have correct permissions
RUN mkdir -p ./outputs/code ./outputs/media ./outputs/jobs ./outputs/logs && \
    chmod -R 777 ./outputs

# Expose the port
EXPOSE 8000

# Command to run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
