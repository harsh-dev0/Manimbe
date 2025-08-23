FROM python:3.10-slim

# Install system dependencies with SAFE LaTeX packages that definitely exist
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core dependencies
    ffmpeg \
    libcairo2-dev \
    libpango1.0-dev \
    build-essential \
    pkg-config \
    python3-dev \
    libffi-dev \
    git \
    # SAFE LaTeX packages that are guaranteed to exist
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-science \
    # Essential LaTeX tools
    dvisvgm \
    ghostscript \
    # Font packages
    fonts-dejavu-core \
    fonts-liberation \
    # Clean up in the same layer to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Remove unnecessary files but keep functionality
    && rm -rf \
       /usr/share/texlive/texmf-dist/doc \
       /usr/share/texlive/texmf-dist/source \
    # Remove man pages and other documentation
    && rm -rf /usr/share/man/* /usr/share/doc/* /usr/share/info/* \
    # Clean package cache
    && rm -rf /var/cache/apt/archives/* \
    # Update font and TeX databases
    && fc-cache -f -v \
    && texhash

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
