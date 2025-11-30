FROM python:3.11-slim

# Install system dependencies required for Tkinter and potential build tools
RUN apt-get update && apt-get install -y \
    python3-tk \
    tk-dev \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ src/
COPY README.md . 

# Set PYTHONPATH to ensure imports work correctly
ENV PYTHONPATH=/app

# Create directory for saved data to ensure permissions are correct if mounted
RUN mkdir -p hand_made_kgs .cache && \
    chmod 777 .cache

# Command to launch the demo app
CMD ["python", "-m", "src.demo_app.app"]
