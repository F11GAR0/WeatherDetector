FROM python:3.12

# Set working directory
WORKDIR /app

# Install system dependencies including Graphviz
RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY ./wd/requirements.txt .

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory
RUN mkdir -p /data

# Copy application code
COPY ./wd .

CMD ["python", "wd.py"]