FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install --no-cache-dir -e .

# Default environment
ENV OMNIMODAL_DIFFUSERS_MODEL=stabilityai/stable-diffusion-xl-base-1.0
ENV OMNIMODAL_DIFFUSERS_DTYPE=float16
ENV OMNIMODAL_DIFFUSERS_STEPS=25

EXPOSE 8000

CMD ["uvicorn", "web_stable_diffusion.runtime.api:app", "--host", "0.0.0.0", "--port", "8000"]
