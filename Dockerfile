FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Optional: extend library paths if needed
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

WORKDIR /UFDM

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt