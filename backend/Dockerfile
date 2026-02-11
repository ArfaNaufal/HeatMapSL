FROM python:3.12-slim

# We still need libglib2.0-0 for general image processing 
# but we can remove the Mesa/OpenGL packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure storage directory exists
RUN mkdir -p data/heatmap_storage

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]