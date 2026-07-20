FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    tar \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/llama && \
    wget -O /tmp/llama.tar.gz \
    https://github.com/ggml-org/llama.cpp/releases/download/b10057/llama-b10057-bin-ubuntu-x64.tar.gz && \
    tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1 && \
    rm /tmp/llama.tar.gz

RUN chmod +x /opt/llama/*

ENV LD_LIBRARY_PATH=/opt/llama:$LD_LIBRARY_PATH

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 10000

CMD ["./start.sh"]
