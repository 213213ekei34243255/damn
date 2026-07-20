#!/bin/bash
set -e

export LD_LIBRARY_PATH=/opt/llama:$LD_LIBRARY_PATH

MODEL_DIR="/var/data/models"
MODEL_FILE="$MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_FILE" ]; then
    echo "========================================"
    echo "Downloading Qwen model..."
    echo "========================================"

    curl -L -C - \
        "$MODEL_URL" \
        -o "$MODEL_FILE"

    echo "========================================"
    echo "Model download complete."
    echo "========================================"
fi

echo "========================================"
echo "Starting llama-server..."
echo "========================================"

/opt/llama/llama-server \
    -m "$MODEL_FILE" \
    --host 127.0.0.1 \
    --port 8080 \
    > /tmp/llama.log 2>&1 &

LLAMA_PID=$!

echo "Waiting for llama-server to load the model..."

for i in {1..120}; do
    if curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo "✓ llama-server is ready."
        break
    fi

    if ! kill -0 $LLAMA_PID 2>/dev/null; then
        echo "❌ llama-server exited unexpectedly."
        echo "----- llama-server log -----"
        cat /tmp/llama.log
        exit 1
    fi

    sleep 2
done

if ! curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "❌ Timed out waiting for llama-server."
    echo "----- llama-server log -----"
    cat /tmp/llama.log
    exit 1
fi

echo "========================================"
echo "llama-server is ready."
echo "Starting Gunicorn..."
echo "========================================"

exec gunicorn app:app \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --timeout 120
