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

# CORRECTED for Render's actual compute model: CPU + RAM only, no GPU
# (earlier -ngl/--flash-attn advice assumed a GPU instance that doesn't
# exist here - those flags are removed).
#
# The real problem with the original script: no --threads was set at
# all. llama.cpp's thread auto-detection frequently misreads cgroup CPU
# quotas inside containers and picks a number far below what's actually
# allocated - so despite paying for 8-12 CPUs, llama-server may have
# been generating on as few as 1-4 threads the whole time. A 1.5B Q4
# model on properly-used 8-12 CPU cores should produce a short JSON plan
# in low single-digit seconds, not 45-120s.
#
#   --threads N          Threads used for token generation. Set explicitly
#                         rather than trusting auto-detect. Using N-2
#                         leaves headroom for the Flask/Gunicorn process
#                         and OS scheduling running alongside it in the
#                         same container. Adjust the number below to
#                         whatever this specific service is actually
#                         provisioned with (8-12 cores per your answer -
#                         using 10 as a middle-ground default; push to 12
#                         if you confirm the full 12 is dedicated to this
#                         service, or down to 6-8 if the count is on the
#                         lower end of that range).
#
#   --threads-batch N     Threads used for prompt processing specifically
#                         (can differ from generation threads). Matching
#                         it to --threads is a reasonable default.
#
#   --ctx-size 4096       This task is short structured JSON output - 4096
#                         tokens is plenty and keeps prompt-processing
#                         cost bounded. Raise only if you hit truncation.
#
#   --parallel 1          The Electron client only ever has one goal in
#                         flight at a time (see runtime.cjs's
#                         currentGoalEntry gate), so a single processing
#                         slot is correct and avoids wasted memory on
#                         unused parallel slots.
#
#   --batch-size 512      Default is usually fine, but setting it
#                         explicitly avoids surprises across llama.cpp
#                         versions. Raise if prompt processing (not
#                         generation) turns out to be the slow part.
THREADS=10

/opt/llama/llama-server \
    -m "$MODEL_FILE" \
    --host 127.0.0.1 \
    --port 8080 \
    --threads "$THREADS" \
    --threads-batch "$THREADS" \
    --ctx-size 4096 \
    --batch-size 512 \
    --parallel 1 \
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

# Sanity check: confirm the thread count actually took effect and see
# what CPU features (AVX2/AVX512/etc.) this build detected - a generic,
# non-SIMD-optimized build can be several times slower than one that
# uses the CPU's actual instruction set, independent of thread count.
echo "========================================"
echo "Startup diagnostics:"
grep -i "n_threads\|system_info\|AVX" /tmp/llama.log | tail -20 || echo "(no matching diagnostic lines found - check /tmp/llama.log manually)"
echo "========================================"

# One-shot timed benchmark: sends a tiny real request through the exact
# same endpoint the app uses, so you get a concrete "this took Xs"
# number in the deploy log on every restart, instead of only finding out
# it's slow when a real user hits a timeout.
echo "========================================"
echo "Benchmark: timing a minimal completion request..."
BENCH_START=$(date +%s%N)
curl -s -m 60 http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"local","messages":[{"role":"user","content":"Reply with just: ok"}],"max_tokens":8}' \
    > /tmp/llama_bench.json 2>&1 || echo "(benchmark request failed - see /tmp/llama_bench.json)"
BENCH_END=$(date +%s%N)
echo "Benchmark completed in $(( (BENCH_END - BENCH_START) / 1000000 ))ms"
echo "========================================"

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
