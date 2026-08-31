# Build and Run NInfer with Qwen3.8-27B QUASAR NVFP4 on Ubuntu 26.04 LTS

> **Target machine:** Ubuntu 26.04 LTS, NVIDIA GeForce RTX 5090 Laptop GPU with 24 GiB VRAM, NVIDIA driver already working, CUDA Toolkit 13.2, and 64 GiB system RAM.
>
> **Tutorial snapshot:** 2026-08-30.
>
> **Outcome:** build the exact NInfer runtime required by the published QUASAR `.ninfer` artifact, download and verify the model, run a one-shot smoke test, expose an OpenAI-compatible local API, and tune the context size for a 24 GiB laptop GPU.

---

## 1. What this guide installs

This is **not a llama.cpp setup**. It uses:

- Mirko Covizzi's RTX 5090 Mobile fork of NInfer;
- the exact runtime commit required by the QUASAR artifact;
- the preconverted `qwen3_8_27b_nvfp4.ninfer` model file;
- native NVIDIA Blackwell NVFP4 execution;
- an INT8 KV cache;
- optional MTP speculative decoding;
- a local OpenAI-compatible HTTP server.

You do **not** need to convert the Hugging Face Safetensors checkpoint yourself. The published `.ninfer` artifact already contains the text model, vision tower, MTP draft model, tokenizer, chat template, generation configuration, and media-processing resources.

NInfer is deliberately specialized. It runs one resident model on one RTX 5090-class GPU. It does not provide llama.cpp-style CPU offload, multi-GPU sharding, or generic checkpoint loading.

---

## 2. Important compatibility lock

The QUASAR artifact is tied to this runtime revision:

```text
Repository: https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile
Branch:     feat/quasar-nvfp4-converter
Commit:     2d11c9927be0ed8be20566814ec4bf97db2fab4f
```

The Hugging Face model card still shows the repository's former `ninfer-rtx5090-laptop` name. GitHub redirects that name to `ninfer-rtx5090-mobile`.

**Do not build arbitrary current upstream NInfer for this particular artifact.** Build the pinned commit first. Updating the runtime independently may cause artifact-profile or load-plan incompatibilities.

The model artifact is also pinned by checksum:

```text
Filename: qwen3_8_27b_nvfp4.ninfer
Size:     17,555,331,072 bytes (16.35 GiB)
SHA-256:  931816373707010b03e6e4dcba10f5265c3e820584dacd0eb2c6039e397045cd
```

---

## 3. Planned directory layout

This tutorial uses:

```text
~/src/ninfer-quasar/
    build/apps/ninfer
    build/apps/ninfer-serve

~/models/ninfer/qwen3.8-27b-quasar/
    qwen3_8_27b_nvfp4.ninfer

~/.local/bin/ninfer-qwen38-quasar

~/.config/ninfer/qwen38.env
~/.config/systemd/user/ninfer-qwen38.service
```

No system-wide NInfer installation is required. NInfer currently has no normal install target; its binaries are run from the source build tree.

---

## 4. Preflight checks

### 4.1 Confirm the operating system and architecture

```bash
source /etc/os-release

printf 'OS: %s\n' "$PRETTY_NAME"
printf 'Architecture: %s\n' "$(uname -m)"
printf 'Kernel: %s\n' "$(uname -r)"
```

Expected architecture:

```text
x86_64
```

### 4.2 Confirm that the NVIDIA driver sees the GPU

```bash
nvidia-smi
```

A compact query:

```bash
nvidia-smi \
  --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader
```

Expected result should identify an RTX 5090 Laptop GPU and approximately 24 GiB of VRAM.

Also check for other processes consuming VRAM:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Graphical applications may not all appear in the compute-process list, so also inspect the process table shown by ordinary `nvidia-smi`.

### 4.3 Confirm that CUDA 13.2 is the compiler actually selected

```bash
command -v nvcc
nvcc --version
readlink -f "$(command -v nvcc)"
```

NInfer requires CUDA Toolkit 13.1 or newer. Your CUDA 13.2 installation satisfies that requirement.

This guide assumes `nvcc --version` reports CUDA 13.2. Do not replace a working NVIDIA driver or toolkit merely to follow the tutorial.

### 4.4 Confirm available disk space

The model alone occupies 16.35 GiB. Leave additional space for the Git repository, build products, temporary download state, and logs.

```bash
df -h "$HOME"
```

Having at least 25-30 GiB free before beginning is a practical minimum.

---

## 5. Install Ubuntu build dependencies

Ubuntu 26.04 provides a sufficiently new CMake and the required FFmpeg and libcurl development packages.

```bash
sudo apt update

sudo apt install -y \
  build-essential \
  git \
  cmake \
  ninja-build \
  pkg-config \
  libavformat-dev \
  libavcodec-dev \
  libavutil-dev \
  libswscale-dev \
  libcurl4-openssl-dev \
  ca-certificates \
  curl \
  jq
```

The NInfer build requires:

- CMake 3.28 or newer;
- a C++20-capable host compiler;
- Ninja;
- `pkg-config`;
- `libavformat >= 60`;
- `libavcodec >= 60`;
- `libavutil >= 58`;
- `libswscale >= 7`;
- `libcurl >= 7.85`;
- CUDA Toolkit 13.1 or newer.

Verify the installed versions:

```bash
cmake --version
ninja --version
g++ --version
pkg-config --version

pkg-config --modversion \
  libavformat \
  libavcodec \
  libavutil \
  libswscale \
  libcurl
```

Ubuntu 26.04's `cmake` package is 4.2.x, and its current FFmpeg/libcurl packages exceed NInfer's minimum versions.

Linux kernel headers are not required merely to compile NInfer when the NVIDIA driver is already installed and working. They matter when building or rebuilding the NVIDIA kernel module.

---

## 6. Make the existing CUDA toolkit visible

Run:

```bash
NVCC="$(readlink -f "$(command -v nvcc)")"
printf 'Using nvcc: %s\n' "$NVCC"
```

When `nvcc` is already found and reports 13.2, proceed to the next section.

Only when `nvcc` exists under `/usr/local/cuda-13.2/bin` but is not on `PATH`, set:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

To persist those variables for interactive shells:

```bash
cat >> "$HOME/.profile" <<'EOF'

# CUDA 13.2
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF
```

Then reload the profile:

```bash
source "$HOME/.profile"
nvcc --version
```

Do not add this block when your CUDA toolkit is installed elsewhere. Use the real path returned by `readlink -f "$(command -v nvcc)"`.

---

## 7. Clone the exact NInfer fork and commit

Create source and model parent directories:

```bash
mkdir -p "$HOME/src"
mkdir -p "$HOME/models/ninfer/qwen3.8-27b-quasar"
```

Clone the branch containing the QUASAR profile:

```bash
git clone \
  --branch feat/quasar-nvfp4-converter \
  https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile.git \
  "$HOME/src/ninfer-quasar"
```

Pin the working tree to the model-card revision:

```bash
cd "$HOME/src/ninfer-quasar"

git checkout --detach \
  2d11c9927be0ed8be20566814ec4bf97db2fab4f
```

Verify it exactly:

```bash
EXPECTED_COMMIT=2d11c9927be0ed8be20566814ec4bf97db2fab4f
ACTUAL_COMMIT="$(git rev-parse HEAD)"

printf 'Expected: %s\n' "$EXPECTED_COMMIT"
printf 'Actual:   %s\n' "$ACTUAL_COMMIT"

test "$ACTUAL_COMMIT" = "$EXPECTED_COMMIT"
```

No output from the final `test` command means the comparison succeeded.

You should now be in detached-HEAD mode. That is intentional: it prevents an ordinary branch update from silently moving the runtime away from the version expected by the artifact.

---

## 8. Configure and build NInfer

NInfer's build is intentionally restricted to the RTX 5090 Blackwell target `sm_120a`. The project rejects other CUDA architectures.

Select the same `nvcc` that passed the preflight check:

```bash
cd "$HOME/src/ninfer-quasar"

NVCC="$(readlink -f "$(command -v nvcc)")"
"$NVCC" --version
```

Configure a clean Release build:

```bash
rm -rf build

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES=120a
```

Build:

```bash
cmake --build build --parallel "$(nproc)"
```

The project's Ninja configuration serializes the CUDA link stage, so using all CPU cores for compilation does not imply several simultaneous GPU-link operations.

On a thermally constrained laptop, a lower parallel count is also valid:

```bash
cmake --build build --parallel 8
```

### 8.1 Verify the binaries

```bash
test -x "$HOME/src/ninfer-quasar/build/apps/ninfer"
test -x "$HOME/src/ninfer-quasar/build/apps/ninfer-serve"

"$HOME/src/ninfer-quasar/build/apps/ninfer" --help
"$HOME/src/ninfer-quasar/build/apps/ninfer-serve" --help
```

Check for unresolved shared libraries:

```bash
if ldd "$HOME/src/ninfer-quasar/build/apps/ninfer-serve" | grep -q 'not found'; then
  echo "ERROR: unresolved runtime libraries"
  ldd "$HOME/src/ninfer-quasar/build/apps/ninfer-serve" | grep 'not found'
  exit 1
else
  echo "All ninfer-serve shared libraries resolved."
fi
```

---

## 9. Download the preconverted QUASAR artifact

The Hugging Face CLI can run directly through `uvx`, so no persistent Python package installation is necessary.

Confirm `uvx` is available:

```bash
command -v uvx
```

Download only the required `.ninfer` file:

```bash
MODEL_DIR="$HOME/models/ninfer/qwen3.8-27b-quasar"

uvx hf download \
  MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir "$MODEL_DIR"
```

The repository is public, so authentication should not be necessary. A Hugging Face login can still be configured with:

```bash
uvx hf auth login
```

### 9.1 Verify file size and checksum

```bash
cd "$HOME/models/ninfer/qwen3.8-27b-quasar"

ls -lh qwen3_8_27b_nvfp4.ninfer

printf '%s  %s\n' \
  '931816373707010b03e6e4dcba10f5265c3e820584dacd0eb2c6039e397045cd' \
  'qwen3_8_27b_nvfp4.ninfer' \
  | sha256sum --check
```

Expected result:

```text
qwen3_8_27b_nvfp4.ninfer: OK
```

Do not continue with a failed checksum. Delete the partial/corrupt file and download it again.

---

## 10. First smoke test: 4K context

Start with a small allocation to validate artifact loading, kernels, text generation, and MTP before attempting long context.

Open a second terminal and monitor the GPU:

```bash
watch -n 1 nvidia-smi
```

In the first terminal:

```bash
NINFER_HOME="$HOME/src/ninfer-quasar"
MODEL="$HOME/models/ninfer/qwen3.8-27b-quasar/qwen3_8_27b_nvfp4.ninfer"

"$NINFER_HOME/build/apps/ninfer" "$MODEL" \
  --prompt "Explain quantization-aware training in exactly three concise sentences." \
  --max-context 4096 \
  --max-new 256 \
  --kv-dtype int8 \
  --spec mtp \
  --draft-tokens 3 \
  --lm-head-draft
```

This is close to the model publisher's validated smoke-test profile.

What success looks like:

- the artifact passes identity and load-plan checks;
- the model weights load onto the RTX 5090 Laptop GPU;
- generation begins;
- the final answer is written to stdout;
- timing, memory, and speculative-decoding statistics appear on stderr;
- no CUDA out-of-memory, invalid-device-function, or artifact-profile error occurs.

### 10.1 Compare without MTP when diagnosing output

MTP should be treated as an optimization layer. To isolate the base model/runtime path, repeat the test without the three MTP flags:

```bash
"$NINFER_HOME/build/apps/ninfer" "$MODEL" \
  --prompt "Explain quantization-aware training in exactly three concise sentences." \
  --max-context 4096 \
  --max-new 256 \
  --kv-dtype int8
```

The no-MTP run also frees approximately 0.75 GiB of weight residency compared with the publisher's measured MTP configuration.

---

## 11. Start the local API server

The exact artifact was validated on a 24 GiB RTX 5090 Laptop with INT8 KV. The publisher measured:

- **15.31 GiB** GPU weight storage for text-only execution;
- **16.06 GiB** after materializing MTP;
- additional VRAM required for KV, runtime workspaces, state, and CUDA graphs.

Because the artifact has not received broad long-context validation, use staged context increases.

### 11.1 Conservative long-context starting profile: 96K

```bash
NINFER_HOME="$HOME/src/ninfer-quasar"
MODEL="$HOME/models/ninfer/qwen3.8-27b-quasar/qwen3_8_27b_nvfp4.ninfer"

"$NINFER_HOME/build/apps/ninfer-serve" "$MODEL" \
  --host 127.0.0.1 \
  --port 18080 \
  --model-id qwen3.8-27b-quasar-nvfp4 \
  --max-concurrency 1 \
  --max-context 98304 \
  --kv-capacity 98304 \
  --kv-dtype int8 \
  --device-state-slots 1 \
  --host-state-slots 2 \
  --host-kv-mib 2048 \
  --spec mtp \
  --draft-tokens 3 \
  --lm-head-draft \
  --default-max-tokens 8192
```

Important choices:

- `127.0.0.1` keeps the service local to the laptop.
- Port `18080` avoids colliding with a typical llama.cpp service on `8080`.
- One active request is appropriate for an interactive coding agent and minimizes fixed memory pressure.
- `98304` gives substantially more room than your previous 65K setup while retaining more safety margin than beginning at 128K.
- INT8 KV is the path explicitly mentioned in the artifact's 24 GiB validation.
- MTP uses three draft tokens, matching the publisher's smoke test.
- Vision is not loaded, because a text/code agent does not need its additional runtime residency.
- Historical thinking is not preserved automatically; this is desirable for conserving effective MCC context.

Keep this terminal open while testing. Stop the server with `Ctrl+C`.

---

## 12. Test the HTTP API

### 12.1 Health check

```bash
curl -fsS http://127.0.0.1:18080/health
echo
```

### 12.2 Discover the model alias

```bash
curl -fsS http://127.0.0.1:18080/v1/models | jq
```

### 12.3 OpenAI Chat Completions request

```bash
curl -fsS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-quasar-nvfp4",
    "messages": [
      {
        "role": "user",
        "content": "Reply with one sentence confirming that the local NInfer API works."
      }
    ],
    "max_tokens": 128,
    "stream": false
  }' | jq
```

NInfer also exposes OpenAI Responses and Anthropic Messages translations. The primary routes include:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/responses
POST /v1/messages
```

NInfer can parse and return tool calls, but it does not execute tools. MCC remains responsible for running tools and feeding their results back to the model.

---

## 13. Tune the context for 24 GiB VRAM

### 13.1 Recommended progression

Use this order:

| Stage | Context | MTP | Purpose |
|---|---:|---:|---|
| Smoke | 4,096 | On | Validate build and artifact |
| Baseline | 32,768 | On | Validate normal agent operation |
| Conservative long context | 98,304 | On | Recommended first daily profile |
| Target | 131,072 | On | Attempt after 96K is stable |
| Target fallback | 131,072 | Off | Trade generation speed for VRAM |
| Capacity probe | logical 262,144, KV `auto` | Off or on | Discover runtime-selected physical KV capacity |

The 96K and 128K recommendations are practical starting points, not published guarantees. Display usage, CUDA allocations, driver behavior, prefix-cache settings, and other processes alter the actual ceiling.

### 13.2 Try 128K

Change both values together:

```bash
--max-context 131072
--kv-capacity 131072
```

For the direct command:

```bash
"$NINFER_HOME/build/apps/ninfer-serve" "$MODEL" \
  --host 127.0.0.1 \
  --port 18080 \
  --model-id qwen3.8-27b-quasar-nvfp4 \
  --max-concurrency 1 \
  --max-context 131072 \
  --kv-capacity 131072 \
  --kv-dtype int8 \
  --device-state-slots 1 \
  --host-state-slots 2 \
  --host-kv-mib 2048 \
  --spec mtp \
  --draft-tokens 3 \
  --lm-head-draft \
  --default-max-tokens 8192
```

When 128K does not initialize reliably:

1. close other processes using the NVIDIA GPU;
2. retry without MTP;
3. return to 98,304;
4. if necessary, use 65,536.

A no-MTP 128K profile:

```bash
"$NINFER_HOME/build/apps/ninfer-serve" "$MODEL" \
  --host 127.0.0.1 \
  --port 18080 \
  --model-id qwen3.8-27b-quasar-nvfp4 \
  --max-concurrency 1 \
  --max-context 131072 \
  --kv-capacity 131072 \
  --kv-dtype int8 \
  --device-state-slots 1 \
  --host-state-slots 2 \
  --host-kv-mib 2048 \
  --default-max-tokens 8192
```

### 13.3 Probe automatic KV capacity

The pinned fork accepts:

```bash
--kv-capacity auto
```

It calculates the largest legal shared KV pool from memory remaining after model residency while keeping 1 GiB of automatic sizing headroom.

Probe with the model's logical ceiling:

```bash
"$NINFER_HOME/build/apps/ninfer-serve" "$MODEL" \
  --host 127.0.0.1 \
  --port 18080 \
  --model-id qwen3.8-27b-quasar-nvfp4 \
  --max-concurrency 1 \
  --max-context 262144 \
  --kv-capacity auto \
  --kv-dtype int8 \
  --device-state-slots 1 \
  --host-state-slots 2 \
  --host-kv-mib 2048 \
  --spec mtp \
  --draft-tokens 3 \
  --lm-head-draft \
  --default-max-tokens 8192
```

Read the startup log to find the resolved physical KV capacity. Then stop the probe and launch a fixed profile below that value. Rounding down by at least 8,192 tokens provides additional operational margin.

`--max-context` is the logical per-sequence ceiling. `--kv-capacity` is the physical shared KV pool. A logical 262K ceiling does not mean a 262K request will be admitted when the automatically selected physical pool is smaller.

### 13.4 Optional FP8 KV experiment

The fork also supports row-scaled FP8 KV and describes it as a compact long-context profile. After the validated INT8 path is stable, test:

```bash
--kv-dtype fp8
```

Evaluate both code quality and stability before making it the default. The QUASAR artifact's publication notes specifically describe the 24 GiB validation in terms of INT8 KV, so INT8 remains the conservative choice for the first deployment.

### 13.5 Context accounting matters at request time

A request must reserve room for both its input and requested output:

```text
input tokens + max_tokens <= max-context
```

For example, a 126K prompt plus `max_tokens: 8192` cannot fit in a 128K logical context. Coding-agent clients should reduce output allowance or compact the history before reaching the ceiling.

---

## 14. Recommended settings for MCC

For a single coding-agent session on the laptop:

| Setting | Recommendation | Reason |
|---|---|---|
| Context | 98,304 first; test 131,072 | Good gain without starting at the VRAM edge |
| KV type | INT8 | Explicit 24 GiB validation path |
| Concurrency | 1 | One interactive MCC run and minimum fixed pressure |
| Speculation | MTP, 3 draft tokens | Publisher-tested speed path |
| Vision | Off | Saves residency for a text/code workflow |
| Preserved historical thinking | Off | Keeps more context for code, tools, and repo data |
| Output default | 8,192 | Enough for coding while avoiding excessive reservation |
| Server bind | `127.0.0.1` | Prevents accidental network exposure |
| Port | `18080` | Avoids a common llama.cpp `8080` collision |

For MCC's provider configuration, use the equivalent of:

```text
Base URL: http://127.0.0.1:18080/v1
API key:  local
Model:    qwen3.8-27b-quasar-nvfp4
```

When NInfer is started without `--api-key`, it ignores the arbitrary client-side API key that some OpenAI-compatible libraries require to be non-empty.

Do not enable server-side `--preserve-thinking` for the normal MCC path unless MCC explicitly relies on raw historical reasoning blocks. Completed reasoning from old tool cycles can consume a large portion of the newly available context.

---

## 15. Install a reusable launcher

Create a launcher that defaults to 96K but can be changed through environment variables.

```bash
mkdir -p "$HOME/.local/bin"

cat > "$HOME/.local/bin/ninfer-qwen38-quasar" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

NINFER_HOME="${NINFER_HOME:-$HOME/src/ninfer-quasar}"
MODEL="${NINFER_MODEL:-$HOME/models/ninfer/qwen3.8-27b-quasar/qwen3_8_27b_nvfp4.ninfer}"

HOST="${NINFER_HOST:-127.0.0.1}"
PORT="${NINFER_PORT:-18080}"
MODEL_ID="${NINFER_MODEL_ID:-qwen3.8-27b-quasar-nvfp4}"

CONTEXT="${NINFER_CONTEXT:-98304}"
KV_CAPACITY="${NINFER_KV_CAPACITY:-$CONTEXT}"
KV_DTYPE="${NINFER_KV_DTYPE:-int8}"
DEFAULT_MAX_TOKENS="${NINFER_DEFAULT_MAX_TOKENS:-8192}"

HOST_STATE_SLOTS="${NINFER_HOST_STATE_SLOTS:-2}"
HOST_KV_MIB="${NINFER_HOST_KV_MIB:-2048}"
MTP="${NINFER_MTP:-1}"

BIN="$NINFER_HOME/build/apps/ninfer-serve"

if [[ ! -x "$BIN" ]]; then
  printf 'NInfer server binary not found or not executable: %s\n' "$BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  printf 'NInfer model artifact not found: %s\n' "$MODEL" >&2
  exit 1
fi

args=(
  "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --model-id "$MODEL_ID"
  --max-concurrency 1
  --max-context "$CONTEXT"
  --kv-capacity "$KV_CAPACITY"
  --kv-dtype "$KV_DTYPE"
  --device-state-slots 1
  --host-state-slots "$HOST_STATE_SLOTS"
  --host-kv-mib "$HOST_KV_MIB"
  --default-max-tokens "$DEFAULT_MAX_TOKENS"
)

if [[ "$MTP" == "1" ]]; then
  args+=(
    --spec mtp
    --draft-tokens 3
    --lm-head-draft
  )
fi

if [[ -n "${NINFER_API_KEY:-}" ]]; then
  args+=(--api-key "$NINFER_API_KEY")
fi

exec "$BIN" "${args[@]}"
BASH

chmod +x "$HOME/.local/bin/ninfer-qwen38-quasar"
```

Run the default 96K profile:

```bash
"$HOME/.local/bin/ninfer-qwen38-quasar"
```

Run 128K temporarily:

```bash
NINFER_CONTEXT=131072 \
NINFER_KV_CAPACITY=131072 \
"$HOME/.local/bin/ninfer-qwen38-quasar"
```

Run 128K without MTP:

```bash
NINFER_CONTEXT=131072 \
NINFER_KV_CAPACITY=131072 \
NINFER_MTP=0 \
"$HOME/.local/bin/ninfer-qwen38-quasar"
```

Probe automatic capacity:

```bash
NINFER_CONTEXT=262144 \
NINFER_KV_CAPACITY=auto \
"$HOME/.local/bin/ninfer-qwen38-quasar"
```

---

## 16. Optional systemd user service

A user service makes start/stop/log handling convenient without running NInfer as root.

### 16.1 Create the environment file

```bash
mkdir -p "$HOME/.config/ninfer"

cat > "$HOME/.config/ninfer/qwen38.env" <<'EOF'
NINFER_HOST=127.0.0.1
NINFER_PORT=18080
NINFER_MODEL_ID=qwen3.8-27b-quasar-nvfp4

NINFER_CONTEXT=98304
NINFER_KV_CAPACITY=98304
NINFER_KV_DTYPE=int8
NINFER_DEFAULT_MAX_TOKENS=8192

NINFER_HOST_STATE_SLOTS=2
NINFER_HOST_KV_MIB=2048
NINFER_MTP=1
EOF
```

Do not place a valuable network credential in this file for the local-only configuration. Binding to `127.0.0.1` is the primary protection here.

### 16.2 Create the unit

```bash
mkdir -p "$HOME/.config/systemd/user"

cat > "$HOME/.config/systemd/user/ninfer-qwen38.service" <<'EOF'
[Unit]
Description=NInfer Qwen3.8-27B QUASAR NVFP4
Documentation=https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer

[Service]
Type=simple
EnvironmentFile=%h/.config/ninfer/qwen38.env
Environment=CUDA_VISIBLE_DEVICES=0
WorkingDirectory=%h/src/ninfer-quasar
ExecStart=%h/.local/bin/ninfer-qwen38-quasar
Restart=on-failure
RestartSec=3
TimeoutStopSec=30

[Install]
WantedBy=default.target
EOF
```

Load and start it:

```bash
systemctl --user daemon-reload
systemctl --user start ninfer-qwen38.service
```

Inspect status and logs:

```bash
systemctl --user status ninfer-qwen38.service
journalctl --user -u ninfer-qwen38.service -f
```

Stop it:

```bash
systemctl --user stop ninfer-qwen38.service
```

Enable automatic start on login only after the chosen context profile is stable:

```bash
systemctl --user enable ninfer-qwen38.service
```

Disable automatic start:

```bash
systemctl --user disable ninfer-qwen38.service
```

After changing `~/.config/ninfer/qwen38.env`, restart:

```bash
systemctl --user restart ninfer-qwen38.service
```

---

## 17. API authentication and remote access

For use only from the same laptop, retain:

```bash
--host 127.0.0.1
```

That avoids exposing the model server on Wi-Fi, Ethernet, VPN, or other network interfaces.

When a bearer token is required, set one before launching:

```bash
export NINFER_API_KEY="$(openssl rand -hex 32)"
"$HOME/.local/bin/ninfer-qwen38-quasar"
```

Clients then send:

```http
Authorization: Bearer <the same value>
```

NInfer also accepts the Anthropic `x-api-key` form. The `/health` endpoint remains unauthenticated.

For access from another machine, prefer an SSH or Tailscale tunnel to changing the listener to `0.0.0.0`. Binding publicly should also involve an API key and host firewall policy.

---

## 18. Validate an agent workload, not only a chat prompt

The converted artifact was smoke-tested by its publisher, but broad accuracy and long-context evaluation were not performed on this exact conversion at publication time. Before replacing your llama.cpp baseline, run an MCC A/B comparison.

A useful local evaluation set should include:

1. repository exploration across several files;
2. a small bug fix with tests;
3. a refactor requiring cross-file consistency;
4. tool calls containing long command output;
5. a task that reaches at least 50-75% of the configured context;
6. structured output or patch generation;
7. a recovery case after a failed test.

Record:

- task success;
- invalid or malformed tool calls;
- patch correctness;
- test pass rate;
- time to first token;
- decode tokens per second;
- prompt/prefill time;
- MTP acceptance rate;
- peak VRAM;
- context at which failures begin.

For a fair diagnosis:

- compare MTP on and off;
- keep the same prompts and sampling configuration;
- separate quantization/model quality from runtime integration errors;
- do not conclude that 128K is reliable from startup alone—send genuinely long prompts.

---

## 19. Troubleshooting

### 19.1 `nvcc: command not found`

Find possible toolkit installations:

```bash
find /usr/local -maxdepth 3 -type f -name nvcc 2>/dev/null
```

When CUDA 13.2 is at `/usr/local/cuda-13.2`, export:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Then clean and reconfigure the build.

### 19.2 CMake finds an older CUDA toolkit

Inspect:

```bash
which nvcc
readlink -f "$(which nvcc)"
nvcc --version
```

Remove the cached build and pass the compiler explicitly:

```bash
cd "$HOME/src/ninfer-quasar"
rm -rf build

NVCC="$(readlink -f "$(command -v nvcc)")"

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES=120a
```

### 19.3 `NInfer requires CUDA 13.1 or newer`

The selected `nvcc` is too old, even when another toolkit is installed. Correct `PATH` or pass the desired CUDA 13.2 compiler with `-DCMAKE_CUDA_COMPILER`.

### 19.4 Architecture error mentioning `120a`

The pinned build accepts only:

```text
CMAKE_CUDA_ARCHITECTURES=120a
```

Delete `build/` and configure exactly as shown. Do not use `120`, `native`, `all`, or a list of architectures for this runtime.

### 19.5 FFmpeg or libcurl is not found

Reinstall the required development packages:

```bash
sudo apt install --reinstall \
  pkg-config \
  libavformat-dev \
  libavcodec-dev \
  libavutil-dev \
  libswscale-dev \
  libcurl4-openssl-dev
```

Verify:

```bash
pkg-config --modversion \
  libavformat libavcodec libavutil libswscale libcurl
```

### 19.6 Artifact identity, profile, or load-plan failure

Confirm both pins:

```bash
cd "$HOME/src/ninfer-quasar"
git rev-parse HEAD

cd "$HOME/models/ninfer/qwen3.8-27b-quasar"
sha256sum qwen3_8_27b_nvfp4.ninfer
```

Expected:

```text
Runtime: 2d11c9927be0ed8be20566814ec4bf97db2fab4f
Model:   931816373707010b03e6e4dcba10f5265c3e820584dacd0eb2c6039e397045cd
```

### 19.7 CUDA out of memory at startup

Check current usage:

```bash
nvidia-smi
```

Then apply these reductions in order:

1. close other GPU-heavy applications;
2. reduce 131,072 to 98,304;
3. disable MTP with `NINFER_MTP=0`;
4. reduce host/device checkpoint settings only when logs show those allocations are involved;
5. reduce to 65,536;
6. keep `--max-concurrency 1`;
7. do not enable `--vision`.

Do not expect CPU offload to rescue the configuration; NInfer does not provide it.

### 19.8 Server starts, but a long request is rejected

The input plus reserved output may exceed either:

- the per-sequence logical `--max-context`; or
- the shared physical `--kv-capacity`.

Reduce `max_tokens`, compact the conversation, or launch a larger fixed KV pool that fits VRAM.

### 19.9 Port already in use

```bash
ss -ltnp | grep ':18080'
```

Stop the existing process or select another port:

```bash
NINFER_PORT=18081 "$HOME/.local/bin/ninfer-qwen38-quasar"
```

### 19.10 systemd starts the binary but cannot find CUDA libraries

First test the launcher interactively. Then inspect:

```bash
journalctl --user -u ninfer-qwen38.service -n 200 --no-pager
ldd "$HOME/src/ninfer-quasar/build/apps/ninfer-serve"
```

When the toolkit libraries are not in the dynamic linker's normal paths, add the correct location to the service:

```ini
Environment=LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64
```

Then:

```bash
systemctl --user daemon-reload
systemctl --user restart ninfer-qwen38.service
```

Use the actual toolkit path on your system.

### 19.11 Strange output or apparent quality regression

Diagnose in this order:

1. repeat with MTP disabled;
2. use a deterministic/fixed sampling setup where supported;
3. compare thinking and non-thinking modes;
4. verify the model checksum and runtime commit;
5. test the same task against your known Qwen3.8 GGUF baseline;
6. distinguish context truncation from quantization quality.

The publisher explicitly states that broad accuracy and long-context testing had not yet been run on this converted artifact, so local A/B evaluation is necessary.

---

## 20. Safe update strategy

Do not run `git pull` inside the pinned source tree and assume the existing model remains compatible.

Keep the known-good installation:

```text
~/src/ninfer-quasar/
```

Test a new runtime in another directory:

```text
~/src/ninfer-quasar-next/
```

Before moving to a new revision:

1. read the artifact model card for an updated required commit;
2. verify whether the artifact format or profile changed;
3. perform a clean build;
4. rerun the checksum and smoke test;
5. rerun the MCC A/B workload;
6. keep the old launcher available for rollback.

After any CUDA toolkit or host-compiler change, remove the old CMake cache:

```bash
rm -rf "$HOME/src/ninfer-quasar/build"
```

Then configure and build again.

---

## 21. Removal

Stop and disable the optional service:

```bash
systemctl --user disable --now ninfer-qwen38.service 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/ninfer-qwen38.service"
systemctl --user daemon-reload
```

Remove the launcher and configuration:

```bash
rm -f "$HOME/.local/bin/ninfer-qwen38-quasar"
rm -rf "$HOME/.config/ninfer"
```

Remove source and model files only when no longer needed:

```bash
rm -rf "$HOME/src/ninfer-quasar"
rm -rf "$HOME/models/ninfer/qwen3.8-27b-quasar"
```

The Ubuntu packages installed in this tutorial are normal development dependencies and may be useful for other projects. Removing them is optional.

---

## 22. Final recommended profile

Begin daily MCC testing with:

```text
Model:             Qwen3.8-27B QUASAR NVFP4
Runtime:           pinned NInfer fork commit 2d11c992
Context:           98,304
Physical KV:       98,304
KV type:           INT8
Concurrency:       1
MTP:               enabled, 3 draft tokens
Vision:            disabled
Preserve thinking: disabled
Bind:              127.0.0.1:18080
```

After that profile survives realistic MCC workloads, test 131,072. Keep 96K as the known-good fallback and test 128K both with and without MTP.

---

## Sources

- QUASAR NInfer artifact and exact build/run instructions:  
  <https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer>

- RTX 5090 Mobile NInfer fork:  
  <https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile>

- Exact required runtime commit:  
  <https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile/commit/2d11c9927be0ed8be20566814ec4bf97db2fab4f>

- NInfer serving guide at the pinned commit:  
  <https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile/blob/2d11c9927be0ed8be20566814ec4bf97db2fab4f/docs/serving.md>

- NInfer build requirements at the pinned commit:  
  <https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile/blob/2d11c9927be0ed8be20566814ec4bf97db2fab4f/CMakeLists.txt>

- Upstream NInfer:  
  <https://github.com/Neroued/ninfer>

- Hugging Face CLI documentation:  
  <https://huggingface.co/docs/huggingface_hub/guides/cli>

- NVIDIA CUDA 13.2 documentation:  
  <https://docs.nvidia.com/cuda/archive/13.2.0/>

- Ubuntu 26.04 package index:  
  <https://packages.ubuntu.com/resolute/>
