# Deployment and Operations Guide

This document covers all deployment paths for Web Stable Diffusion: the recommended server-side Diffusers runtime, the browser-side WebGPU compilation path, and hybrid architectures that combine both.

---

## Server Deployment (Recommended)

The fastest way to generate real images is via the Python FastAPI backend powered by HuggingFace Diffusers.

### Quick Start

```bash
pip install -e ".[all]"
export OMNIMODAL_DIFFUSERS_MODEL=stabilityai/stable-diffusion-xl-base-1.0
uvicorn web_stable_diffusion.runtime.api:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

Build and run with Docker Compose (recommended):

```bash
docker compose up
```

Or build the image directly:

```bash
docker build -t web-sd .
docker run --gpus all -p 8000:8000 \
  -e OMNIMODAL_DIFFUSERS_MODEL=stabilityai/stable-diffusion-xl-base-1.0 \
  -v hf-cache:/root/.cache/huggingface \
  web-sd
```

The [`Dockerfile`](../Dockerfile) at the project root uses `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` as the base image and installs all dependencies from `requirements.txt` and `setup.py`.

The [`docker-compose.yml`](../docker-compose.yml) reserves an NVIDIA GPU, exposes port 8000, and mounts a named volume for the HuggingFace model cache so models are downloaded only once.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OMNIMODAL_DIFFUSERS_MODEL` | `stabilityai/stable-diffusion-xl-base-1.0` | HuggingFace model ID or local path |
| `OMNIMODAL_DIFFUSERS_DTYPE` | `float16` | Torch dtype for inference |
| `OMNIMODAL_DIFFUSERS_STEPS` | `25` | Number of inference steps |
| `OMNIMODAL_DIFFUSERS_GUIDANCE` | `7.5` | Classifier-free guidance scale |
| `OMNIMODAL_DIFFUSERS_NEGATIVE_PROMPT` | *(none)* | Default negative prompt |
| `OMNIMODAL_DIFFUSERS_LORA` | *(none)* | LoRA weights path or HuggingFace ID |
| `OMNIMODAL_TORCH_COMPILE` | `0` | Set to `1` to enable `torch.compile` |
| `OMNIMODAL_API_KEY` | *(none)* | API key for `X-API-Key` authentication |
| `OMNIMODAL_RATE_LIMIT` | `60` | Max requests per rate-limit period |
| `OMNIMODAL_RATE_PERIOD` | `60` | Rate-limit window in seconds |
| `OMNIMODAL_MANIFEST_DB` | `log_db/manifests.db` | SQLite path for manifest persistence |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate` | Generate image from text prompt (streaming JSONL) |
| `POST` | `/img2img` | Image-to-image generation |
| `POST` | `/cancel/{token_id}` | Cancel an in-progress generation |
| `GET` | `/manifests` | List recent generation manifests |
| `GET` | `/manifests/{token_id}` | Retrieve a specific manifest |
| `DELETE` | `/manifests/{token_id}` | Delete a manifest |
| `GET` | `/healthz` | Health check |

#### Example: text-to-image

```bash
curl -N -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat sitting on a rainbow"}'
```

The response is a stream of JSON Lines. Each line is either a progress event (with step number and preview) or the final manifest containing the base64-encoded image and metadata.

#### Example: authenticated request

```bash
curl -N -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "Mountain landscape at sunset"}'
```

---

## Browser WebGPU Deployment (Advanced)

The original MLC project compiles Stable Diffusion to run entirely in the browser via WebGPU. This path requires the TVM compilation toolchain and is best suited for offline demos or edge deployments.

### Prerequisites

The build toolchain requires Emscripten, Rust with the `wasm32-unknown-unknown` target, `wasm-pack`, Node.js, and Jekyll. Two provisioning methods are provided:

**Docker toolchain image:**

```bash
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.toolchain \
  -t websd/toolchain:latest .

docker run --rm -it \
  -v "$PWD":/workspace \
  -p 8889:8889 \
  websd/toolchain:latest
```

**Conda environment:**

```bash
mamba env create -f environments/toolchain.yml
mamba activate websd-toolchain
gem install jekyll jekyll-remote-theme
pip install -r requirements.txt
pip install mlc-ai-nightly -f https://mlc.ai/wheels
```

### Build Steps

1. Install TVM Unity via `pip3 install mlc-ai-nightly -f https://mlc.ai/wheels` or build from source (checkout `origin/unity`).

2. Import, optimize, and build the model:
   ```bash
   python3 build.py                    # Apple M-series (default)
   python3 build.py --target cuda      # CUDA GPUs
   ```

3. Prepare web dependencies:
   ```bash
   make deps
   ```

4. Compile to WebGPU/WASM:
   ```bash
   make webgpu
   ```

5. Serve the site:
   ```bash
   make serve
   ```

6. Open Chrome Canary at `http://localhost:8889/`. For best performance, launch with:
   ```bash
   /Applications/Google\ Chrome\ Canary.app/Contents/MacOS/Google\ Chrome\ Canary \
     --enable-dawn-features=disable_robustness
   ```

### Build Targets

```
make deps        # Download tvm.js, wasm tokenizer, and supporting assets
make webgpu      # Compile Stable Diffusion for the WebGPU runtime
make site        # Assemble the static site into site/dist/
make serve       # Serve the site locally via Jekyll
make package     # Emit CDN-friendly tarballs under dist/packages/
```

---

## Hybrid Architecture (Frontend + Backend)

The React single-page application under `web/` connects to the FastAPI streaming backend. This is the recommended setup for interactive use.

### How It Works

1. The FastAPI server runs the Diffusers pipeline and streams JSONL progress events.
2. The React frontend consumes the stream and renders previews in real time.
3. As each modality completes, the UI updates the canvas, audio player, video preview, and telemetry panel.

### Configuration

The frontend discovers the backend via one of these methods (in priority order):

1. `window.__omnimodalApiConfig = { baseUrl: "http://your-server:8000" }` in the HTML.
2. `localStorage` key `omnimodal.apiBaseUrl`.
3. Automatic default to `http://127.0.0.1:8000` when served from `localhost`.

If the backend is unreachable, the UI falls back to the legacy `tvmjsGlobalEnv` WebGPU pipeline.

### Running Locally

Terminal 1 (backend):
```bash
docker compose up
```

Terminal 2 (frontend):
```bash
cd web && npm install && npm start
```

Or serve the pre-built static site:
```bash
make site CONFIG=web/gh-page-config.json
npx http-server site/dist -p 4173
```

### Production Setup

For production, place the static frontend behind a CDN and point it at the FastAPI backend. Use a reverse proxy (nginx, Caddy, etc.) to handle TLS termination and CORS:

```nginx
server {
    listen 443 ssl;
    server_name sd.example.com;

    location / {
        root /var/www/websd/dist;
        try_files $uri $uri/ /index.html;
    }

    location /generate {
        proxy_pass http://127.0.0.1:8000;
        proxy_buffering off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }

    location /img2img { proxy_pass http://127.0.0.1:8000; }
    location /healthz { proxy_pass http://127.0.0.1:8000; }
    location /manifests { proxy_pass http://127.0.0.1:8000; }
}
```

---

## CI/CD Automation

### GitHub Actions Workflow

Save as `.github/workflows/site-deploy.yml`:

```yaml
name: Build and Deploy Web Stable Diffusion

on:
  push:
    branches: [ main ]
  workflow_dispatch:

env:
  EMSDK_VERSION: 3.1.45
  WASM_PACK_VERSION: 0.12.1
  RUST_TOOLCHAIN: stable

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - uses: mymindstorm/setup-emsdk@v12
        with:
          version: ${{ env.EMSDK_VERSION }}
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: ${{ env.RUST_TOOLCHAIN }}
          target: wasm32-unknown-unknown
          profile: minimal
      - name: Install wasm-pack
        run: cargo install wasm-pack --version ${{ env.WASM_PACK_VERSION }}
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
      - name: Install Ruby + Jekyll
        run: |
          sudo apt-get update
          sudo apt-get install -y ruby-full build-essential zlib1g-dev
          echo "GEM_HOME=$HOME/gems" >> $GITHUB_ENV
          echo "PATH=$HOME/gems/bin:$PATH" >> $GITHUB_PATH
          gem install bundler jekyll jekyll-remote-theme
      - name: Install Python dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install mlc-ai-nightly -f https://mlc.ai/wheels
      - name: Prepare toolchain assets
        run: make deps
      - name: Build WebGPU package
        run: make webgpu
      - name: Assemble static site
        run: make site CONFIG=web/gh-page-config.json
      - name: Archive site bundle
        uses: actions/upload-artifact@v4
        with:
          name: websd-site
          path: |
            dist
            site/dist
            site/stable-diffusion-config.json

  deploy:
    if: github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: websd-site
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: site/dist
          commit_message: Deploy Web Stable Diffusion site
```

---

## CDN Packaging

```bash
make package
aws s3 sync dist/packages s3://websd-artifacts/releases/$(git rev-parse --short HEAD)/
aws cloudfront create-invalidation --distribution-id <id> \
  --paths '/stable_diffusion.js' '/stable_diffusion_webgpu.wasm'
```

Version artifacts by commit SHA or semantic release. Store checksums alongside tarballs for provenance tracking:

```bash
shasum -a 256 dist/packages/*.tar.gz > dist/packages/sha256sum.txt
```

---

## Operational Readiness

### Health Checks

- **API health:** `curl -f http://localhost:8000/healthz` returns a 200 with pipeline status.
- **Edge health:** Monitor key static resources at 1-5 minute intervals.
- **Model availability:** Verify `dist/tokenizers-wasm` contains `pkg/package.json`.

### Rollback Strategy

1. Keep at least the last two releases in your CDN bucket (artifacts are named deterministically by `make package`).
2. Maintain a `rollback` CI job that re-publishes the previous artifact.
3. Store historical site configs in Git so rollbacks reuse the correct JSON.
4. Rerun smoke tests after any rollback before marking the incident resolved.
