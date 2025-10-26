# Deployment and Operations Guide

This document captures a production-oriented workflow for building, testing, and shipping the Web Stable Diffusion demo. It consolidates container definitions, scripted entry points, CI/CD automation, CDN packaging, and operational guard rails.

## 1. Toolchain Containers and Conda Environments

### Docker toolchain image

The [`docker/Dockerfile.toolchain`](../docker/Dockerfile.toolchain) image provisions all build-time dependencies including Emscripten, Rust, `wasm-pack`, and Jekyll.

```
# Build the toolchain image
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.toolchain \
  -t websd/toolchain:latest .

# Start an interactive shell with the project mounted
docker run --rm -it \
  -v "$PWD":/workspace \
  -p 8889:8889 \
  websd/toolchain:latest
```

The container sets up `emsdk_env.sh`, the Rust `wasm32-unknown-unknown` target, the tokenizer build chain, and pre-downloads the tvm.js runtime. After launching the container, the `make` targets described below are immediately available.

### Conda environment

For teams standardising on Conda/Mamba, [`environments/toolchain.yml`](../environments/toolchain.yml) mirrors the toolchain.

```
# Create and activate the environment
mamba env create -f environments/toolchain.yml
mamba activate websd-toolchain

# Finish the Ruby portion of the toolchain
bundle config set path "$CONDA_PREFIX/lib/ruby/gems"  # optional vendor location
gem install jekyll jekyll-remote-theme

# Install the project's Python dependencies that require an external wheel index
pip install -r requirements.txt
pip install mlc-ai-nightly -f https://mlc.ai/wheels
```

The environment pins the versions of Emscripten, Rust, `wasm-pack`, and includes Jekyll via `conda-forge`. Ruby gems are installed into the active environment to keep the runtime self-contained.

## 2. Scripted build entry points

The project now exposes a Makefile with opinionated targets:

```
make deps        # Download tvm.js, wasm tokenizer, and supporting assets
make webgpu      # Compile Stable Diffusion for the WebGPU runtime
make site        # Assemble the static site into site/dist (CONFIG=web/gh-page-config.json overrides)
make serve       # Serve the site locally via Jekyll (PORT=8890 overrides)
make package     # Emit CDN-friendly tarballs under dist/packages/
```

All targets can be chained by CI/CD pipelines or executed locally. `make serve` blocks with a long-running Jekyll process so it should be run in a dedicated terminal or supervisor.

## 3. CI/CD automation template

The workflow below automates builds on GitHub Actions using the scripted entry points and publishes to GitHub Pages. Save it as [`.github/workflows/site-deploy.yml`](../.github/workflows/site-deploy.yml).

```
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
      - name: Smoke-test deployed CDN
        run: |
          curl -f https://websd.example.com/stable_diffusion.js
          curl -f https://websd.example.com/stable_diffusion_webgpu.wasm
```

### Pipeline notes

- Use repository secrets for any access tokens (e.g. `AWS_ACCESS_KEY_ID`) referenced by additional steps.
- The build job archives both the compiled wasm artifacts (`dist/`) and the static site (`site/dist`). The deploy job retrieves the artifacts and pushes to GitHub Pages (or any other Git ref you configure).
- The smoke tests perform HTTP health checks by requesting the JavaScript harness and the primary WASM module. Replace the URLs with your CDN hostnames.

## 4. CDN packaging & promotion

The `make package` target creates two tarballs under `dist/packages/`:

- `websd-site.tar.gz` – static site assets ready to upload to object storage or a CDN origin.
- `websd-model.tar.gz` – compiled WebGPU binaries and schedulers that can be versioned independently.

Example upload and promotion flow:

```
make package
aws s3 sync dist/packages s3://websd-artifacts/releases/$(git rev-parse --short HEAD)/
aws cloudfront create-invalidation --distribution-id <id> --paths '/stable_diffusion.js' '/stable_diffusion_webgpu.wasm'
```

Version the artifacts by commit SHA or semantic release. Store checksums alongside the tarballs (`shasum -a 256 dist/packages/*.tar.gz > dist/packages/sha256sum.txt`) to support provenance tracking.

## 5. Operational readiness

### Health checks

- **Edge health** – Use `curl -f` (as in the CI smoke tests) against key static resources. Automate these with your monitoring platform at 1–5 minute intervals.
- **Model availability** – Verify that `dist/tokenizers-wasm` is reachable and contains `pkg/package.json`. Static site monitors can fetch the JSON and validate the `version` field against the expected release.
- **Browser sanity** – Run `npm install -g @web/test-runner` and script a headless WebGPU smoke test that loads the index.html via `npx http-server site/dist -p 4173` and performs a prompt round-trip. Integrate the test into nightly CI.

### Rollback strategy

1. **Immutable artifacts** – Because `make package` names outputs deterministically, keep at least the last two releases in your CDN bucket.
2. **Automated rollback** – Maintain a `rollback` job in CI that downloads the previous artifact (tracked via a manifest or Git tag) and republishes it to your hosting target.
3. **Versioned config** – The site configuration (`site/stable-diffusion-config.json`) is built from the `CONFIG` file passed to `make site`. Store historical configs in Git so a rollback reuses the previous JSON without manual edits.
4. **Verification gate** – After any rollback, rerun the smoke tests and confirm logs via your CDN provider. Only mark the incident resolved once the checks succeed.

With these assets, teams can reproduce the build, ship to staging or production, monitor the deployment, and revert quickly if necessary.
