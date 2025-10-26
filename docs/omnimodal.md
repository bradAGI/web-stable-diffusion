# OmniModalMiniturbo architecture

The `OmniModalMiniturbo` module implements a deterministic demonstration of an
"omni-modal" generator that follows the `NeuroSymbolicEngine` blueprint from the
project brief.  The intent is to provide an end-to-end but lightweight reference
that can be executed in unit tests, automation pipelines, and documentation
examples without downloading large diffusion checkpoints.

## Current capabilities

* **Symbolic reasoning** – The `MetaLogic` class provides lambda-calculus
  evaluation, forward-chaining deduction, and proof checking.  Prompt tokens are
  mapped into a small knowledge base that influences the numerical pipelines.
* **Modalities** – Audio waveforms, RGB images, volumetric tensors, and short
  video clips are generated in parallel threads.  Each modality uses seeded
  pseudo-randomness so results remain deterministic for a given prompt.
* **Device awareness** – Results are wrapped in `DeviceAwareResult`, allowing the
  payloads to live in PyTorch, TVM, WebGPU-emulated, or NumPy backends.  The
  selection happens automatically via `select_device()`.
* **CLI and automation** – `python -m web_stable_diffusion.cli.omnimodal` (or the
  installed `omnimodal-generate` entry point) generates every modality and writes
  compressed `.npz` files plus a `manifest.json` describing metadata.

## Limitations

This module is intentionally **not** a production-ready diffusion engine.  Key
limitations include:

* **Synthetic outputs** – The numerical pipelines are handcrafted to showcase the
  orchestration, not trained neural networks.  They are not photorealistic.
* **CPU-first execution** – Parallelism relies on Python thread pools, which can
  bottleneck under heavy CPU workloads.  GPU acceleration is only used when the
  selected backend automatically places NumPy arrays on CUDA devices.
* **No web integration yet** – The React site still renders results from the
  legacy Stable Diffusion pipeline.  Hooking the omni-modal outputs into the UI
  remains future work.
* **Minimal error handling** – The CLI validates command-line arguments but the
  deeper numerical routines do not yet reject malformed prompts or enforce strict
  resource budgets.

## Roadmap

To make the subsystem deployable in production scenarios, the following items are
being tracked:

1. Replace the synthetic generation stages with trained diffusion/decoder models
   and expose ONNX/TVM graphs for compilation.
2. Introduce multiprocessing or GPU-aware task schedulers to overlap modalities
   without Python GIL contention.
3. Provide REST and WebSocket APIs plus browser integrations that stream
   incremental results to the front-end.
4. Expand automated testing to include CLI smoke tests (added in this revision),
   golden manifests for regression detection, and UI automation via Playwright.
5. Document keyboard shortcuts, theming, and accessibility expectations alongside
   the existing dark-mode implementation in the React application.

Contributions toward this roadmap are welcome.  Please open an issue describing
use cases and constraints before submitting major architectural changes.
