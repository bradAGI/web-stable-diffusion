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
  video clips are generated in parallel workers.  The scheduler automatically
  selects thread or process pools depending on the backend so CPU-heavy
  workloads can scale beyond the Python GIL.  Each modality uses seeded
  pseudo-randomness so results remain deterministic for a given prompt.
* **Device awareness** – Results are wrapped in `DeviceAwareResult`, allowing the
  payloads to live in PyTorch, TVM, WebGPU-emulated, or NumPy backends.  The
  selection happens automatically via `select_device()`.
* **CLI and automation** – `python -m web_stable_diffusion.cli.omnimodal` (or the
  installed `omnimodal-generate` entry point) generates every modality, enforces
  argument validation, surfaces structured logging, and writes compressed `.npz`
  files plus a `manifest.json` describing metadata and per-modality timing
  metrics.

## Limitations

This module is intentionally **not** a production-ready diffusion engine.  Key
limitations include:

* **Synthetic outputs** – The numerical pipelines are handcrafted to showcase the
  orchestration, not trained neural networks.  They are not photorealistic.
* **Backend-sensitive scheduling** – Multiprocessing is only enabled when the
  NumPy backend is active.  GPU-backed tensors continue to use thread pools to
  avoid cross-process serialisation, so GIL contention can still surface in
  those scenarios.
* **No web integration yet** – The React site still renders results from the
  legacy Stable Diffusion pipeline.  Hooking the omni-modal outputs into the UI
  remains future work.
* **Guardrails still evolving** – Core generation routines now validate prompt
  and dimension parameters, yet advanced resource controls (memory caps,
  streaming cancellation) remain future work.

## Roadmap

To make the subsystem deployable in production scenarios, the following items are
being tracked:

1. Replace the synthetic generation stages with trained diffusion/decoder models
   and expose ONNX/TVM graphs for compilation.
2. Expand GPU-aware task schedulers to overlap modalities without falling back to
   threads when CUDA/TVM backends are active.
3. Provide REST and WebSocket APIs plus browser integrations that stream
   incremental results to the front-end.
4. Expand automated testing to include golden manifests for regression detection
   and UI automation via Playwright.
5. Document keyboard shortcuts, theming, and accessibility expectations alongside
   the existing dark-mode implementation in the React application.

Contributions toward this roadmap are welcome.  Please open an issue describing
use cases and constraints before submitting major architectural changes.
