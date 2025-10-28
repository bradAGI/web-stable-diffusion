# Interface Customization & Shortcuts

This guide describes every interactive customization hook exposed by the Web Stable Diffusion UI, including keyboard shortcuts, theming controls, and runtime APIs for automations.

## Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| `Ctrl`/`⌘` + `Enter` | Trigger image generation using the current prompt. |
| `Shift` + `?` | Open the help dialog that summarises shortcuts and contextual tips. |
| `Alt` + `T` | Cycle through the available theme presets (Light → Dark → Custom). |
| `Alt` + `L` | Toggle between the available control layouts (Classic, Compact, Adaptive). |
| `Alt` + `\`` | Pause or resume live log updates in the advanced logging panel. |

All shortcut handlers respect the focused element and never steal focus from text inputs. Screen reader announcements mirror shortcut-triggered changes, ensuring the UI remains navigable without a pointer.

## Theming controls

The app exposes three theme presets—Light, Dark, and Custom. Selecting **Custom** unlocks direct hue configuration:

* **Accent colour picker** — Updates the accent hue in real time and persists to `localStorage` (`wsd-accent`). The app derives a full palette from this hue, including focus rings, gradients, and outline tones.
* **Background & surface pickers** — Control the page background and card surface colours (`wsd-custom-background`, `wsd-custom-surface`).
* **Hue-based design tokens** — Each theme writes hue (`--accent-hue`), saturation, and lightness tokens to `document.documentElement`. CSS design tokens reuse these values to produce gradients, subtle fills, and accessible borders.
* **Motion guidelines** — Motion tokens (`--motion-duration-*`, `--motion-ease-*`) govern the cadence of focus rings, status timelines, and progress bars. Users who prefer reduced motion inherit near-instant transitions via the `prefers-reduced-motion` media query.

Changes persist between sessions and propagate immediately without reloading the page.

## Control layout options

Use the **Control layout** field to switch between:

* **Classic** — A two-column layout with equal weight on form controls and the canvas preview.
* **Compact** — A single-column layout optimised for narrow screens and assistive technologies.
* **Adaptive** — A wide visualization column with condensed controls, ideal for monitoring timelines and logs.

The selected mode is stored as `wsd-control-layout` and reflected on the `<html>` dataset (`data-layout`). Developers can target these hooks to tailor additional UI modules.

Keyboard shortcut `Alt + L` mirrors the UI toggle, providing accessible layout changes without pointer interaction.

## Customization APIs

The app exposes a handful of automation-friendly APIs via `window.tvmjsGlobalEnv`:

* `getPrompt()` / `getNegativePrompt()` — Retrieve the current prompt values.
* `setPrompts({ prompt, negativePrompt })` — Replace prompts programmatically.
* `getSchedulerId()` / `setSchedulerId(value)` — Query or set the active scheduler.
* `getVaeCycle()` — Return the configured VAE cadence.
* `updateProgress({ text, progress, stage })` — Stream progress updates. Each call updates the animated status timeline and the progress bar.
* `updateGpuStatus(message)` — Push telemetry messages into the GPU status block.
* `logMessage(message, level)` — Append entries to the advanced logging panel. Supported levels are `info`, `warn`, and `error`.
* `onSchedulerFallback(payload)` — Notify the UI when a scheduler fallback occurs.
* `reportValidationError({ field, message })` / `clearValidationErrors()` — Surface validation feedback inline.
* `onGenerationLifecycle(stage)` — Valid stages include `start`, `validation-error`, `error`, `complete`, and `end`. Lifecycle updates reset the timeline, drive announcements, and control the animated status marker.
* `isStreamingEnabled` — Boolean flag indicating whether the UI has resolved a streaming API base URL.

Automation clients can combine these APIs with the layout dataset attributes to keep the UI in sync with remote workflows.

## Streaming integration

By default the UI attempts to stream results from the FastAPI service exposed by
`web_stable_diffusion.runtime.api`. The target URL can be controlled by:

* Setting `window.__omnimodalApiConfig = { baseUrl: "http://host:port" }` before
  the app mounts.
* Persisting `omnimodal.apiBaseUrl` in `localStorage` (the value is read on
  startup).

When the page is served from `localhost` or `127.0.0.1`, the UI falls back to
`http://127.0.0.1:8000` automatically. If the streaming handshake fails the app
logs the error, announces a fallback banner, and resumes using the legacy
`tvmjsGlobalEnv` hooks so that demos continue to work without the API.
