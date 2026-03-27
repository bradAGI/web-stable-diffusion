/**
 * Sana 0.6B Browser Pipeline — runs entirely client-side via onnxruntime-web + WebGPU.
 *
 * Usage:
 *   const sana = new SanaPipeline();
 *   await sana.init("1024");  // or "2048", "4096"
 *   const imageData = await sana.generate("a cat on a rainbow", {
 *     negativePrompt: "blurry",
 *     steps: 20,
 *     onProgress: (step, total) => console.log(step, total),
 *   });
 *   ctx.putImageData(imageData, 0, 0);
 */

const MODEL_BASE_URL = "https://huggingface.co/brad-agi/sana-0.6b-onnx-webgpu/resolve/main";

const VARIANT_CONFIG = {
  "1024": { resolution: 1024, latentSize: 32, ditFile: "1024/sana_dit_1024.onnx", vaeFile: "1024/sana_vae_1024.onnx" },
  "2048": { resolution: 2048, latentSize: 64, ditFile: "2048/sana_dit_2048.onnx", vaeFile: "2048/sana_vae_2048.onnx" },
  "4096": { resolution: 4096, latentSize: 128, ditFile: "4096/sana_dit_4096.onnx", vaeFile: "4096/sana_vae_4096.onnx" },
};

class SanaPipeline {
  constructor() {
    this.ditSession = null;
    this.vaeSession = null;
    this.variant = null;
    this.config = null;
    this.ort = null;
  }

  /**
   * Initialize the pipeline for a given resolution variant.
   * Downloads and caches ONNX models on first use.
   */
  async init(variant = "1024", { onProgress } = {}) {
    if (!VARIANT_CONFIG[variant]) {
      throw new Error(`Unknown variant: ${variant}. Use "1024", "2048", or "4096".`);
    }
    this.config = VARIANT_CONFIG[variant];
    this.variant = variant;

    // Import onnxruntime-web with WebGPU support
    if (!this.ort) {
      if (typeof ort !== "undefined") {
        this.ort = ort;
      } else {
        throw new Error("onnxruntime-web not loaded. Include ort.webgpu.min.js before this script.");
      }
    }

    const sessionOptions = {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    };

    if (onProgress) onProgress("Loading DiT model...", 0, 2);

    // Load DiT (transformer / denoiser)
    const ditUrl = `${MODEL_BASE_URL}/${this.config.ditFile}`;
    console.log("Loading DiT from:", ditUrl);
    this.ditSession = await this.ort.InferenceSession.create(ditUrl, sessionOptions);

    if (onProgress) onProgress("Loading VAE decoder...", 1, 2);

    // Load VAE decoder
    const vaeUrl = `${MODEL_BASE_URL}/${this.config.vaeFile}`;
    console.log("Loading VAE from:", vaeUrl);
    this.vaeSession = await this.ort.InferenceSession.create(vaeUrl, sessionOptions);

    if (onProgress) onProgress("Ready!", 2, 2);
    console.log(`Sana pipeline initialized for ${variant}x${variant}`);
  }

  /**
   * Generate an image from a text prompt.
   * Returns an ImageData object that can be drawn to a canvas.
   */
  async generate(prompt, { negativePrompt = "", steps = 20, guidanceScale = 7.5, seed = null, onProgress } = {}) {
    if (!this.ditSession || !this.vaeSession) {
      throw new Error("Pipeline not initialized. Call init() first.");
    }

    const { resolution, latentSize } = this.config;
    const latentChannels = 32; // Sana uses 32 latent channels with 32x compression

    // Step 1: Create random latent noise
    if (onProgress) onProgress("Generating latent noise...", 0, steps + 1);
    const latent = this._createNoise(1, latentChannels, latentSize, latentSize, seed);

    // Step 2: Create simple text embedding (placeholder — full pipeline needs Gemma encoder)
    // For now, create a deterministic embedding from the prompt hash
    const textEmbedding = this._promptToEmbedding(prompt, 300, 2304);

    // Step 3: Denoising loop (DiT)
    let currentLatent = latent;
    for (let step = 0; step < steps; step++) {
      if (onProgress) onProgress(`Denoising step ${step + 1}/${steps}`, step + 1, steps + 1);

      const timestep = this._getTimestep(step, steps);

      // Run DiT inference
      const feeds = {
        hidden_states: new this.ort.Tensor("float16", currentLatent, [1, latentChannels, latentSize, latentSize]),
        encoder_hidden_states: new this.ort.Tensor("float16", textEmbedding, [1, 300, 2304]),
        timestep: new this.ort.Tensor("float16", new Float32Array([timestep]), [1]),
      };

      try {
        const ditOutput = await this.ditSession.run(feeds);
        const noisePred = Object.values(ditOutput)[0];

        // Simple Euler step: latent = latent - noise_pred * dt
        const dt = 1.0 / steps;
        const noisePredData = noisePred.data;
        for (let i = 0; i < currentLatent.length; i++) {
          currentLatent[i] = currentLatent[i] - noisePredData[i] * dt;
        }
      } catch (e) {
        console.error(`DiT inference failed at step ${step}:`, e);
        throw e;
      }
    }

    // Step 4: VAE decode
    if (onProgress) onProgress("Decoding image...", steps, steps + 1);
    const vaeFeed = {
      latent: new this.ort.Tensor("float32", new Float32Array(currentLatent), [1, latentChannels, latentSize, latentSize]),
    };

    try {
      const vaeOutput = await this.vaeSession.run(vaeFeed);
      const imageData = Object.values(vaeOutput)[0];

      // Convert to ImageData for canvas
      if (onProgress) onProgress("Done!", steps + 1, steps + 1);
      return this._tensorToImageData(imageData.data, resolution, resolution);
    } catch (e) {
      console.error("VAE decode failed:", e);
      throw e;
    }
  }

  /**
   * Create random noise tensor.
   */
  _createNoise(batch, channels, height, width, seed) {
    const size = batch * channels * height * width;
    const data = new Float32Array(size);
    // Simple seeded PRNG (xorshift32)
    let s = seed || (Date.now() & 0xffffffff);
    if (s === 0) s = 1;
    const scale = Math.sqrt(12) / 2; // Match uniform distribution variance
    for (let i = 0; i < size; i++) {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      data[i] = ((s >>> 0) / 0xffffffff - 0.5) * 2 * scale;
    }
    return data;
  }

  /**
   * Convert prompt to a simple deterministic embedding.
   * NOTE: This is a placeholder. The real pipeline needs the Gemma text encoder.
   * For proper text conditioning, the Gemma encoder must run first (either
   * server-side or via a separate ONNX model in the browser).
   */
  _promptToEmbedding(prompt, seqLen, hiddenSize) {
    const size = seqLen * hiddenSize;
    const data = new Float32Array(size);
    // Hash-based deterministic embedding
    let hash = 0;
    for (let i = 0; i < prompt.length; i++) {
      hash = ((hash << 5) - hash + prompt.charCodeAt(i)) | 0;
    }
    let s = hash || 1;
    for (let i = 0; i < size; i++) {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      data[i] = (s >>> 0) / 0xffffffff * 0.02 - 0.01;
    }
    return data;
  }

  /**
   * Get timestep value for a given step in the flow schedule.
   */
  _getTimestep(step, totalSteps) {
    // Linear flow schedule: t goes from 1.0 to 0.0
    return 1.0 - (step / totalSteps);
  }

  /**
   * Convert a CHW float tensor to canvas ImageData.
   */
  _tensorToImageData(data, width, height) {
    const imageData = new ImageData(width, height);
    const pixels = imageData.data; // RGBA uint8
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIdx = (y * width + x) * 4;
        // CHW format: data[c * H * W + y * W + x]
        const r = data[0 * height * width + y * width + x];
        const g = data[1 * height * width + y * width + x];
        const b = data[2 * height * width + y * width + x];
        pixels[pixelIdx + 0] = Math.max(0, Math.min(255, Math.round((r * 0.5 + 0.5) * 255)));
        pixels[pixelIdx + 1] = Math.max(0, Math.min(255, Math.round((g * 0.5 + 0.5) * 255)));
        pixels[pixelIdx + 2] = Math.max(0, Math.min(255, Math.round((b * 0.5 + 0.5) * 255)));
        pixels[pixelIdx + 3] = 255;
      }
    }
    return imageData;
  }

  /**
   * Dispose of ONNX sessions to free GPU memory.
   */
  async dispose() {
    if (this.ditSession) { await this.ditSession.release(); this.ditSession = null; }
    if (this.vaeSession) { await this.vaeSession.release(); this.vaeSession = null; }
  }
}

// Export for use in browser
if (typeof window !== "undefined") {
  window.SanaPipeline = SanaPipeline;
}
