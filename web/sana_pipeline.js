/**
 * Sana 0.6B Browser Pipeline — runs entirely client-side via onnxruntime-web + WebGPU.
 *
 * Pipeline: Tokenizer → Gemma Text Encoder → DiT Denoiser → DC-AE Decoder → Canvas
 * All models loaded from HuggingFace CDN and cached in browser.
 */

const MODEL_BASE_URL = "https://huggingface.co/brad-agi/sana-0.6b-onnx-webgpu/resolve/main";

const TEXT_ENCODER_FILES = {
  "int8": { file: "sana_text_encoder_int8.onnx", size: "2.5 GB", label: "Faster download" },
  "fp16": { file: "sana_text_encoder.onnx", size: "4.9 GB", label: "Best quality" },
};
const TOKENIZER_URL = `${MODEL_BASE_URL}/tokenizer/tokenizer.json`;

const VARIANT_CONFIG = {
  "1024": { resolution: 1024, latentSize: 32, latentChannels: 32, ditFile: "1024/sana_dit_1024.onnx", vaeFile: "1024/sana_vae_1024.onnx" },
  "2048": { resolution: 2048, latentSize: 64, latentChannels: 32, ditFile: "2048/sana_dit_2048.onnx", vaeFile: "2048/sana_vae_2048.onnx" },
  "4096": { resolution: 4096, latentSize: 128, latentChannels: 32, ditFile: "4096/sana_dit_4096.onnx", vaeFile: "4096/sana_vae_4096.onnx" },
};

class SanaPipeline {
  constructor() {
    this.textEncoderSession = null;
    this.ditSession = null;
    this.vaeSession = null;
    this.tokenizer = null;
    this.variant = null;
    this.config = null;
    this.ort = null;
  }

  /**
   * Initialize the pipeline for a given resolution variant.
   * @param variant - "1024", "2048", or "4096"
   * @param options.quality - "int8" (2.5 GB, fast) or "fp16" (4.9 GB, best quality)
   * @param options.onProgress - progress callback
   */
  async init(variant = "1024", { quality = "int8", onProgress } = {}) {
    if (!VARIANT_CONFIG[variant]) {
      throw new Error(`Unknown variant: ${variant}. Use "1024", "2048", or "4096".`);
    }
    this.config = VARIANT_CONFIG[variant];
    this.variant = variant;

    if (!this.ort) {
      if (typeof ort !== "undefined") {
        this.ort = ort;
      } else {
        throw new Error("onnxruntime-web not loaded. Include ort.webgpu.min.js before this script.");
      }
    }

    const sessionOptions = {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all",
    };

    const steps = 4; // tokenizer + text encoder + dit + vae
    let step = 0;

    // Load tokenizer
    if (!this.tokenizer) {
      if (onProgress) onProgress("Loading tokenizer...", step, steps);
      console.log("Loading tokenizer from:", TOKENIZER_URL);
      const resp = await fetch(TOKENIZER_URL);
      this.tokenizer = await resp.json();
      step++;
    }

    // Load text encoder (shared across resolutions, reload if quality changes)
    const encConfig = TEXT_ENCODER_FILES[quality] || TEXT_ENCODER_FILES["int8"];
    if (!this.textEncoderSession || this._textEncoderQuality !== quality) {
      if (this.textEncoderSession) await this.textEncoderSession.release();
      if (onProgress) onProgress(`Loading text encoder (${encConfig.size}, ${encConfig.label})...`, step, steps);
      const textEncUrl = `${MODEL_BASE_URL}/${encConfig.file}`;
      console.log("Loading text encoder from:", textEncUrl);
      this.textEncoderSession = await this.ort.InferenceSession.create(textEncUrl, sessionOptions);
      this._textEncoderQuality = quality;
      step++;
    }

    // Load DiT (resolution-specific)
    if (onProgress) onProgress(`Loading DiT for ${variant}×${variant}...`, step, steps);
    const ditUrl = `${MODEL_BASE_URL}/${this.config.ditFile}`;
    console.log("Loading DiT from:", ditUrl);
    if (this.ditSession) await this.ditSession.release();
    this.ditSession = await this.ort.InferenceSession.create(ditUrl, sessionOptions);
    step++;

    // Load VAE (resolution-specific)
    if (onProgress) onProgress(`Loading VAE decoder...`, step, steps);
    const vaeUrl = `${MODEL_BASE_URL}/${this.config.vaeFile}`;
    console.log("Loading VAE from:", vaeUrl);
    if (this.vaeSession) await this.vaeSession.release();
    this.vaeSession = await this.ort.InferenceSession.create(vaeUrl, sessionOptions);
    step++;

    if (onProgress) onProgress("Ready!", steps, steps);
    console.log(`Sana pipeline initialized for ${variant}×${variant}`);
  }

  /**
   * Generate an image from a text prompt.
   * Returns an ImageData object for canvas rendering.
   */
  async generate(prompt, { negativePrompt = "", steps = 20, guidanceScale = 7.5, seed = null, onProgress } = {}) {
    if (!this.ditSession || !this.vaeSession || !this.textEncoderSession) {
      throw new Error("Pipeline not initialized. Call init() first.");
    }

    const { resolution, latentSize, latentChannels } = this.config;
    const totalSteps = steps + 2; // +1 for text encoding, +1 for VAE decode

    // Step 1: Tokenize + encode text
    if (onProgress) onProgress("Encoding text...", 0, totalSteps);
    const textEmbedding = await this._encodeText(prompt);

    // Step 2: Create random latent noise
    const latent = this._createNoise(1, latentChannels, latentSize, latentSize, seed);

    // Step 3: Denoising loop (DiT)
    let currentLatent = new Float32Array(latent);
    for (let step = 0; step < steps; step++) {
      if (onProgress) onProgress(`Denoising step ${step + 1}/${steps}`, step + 1, totalSteps);

      const timestep = this._getTimestep(step, steps);

      const feeds = {};
      // Map to the actual input names from the exported model
      const inputNames = this.ditSession.inputNames;
      for (const name of inputNames) {
        if (name === "hidden_states" || name.includes("hidden")) {
          feeds[name] = new this.ort.Tensor("float16", new Uint16Array(this._f32ToF16(currentLatent)), [1, latentChannels, latentSize, latentSize]);
        } else if (name === "encoder_hidden_states" || name.includes("encoder")) {
          feeds[name] = textEmbedding;
        } else if (name === "timestep" || name.includes("time")) {
          feeds[name] = new this.ort.Tensor("float16", new Uint16Array(this._f32ToF16(new Float32Array([timestep]))), [1]);
        }
      }

      try {
        const ditOutput = await this.ditSession.run(feeds);
        const noisePred = Object.values(ditOutput)[0];
        const noisePredData = noisePred.cpuData || noisePred.data;

        // Euler step: latent = latent - noise_pred * dt
        const dt = 1.0 / steps;
        if (noisePredData.constructor === Uint16Array) {
          // fp16 output — convert back
          const f32Noise = this._f16ToF32(noisePredData);
          for (let i = 0; i < currentLatent.length; i++) {
            currentLatent[i] -= f32Noise[i] * dt;
          }
        } else {
          for (let i = 0; i < currentLatent.length; i++) {
            currentLatent[i] -= noisePredData[i] * dt;
          }
        }
      } catch (e) {
        console.error(`DiT inference failed at step ${step}:`, e);
        throw e;
      }
    }

    // Step 4: VAE decode
    if (onProgress) onProgress("Decoding image...", steps + 1, totalSteps);
    const vaeInputNames = this.vaeSession.inputNames;
    const vaeFeed = {};
    vaeFeed[vaeInputNames[0]] = new this.ort.Tensor("float32", currentLatent, [1, latentChannels, latentSize, latentSize]);

    try {
      const vaeOutput = await this.vaeSession.run(vaeFeed);
      const imageData = Object.values(vaeOutput)[0];

      if (onProgress) onProgress("Done!", totalSteps, totalSteps);
      return this._tensorToImageData(imageData.cpuData || imageData.data, resolution, resolution);
    } catch (e) {
      console.error("VAE decode failed:", e);
      throw e;
    }
  }

  /**
   * Encode text prompt using the Gemma text encoder.
   */
  async _encodeText(prompt) {
    // Simple tokenization using the loaded tokenizer.json
    const tokens = this._tokenize(prompt);
    const maxLen = 300;
    const inputIds = new BigInt64Array(maxLen);
    const attentionMask = new BigInt64Array(maxLen);

    for (let i = 0; i < Math.min(tokens.length, maxLen); i++) {
      inputIds[i] = BigInt(tokens[i]);
      attentionMask[i] = 1n;
    }
    // Pad remaining with 0
    for (let i = tokens.length; i < maxLen; i++) {
      inputIds[i] = 0n;
      attentionMask[i] = 0n;
    }

    const feeds = {};
    const inputNames = this.textEncoderSession.inputNames;
    for (const name of inputNames) {
      if (name === "input_ids" || name.includes("input")) {
        feeds[name] = new this.ort.Tensor("int64", inputIds, [1, maxLen]);
      } else if (name === "attention_mask" || name.includes("mask")) {
        feeds[name] = new this.ort.Tensor("int64", attentionMask, [1, maxLen]);
      }
    }

    const output = await this.textEncoderSession.run(feeds);
    return Object.values(output)[0]; // hidden_states tensor
  }

  /**
   * Simple BPE tokenizer using the tokenizer.json vocabulary.
   */
  _tokenize(text) {
    if (!this.tokenizer || !this.tokenizer.model || !this.tokenizer.model.vocab) {
      // Fallback: basic character-level tokenization
      console.warn("Tokenizer vocab not found, using fallback");
      const tokens = [2]; // BOS token
      for (let i = 0; i < text.length && tokens.length < 299; i++) {
        tokens.push(text.charCodeAt(i) % 30000 + 100);
      }
      return tokens;
    }

    // Use the vocabulary to do simple word-level tokenization
    const vocab = this.tokenizer.model.vocab;
    const tokens = [2]; // BOS token for Gemma

    // Simple whitespace + subword tokenization
    const words = text.toLowerCase().split(/\s+/);
    for (const word of words) {
      const prefixed = "▁" + word; // Gemma uses sentencepiece with ▁ prefix
      if (vocab[prefixed] !== undefined) {
        tokens.push(vocab[prefixed]);
      } else {
        // Character fallback
        for (const char of word) {
          const charToken = vocab["▁" + char] || vocab[char];
          if (charToken !== undefined) {
            tokens.push(charToken);
          }
        }
      }
    }
    return tokens;
  }

  /** Create random noise tensor with seeded PRNG. */
  _createNoise(batch, channels, height, width, seed) {
    const size = batch * channels * height * width;
    const data = new Float32Array(size);
    let s = seed || (Date.now() & 0xffffffff);
    if (s === 0) s = 1;
    const scale = Math.sqrt(12) / 2;
    for (let i = 0; i < size; i++) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      data[i] = ((s >>> 0) / 0xffffffff - 0.5) * 2 * scale;
    }
    return data;
  }

  /** Flow schedule timestep. */
  _getTimestep(step, totalSteps) {
    return 1.0 - (step / totalSteps);
  }

  /** Convert Float32Array to Float16 (Uint16Array). */
  _f32ToF16(f32arr) {
    const u16 = new Uint16Array(f32arr.length);
    for (let i = 0; i < f32arr.length; i++) {
      const f = f32arr[i];
      const view = new DataView(new ArrayBuffer(4));
      view.setFloat32(0, f);
      const bits = view.getUint32(0);
      const sign = (bits >> 16) & 0x8000;
      const exp = ((bits >> 23) & 0xff) - 127 + 15;
      const mantissa = (bits >> 13) & 0x3ff;
      if (exp <= 0) u16[i] = sign;
      else if (exp >= 31) u16[i] = sign | 0x7c00;
      else u16[i] = sign | (exp << 10) | mantissa;
    }
    return u16;
  }

  /** Convert Float16 (Uint16Array) to Float32Array. */
  _f16ToF32(u16arr) {
    const f32 = new Float32Array(u16arr.length);
    for (let i = 0; i < u16arr.length; i++) {
      const h = u16arr[i];
      const sign = (h & 0x8000) >> 15;
      const exp = (h & 0x7c00) >> 10;
      const mantissa = h & 0x03ff;
      let f;
      if (exp === 0) f = mantissa === 0 ? 0 : Math.pow(2, -14) * (mantissa / 1024);
      else if (exp === 31) f = mantissa === 0 ? Infinity : NaN;
      else f = Math.pow(2, exp - 15) * (1 + mantissa / 1024);
      f32[i] = sign ? -f : f;
    }
    return f32;
  }

  /** Convert CHW float tensor to RGBA ImageData for canvas. */
  _tensorToImageData(data, width, height) {
    const imageData = new ImageData(width, height);
    const pixels = imageData.data;
    const isF16 = data.constructor === Uint16Array;
    const f32Data = isF16 ? this._f16ToF32(data) : data;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIdx = (y * width + x) * 4;
        const r = f32Data[0 * height * width + y * width + x];
        const g = f32Data[1 * height * width + y * width + x];
        const b = f32Data[2 * height * width + y * width + x];
        // Normalize from [-1,1] or [0,1] range to [0,255]
        pixels[pixelIdx + 0] = Math.max(0, Math.min(255, Math.round((r * 0.5 + 0.5) * 255)));
        pixels[pixelIdx + 1] = Math.max(0, Math.min(255, Math.round((g * 0.5 + 0.5) * 255)));
        pixels[pixelIdx + 2] = Math.max(0, Math.min(255, Math.round((b * 0.5 + 0.5) * 255)));
        pixels[pixelIdx + 3] = 255;
      }
    }
    return imageData;
  }

  /** Dispose of ONNX sessions to free GPU memory. */
  async dispose() {
    if (this.textEncoderSession) { await this.textEncoderSession.release(); this.textEncoderSession = null; }
    if (this.ditSession) { await this.ditSession.release(); this.ditSession = null; }
    if (this.vaeSession) { await this.vaeSession.release(); this.vaeSession = null; }
  }
}

if (typeof window !== "undefined") {
  window.SanaPipeline = SanaPipeline;
}
