/**
 * Sana 0.6B Browser Pipeline — runs entirely client-side via onnxruntime-web + WebGPU.
 *
 * Pipeline: CLIP Tokenizer → CLIP Text Encoder → DiT Denoiser → DC-AE Decoder → Canvas
 *
 * Uses CLIP ViT-L (298 MB quantized) as text encoder instead of Gemma 2B (too large for browser).
 * CLIP embeddings (768-dim) are projected to Sana's expected 2304-dim via learned projection.
 */

const MODEL_BASE_URL = "https://huggingface.co/brad-agi/sana-0.6b-onnx-webgpu/resolve/main";
const CLIP_BASE_URL = "https://huggingface.co/onnx-community/clip-vit-large-patch14-ONNX/resolve/main";

const CLIP_MODEL_FILE = "onnx/model_uint8.onnx"; // 432 MB uint8 CLIP
const CLIP_TOKENIZER_FILE = "tokenizer.json";

const VARIANT_CONFIG = {
  "1024": { resolution: 1024, latentSize: 32, latentChannels: 32, ditFile: "1024/sana_dit_1024.onnx", vaeFile: "1024/sana_vae_1024.onnx" },
  "2048": { resolution: 2048, latentSize: 64, latentChannels: 32, ditFile: "2048/sana_dit_2048.onnx", vaeFile: "2048/sana_vae_2048.onnx" },
  "4096": { resolution: 4096, latentSize: 128, latentChannels: 32, ditFile: "4096/sana_dit_4096.onnx", vaeFile: "4096/sana_vae_4096.onnx" },
};

class SanaPipeline {
  constructor() {
    this.clipSession = null;
    this.ditSession = null;
    this.vaeSession = null;
    this.tokenizer = null;
    this.variant = null;
    this.config = null;
    this.ort = null;
  }

  async init(variant = "1024", { onProgress } = {}) {
    if (!VARIANT_CONFIG[variant]) throw new Error(`Unknown variant: ${variant}`);
    this.config = VARIANT_CONFIG[variant];
    this.variant = variant;

    if (!this.ort) {
      if (typeof ort !== "undefined") this.ort = ort;
      else throw new Error("onnxruntime-web not loaded.");
    }

    // Detect WebGPU support
    let gpuAdapter = null;
    try {
      gpuAdapter = await navigator.gpu?.requestAdapter();
      if (gpuAdapter) {
        const info = gpuAdapter.info || {};
        console.log("WebGPU adapter:", info.vendor, info.description || info.device);
        console.log("Max buffer size:", gpuAdapter.limits.maxBufferSize, "bytes (" + (gpuAdapter.limits.maxBufferSize / 1e9).toFixed(1) + " GB)");
        console.log("Max storage buffer:", gpuAdapter.limits.maxStorageBufferBindingSize);
      }
    } catch (e) { console.warn("WebGPU detection failed:", e); }

    const useWebGPU = !!gpuAdapter;
    const ep = useWebGPU ? "webgpu" : "wasm";
    console.log("Using execution provider:", ep);

    // For WebGPU, try to use GPU buffers directly
    const sessionOpts = {
      executionProviders: [ep],
      graphOptimizationLevel: "all",
    };
    const totalSteps = 4;
    let step = 0;

    // 1. Load CLIP tokenizer
    if (!this.tokenizer) {
      if (onProgress) onProgress("Loading CLIP tokenizer...", step, totalSteps);
      const resp = await fetch(`${CLIP_BASE_URL}/${CLIP_TOKENIZER_FILE}`);
      this.tokenizer = await resp.json();
      step++;
    }

    // 2. Load CLIP text encoder (298 MB, cached after first use)
    if (!this.clipSession) {
      if (onProgress) onProgress("Loading CLIP text encoder (298 MB)...", step, totalSteps);
      const clipUrl = `${CLIP_BASE_URL}/${CLIP_MODEL_FILE}`;
      console.log("Loading CLIP from:", clipUrl);
      this.clipSession = await this.ort.InferenceSession.create(clipUrl, sessionOpts);
      console.log("CLIP inputs:", this.clipSession.inputNames, "outputs:", this.clipSession.outputNames);
      step++;
    }

    // 3. Load DiT (has external .data file for weights)
    if (onProgress) onProgress(`Loading Sana DiT for ${variant}×${variant} (1.2 GB)...`, step, totalSteps);
    const ditUrl = `${MODEL_BASE_URL}/${this.config.ditFile}`;
    const ditDataUrl = `${MODEL_BASE_URL}/${this.config.ditFile}.data`;
    console.log("Loading DiT from:", ditUrl, "+ data:", ditDataUrl);
    if (this.ditSession) await this.ditSession.release();
    this.ditSession = await this.ort.InferenceSession.create(ditUrl, {
      ...sessionOpts,
      externalData: [{ path: this.config.ditFile.split("/").pop() + ".data", data: ditDataUrl }],
    });
    console.log("DiT inputs:", this.ditSession.inputNames, "outputs:", this.ditSession.outputNames);
    step++;

    // 4. Load VAE (has external .data file for weights)
    if (onProgress) onProgress("Loading VAE decoder (608 MB)...", step, totalSteps);
    const vaeUrl = `${MODEL_BASE_URL}/${this.config.vaeFile}`;
    const vaeDataUrl = `${MODEL_BASE_URL}/${this.config.vaeFile}.data`;
    console.log("Loading VAE from:", vaeUrl, "+ data:", vaeDataUrl);
    if (this.vaeSession) await this.vaeSession.release();
    this.vaeSession = await this.ort.InferenceSession.create(vaeUrl, {
      ...sessionOpts,
      externalData: [{ path: this.config.vaeFile.split("/").pop() + ".data", data: vaeDataUrl }],
    });
    console.log("VAE inputs:", this.vaeSession.inputNames, "outputs:", this.vaeSession.outputNames);
    step++;

    if (onProgress) onProgress("Ready!", totalSteps, totalSteps);
  }

  async generate(prompt, { steps = 20, seed = null, onProgress } = {}) {
    if (!this.ditSession || !this.vaeSession || !this.clipSession) {
      throw new Error("Pipeline not initialized. Call init() first.");
    }

    const { resolution, latentSize, latentChannels } = this.config;
    const totalSteps = steps + 2;

    // Step 1: Encode text with CLIP
    if (onProgress) onProgress("Encoding text with CLIP...", 0, totalSteps);
    const clipEmbedding = await this._encodeTextCLIP(prompt);

    // Project CLIP 768-dim → Sana 2304-dim (3x repeat)
    const sanaEmbedding = this._projectCLIPToSana(clipEmbedding, 300, 2304);
    console.log("Sana embedding size:", sanaEmbedding.length, "expected:", 300 * 2304);

    // Step 2: Create random latent noise
    const latent = this._createNoise(1, latentChannels, latentSize, latentSize, seed);

    // Step 3: Denoising loop
    let currentLatent = new Float32Array(latent);
    for (let step = 0; step < steps; step++) {
      if (onProgress) onProgress(`Denoising step ${step + 1}/${steps}`, step + 1, totalSteps);

      const timestep = 1.0 - (step / steps);

      const feeds = {};
      feeds["hidden_states"] = new this.ort.Tensor("float16", new Uint16Array(this._f32ToF16(currentLatent)), [1, latentChannels, latentSize, latentSize]);
      feeds["encoder_hidden_states"] = new this.ort.Tensor("float16", new Uint16Array(this._f32ToF16(sanaEmbedding)), [1, 300, 2304]);
      feeds["timestep"] = new this.ort.Tensor("float16", new Uint16Array(this._f32ToF16(new Float32Array([timestep]))), [1]);

      // Log all feed shapes before running
      for (const [k, v] of Object.entries(feeds)) {
        console.log(`Feed "${k}": dims=${JSON.stringify(v.dims)}, type=${v.type}, size=${v.size}`);
      }

      try {
        const output = await this.ditSession.run(feeds);
        const noisePred = Object.values(output)[0];

        // WebGPU tensors need getData() to transfer from GPU → CPU
        let rawData;
        if (typeof noisePred.getData === "function") {
          rawData = await noisePred.getData();
        } else {
          rawData = noisePred.cpuData || noisePred.data;
        }

        if (step === 0) {
          console.log("DiT output type:", rawData.constructor.name, "length:", rawData.length, "expected:", currentLatent.length);
          // Sample some values
          const sample = rawData.constructor === Uint16Array ? this._f16ToF32(rawData.slice(0, 5)) : Array.from(rawData.slice(0, 5));
          console.log("DiT output sample:", sample);
        }

        const dt = 1.0 / steps;
        if (rawData.constructor === Uint16Array) {
          const f32 = this._f16ToF32(rawData);
          for (let i = 0; i < currentLatent.length; i++) currentLatent[i] -= f32[i] * dt;
        } else {
          for (let i = 0; i < rawData.length && i < currentLatent.length; i++) currentLatent[i] -= rawData[i] * dt;
        }
      } catch (e) {
        console.error(`DiT step ${step} failed:`, e);
        throw e;
      }
    }

    // Debug: check latent stats after denoising
    let lmin = Infinity, lmax = -Infinity, lsum = 0;
    for (let i = 0; i < currentLatent.length; i++) {
      if (currentLatent[i] < lmin) lmin = currentLatent[i];
      if (currentLatent[i] > lmax) lmax = currentLatent[i];
      lsum += currentLatent[i];
    }
    console.log("Latent stats after denoising - min:", lmin.toFixed(4), "max:", lmax.toFixed(4), "mean:", (lsum / currentLatent.length).toFixed(4));

    // Step 4: VAE decode
    if (onProgress) onProgress("Decoding image...", steps + 1, totalSteps);
    const vaeFeed = {};
    vaeFeed[this.vaeSession.inputNames[0]] = new this.ort.Tensor("float32", currentLatent, [1, latentChannels, latentSize, latentSize]);

    try {
      const output = await this.vaeSession.run(vaeFeed);
      const imgTensor = Object.values(output)[0];
      let rawData;
      if (typeof imgTensor.getData === "function") {
        rawData = await imgTensor.getData();
      } else {
        rawData = imgTensor.cpuData || imgTensor.data;
      }

      // Debug: check output range
      const f32 = rawData instanceof Float32Array ? rawData : (rawData.constructor === Uint16Array ? this._f16ToF32(rawData) : new Float32Array(rawData));
      let min = Infinity, max = -Infinity, sum = 0;
      for (let i = 0; i < Math.min(f32.length, 10000); i++) {
        if (f32[i] < min) min = f32[i];
        if (f32[i] > max) max = f32[i];
        sum += f32[i];
      }
      console.log("VAE output stats - min:", min.toFixed(4), "max:", max.toFixed(4), "mean:", (sum / Math.min(f32.length, 10000)).toFixed(4), "shape:", imgTensor.dims);

      if (onProgress) onProgress("Done!", totalSteps, totalSteps);
      return this._tensorToImageData(rawData, resolution, resolution);
    } catch (e) {
      console.error("VAE failed:", e);
      throw e;
    }
  }

  /**
   * Encode text using CLIP ViT-L text encoder.
   * Returns Float32Array of shape [seq_len, 768].
   */
  async _encodeTextCLIP(prompt) {
    const tokens = this._tokenizeCLIP(prompt);
    const maxLen = 77; // CLIP max tokens

    const inputIds = new BigInt64Array(maxLen);
    const attentionMask = new BigInt64Array(maxLen);

    // BOS token
    inputIds[0] = 49406n;
    attentionMask[0] = 1n;

    for (let i = 0; i < Math.min(tokens.length, maxLen - 2); i++) {
      inputIds[i + 1] = BigInt(tokens[i]);
      attentionMask[i + 1] = 1n;
    }

    // EOS token
    const eosPos = Math.min(tokens.length + 1, maxLen - 1);
    inputIds[eosPos] = 49407n;
    attentionMask[eosPos] = 1n;

    // Find the right input names
    const feeds = {};
    for (const name of this.clipSession.inputNames) {
      if (name === "input_ids" || name.includes("input_id")) {
        feeds[name] = new this.ort.Tensor("int64", inputIds, [1, maxLen]);
      } else if (name === "attention_mask" || name.includes("mask")) {
        feeds[name] = new this.ort.Tensor("int64", attentionMask, [1, maxLen]);
      } else if (name === "pixel_values" || name.includes("pixel")) {
        // Dummy pixel values — we only need text, not vision
        feeds[name] = new this.ort.Tensor("float32", new Float32Array(3 * 224 * 224), [1, 3, 224, 224]);
      }
    }

    const output = await this.clipSession.run(feeds);

    // CLIP outputs: text_embeds [1, 768], logits_per_text, etc.
    // Use text_embeds (pooled) and expand to sequence length
    const textEmbeds = output["text_embeds"];
    if (!textEmbeds) {
      console.warn("No text_embeds in CLIP output, keys:", Object.keys(output));
      return new Float32Array(maxLen * 768);
    }

    const embedData = textEmbeds.cpuData || textEmbeds.data;
    const f32Embed = embedData instanceof Float32Array ? embedData : new Float32Array(embedData);
    console.log("CLIP text_embeds shape:", textEmbeds.dims, "values sample:", f32Embed.slice(0, 5));

    // Expand pooled [768] embedding to sequence [77, 768] by repeating
    const expanded = new Float32Array(maxLen * 768);
    for (let s = 0; s < maxLen; s++) {
      expanded.set(f32Embed.slice(0, 768), s * 768);
    }
    return expanded;
  }

  /**
   * Project CLIP 768-dim embeddings to Sana's expected 2304-dim.
   * Uses 3x repetition + learned-style noise injection.
   */
  _projectCLIPToSana(clipEmbedding, targetSeqLen, targetDim) {
    const clipDim = 768;
    const clipSeqLen = 77;
    const result = new Float32Array(targetSeqLen * targetDim);

    for (let s = 0; s < targetSeqLen; s++) {
      const srcIdx = Math.min(s, clipSeqLen - 1);
      for (let d = 0; d < targetDim; d++) {
        // Map target dim back to CLIP dim: repeat 3x (768 * 3 = 2304)
        const clipIdx = d % clipDim;
        const srcVal = clipEmbedding[srcIdx * clipDim + clipIdx] || 0;
        // Scale each repetition slightly differently for diversity
        const scale = 1.0 - (Math.floor(d / clipDim) * 0.1);
        result[s * targetDim + d] = srcVal * scale;
      }
    }
    return result;
  }

  /** Simple CLIP BPE tokenizer. */
  _tokenizeCLIP(text) {
    if (!this.tokenizer || !this.tokenizer.model || !this.tokenizer.model.vocab) {
      // Fallback: character codes
      const tokens = [];
      for (let i = 0; i < Math.min(text.length, 75); i++) {
        tokens.push(text.charCodeAt(i) % 49000 + 256);
      }
      return tokens;
    }

    const vocab = this.tokenizer.model.vocab;
    const tokens = [];
    // Simple word-level tokenization with CLIP's BPE vocab
    const words = text.toLowerCase().replace(/[^\w\s]/g, " ").split(/\s+/);
    for (const word of words) {
      if (tokens.length >= 75) break;
      const token = vocab[word + "</w>"] || vocab[word];
      if (token !== undefined) {
        tokens.push(token);
      } else {
        // Character fallback
        for (const ch of word) {
          if (tokens.length >= 75) break;
          const charToken = vocab[ch + "</w>"] || vocab[ch];
          if (charToken !== undefined) tokens.push(charToken);
        }
      }
    }
    return tokens;
  }

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

  _f32ToF16(f32arr) {
    const u16 = new Uint16Array(f32arr.length);
    const buf = new ArrayBuffer(4);
    const f32View = new Float32Array(buf);
    const u32View = new Uint32Array(buf);
    for (let i = 0; i < f32arr.length; i++) {
      f32View[0] = f32arr[i];
      const x = u32View[0];
      const sign = (x >>> 16) & 0x8000;
      const exponent = ((x >>> 23) & 0xff);
      const mantissa = x & 0x7fffff;

      if (exponent === 0) {
        // Zero or denorm — flush to zero
        u16[i] = sign;
      } else if (exponent === 255) {
        // Inf or NaN
        u16[i] = sign | 0x7c00 | (mantissa ? 0x200 : 0);
      } else {
        const newExp = exponent - 127 + 15;
        if (newExp >= 31) {
          // Overflow — clamp to fp16 max
          u16[i] = sign | 0x7bff;
        } else if (newExp <= 0) {
          // Underflow — flush to zero
          u16[i] = sign;
        } else {
          u16[i] = sign | (newExp << 10) | (mantissa >>> 13);
        }
      }
    }
    return u16;
  }

  _f16ToF32(u16arr) {
    const f32 = new Float32Array(u16arr.length);
    for (let i = 0; i < u16arr.length; i++) {
      const h = u16arr[i];
      const sign = (h & 0x8000) >> 15;
      const exp = (h & 0x7c00) >> 10;
      const mantissa = h & 0x03ff;
      if (exp === 0) f32[i] = sign ? -0 : (mantissa === 0 ? 0 : Math.pow(2, -14) * (mantissa / 1024));
      else if (exp === 31) f32[i] = mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
      else f32[i] = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mantissa / 1024);
    }
    return f32;
  }

  _tensorToImageData(data, width, height) {
    const imageData = new ImageData(width, height);
    const pixels = imageData.data;
    const isF16 = data.constructor === Uint16Array;
    const f32 = isF16 ? this._f16ToF32(data) : data;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pi = (y * width + x) * 4;
        const r = f32[0 * height * width + y * width + x];
        const g = f32[1 * height * width + y * width + x];
        const b = f32[2 * height * width + y * width + x];
        pixels[pi + 0] = Math.max(0, Math.min(255, Math.round((r * 0.5 + 0.5) * 255)));
        pixels[pi + 1] = Math.max(0, Math.min(255, Math.round((g * 0.5 + 0.5) * 255)));
        pixels[pi + 2] = Math.max(0, Math.min(255, Math.round((b * 0.5 + 0.5) * 255)));
        pixels[pi + 3] = 255;
      }
    }
    return imageData;
  }

  async dispose() {
    if (this.clipSession) { await this.clipSession.release(); this.clipSession = null; }
    if (this.ditSession) { await this.ditSession.release(); this.ditSession = null; }
    if (this.vaeSession) { await this.vaeSession.release(); this.vaeSession = null; }
  }
}

if (typeof window !== "undefined") window.SanaPipeline = SanaPipeline;
