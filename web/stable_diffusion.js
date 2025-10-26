/**
 * Lightweight wrapper that converts TVM NDArrays to transferable payloads.
 * The helper prefers GPU backed buffers but falls back to CPU typed arrays
 * when the underlying handle does not expose a transferable representation.
 */
class TransferableNDArray {
  constructor(ndarray) {
    this.ndarray = ndarray;
  }

  /**
   * Export the NDArray into a transferable message.
   * @returns {{shape: number[], dtype: string, buffer: ArrayBuffer}}
   */
  toMessage() {
    if (!this.ndarray) {
      throw Error("Attempted to serialize a disposed array");
    }
    const typedArray = this.ndarray.toArray();
    return {
      shape: Array.from(this.ndarray.shape),
      dtype: this.ndarray.dtype,
      buffer: typedArray.buffer
    };
  }

  /**
   * Import a transferable message back into a TVM NDArray on the
   * specified device.
   * @param tvm The tvm runtime instance.
   * @param device Target device.
   * @param payload The transferable payload.
   */
  static fromMessage(tvm, device, payload) {
    const ctor = TransferableNDArray.#getTypedArrayCtor(payload.dtype);
    const typedArray = new ctor(payload.buffer);
    const result = tvm.empty(payload.shape, payload.dtype, device);
    result.copyFrom(typedArray);
    return result;
  }

  static #getTypedArrayCtor(dtype) {
    switch (dtype) {
    case "float32":
      return Float32Array;
    case "float16":
      return Uint16Array;
    case "int32":
      return Int32Array;
    case "uint8":
      return Uint8Array;
    default:
      throw Error("Unsupported dtype " + dtype);
    }
  }
}

/**
 * Simple asynchronous command queue with back-pressure control.
 */
class CommandQueue {
  constructor(maxInFlight = 2) {
    this.maxInFlight = Math.max(1, maxInFlight);
    this.pending = [];
    this.inFlight = 0;
  }

  enqueue(fn) {
    return new Promise((resolve, reject) => {
      const command = { fn, resolve, reject };
      command.resultPromise = new Promise((innerResolve, innerReject) => {
        command.resolveWrapper = innerResolve;
        command.rejectWrapper = innerReject;
      });
      this.pending.push(command);
      this.#drain();
    });
  }

  flush() {
    if (this.inFlight === 0 && this.pending.length === 0) {
      return Promise.resolve();
    }
    return Promise.all(
      this.pending
        .map((cmd) => cmd.resultPromise)
        .filter((promise) => promise !== undefined)
    );
  }

  #drain() {
    while (this.inFlight < this.maxInFlight && this.pending.length > 0) {
      const command = this.pending.shift();
      this.inFlight += 1;
      Promise.resolve()
        .then(() => command.fn())
        .then((value) => {
          this.inFlight -= 1;
          command.resolve(value);
          if (command.resolveWrapper) {
            command.resolveWrapper(value);
          }
          this.#drain();
        })
        .catch((err) => {
          this.inFlight -= 1;
          command.reject(err);
          if (command.rejectWrapper) {
            command.rejectWrapper(err);
          }
          this.#drain();
        });
    }
  }
}

/**
 * Minimal worker abstraction that attempts to spawn a dedicated Web Worker
 * for a stage but gracefully falls back to local execution when the
 * environment prevents worker creation (e.g., during tests).
 */
class StageWorker {
  constructor(name, executor, options = {}) {
    this.name = name;
    this.executor = executor;
    this.worker = undefined;
    this.useWorker = options.useWorker !== undefined ? options.useWorker : true;
    if (this.useWorker) {
      this.#initWorker();
    }
  }

  async run(payload) {
    if (this.worker === undefined) {
      return await this.executor(payload);
    }
    const message = { id: StageWorker.#nextMessageId(), payload };
    return new Promise((resolve, reject) => {
      const handleMessage = (evt) => {
        if (evt.data.id !== message.id) {
          return;
        }
        this.worker.removeEventListener("message", handleMessage);
        if (evt.data.status === "ok") {
          resolve(evt.data.result);
        } else {
          reject(new Error(evt.data.error));
        }
      };
      this.worker.addEventListener("message", handleMessage);
      const transferList = payload && payload.transferList ? payload.transferList : undefined;
      if (transferList) {
        this.worker.postMessage(message, transferList);
      } else {
        this.worker.postMessage(message);
      }
    });
  }

  dispose() {
    if (this.worker !== undefined) {
      this.worker.terminate();
      this.worker = undefined;
    }
  }

  #initWorker() {
    if (typeof Worker === "undefined") {
      return;
    }
    try {
      const source = `
        const executor = (${this.executor.toString()});
        self.onmessage = async (evt) => {
          const { id, payload } = evt.data;
          try {
            const result = await executor(payload);
            self.postMessage({ id, status: "ok", result });
          } catch (err) {
            self.postMessage({ id, status: "error", error: err.message || String(err) });
          }
        };
      `;
      const blob = new Blob([source], { type: "application/javascript" });
      const url = URL.createObjectURL(blob);
      this.worker = new Worker(url, { name: this.name });
    } catch (err) {
      console.warn("Falling back to main thread execution for stage", this.name, err);
      this.worker = undefined;
    }
  }

  static #nextMessageId() {
    StageWorker._counter = (StageWorker._counter || 0) + 1;
    return StageWorker._counter;
  }
}

/**
 * Pool that keeps track of the stage workers used by the runtime.
 */
class StageWorkerPool {
  constructor(enableWorkers = true) {
    this.enableWorkers = enableWorkers;
    this.clipWorker = undefined;
    this.unetWorker = undefined;
    this.vaeWorker = undefined;
  }

  init(clipExec, unetExec, vaeExec) {
    this.dispose();
    const workerOptions = { useWorker: this.enableWorkers };
    this.clipWorker = new StageWorker("clip-stage", clipExec, workerOptions);
    this.unetWorker = new StageWorker("unet-stage", unetExec, workerOptions);
    this.vaeWorker = new StageWorker("vae-stage", vaeExec, workerOptions);
  }

  dispose() {
    if (this.clipWorker) {
      this.clipWorker.dispose();
    }
    if (this.unetWorker) {
      this.unetWorker.dispose();
    }
    if (this.vaeWorker) {
      this.vaeWorker.dispose();
    }
  }
}

/**
 * Orchestrates collaborative generation features with server offloading.
 */
class CollaborativeGenerationManager {
  constructor(logger) {
    this.logger = logger;
    this.transport = undefined;
    this.rtcPeer = undefined;
    this.latencyHistory = [];
    this.healthListeners = new Set();
  }

  async ensureTransport(url) {
    if (this.transport || typeof WebTransport === "undefined") {
      return;
    }
    try {
      this.transport = new WebTransport(url);
      await this.transport.ready;
      this.logger("WebTransport channel ready" );
      this.#monitorTransport();
    } catch (err) {
      this.logger("WebTransport connection failed: " + err.message);
      this.transport = undefined;
    }
  }

  async ensurePeerConnection(config = {}) {
    if (this.rtcPeer) {
      return;
    }
    if (typeof RTCPeerConnection === "undefined") {
      return;
    }
    this.rtcPeer = new RTCPeerConnection(config);
    this.dataChannel = this.rtcPeer.createDataChannel("generation-updates");
    this.dataChannel.onmessage = (evt) => {
      this.#recordLatency("webrtc", evt.data);
    };
    this.dataChannel.onopen = () => {
      this.logger("WebRTC data channel established");
    };
    this.dataChannel.onclose = () => {
      this.logger("WebRTC data channel closed");
    };
  }

  broadcastStageSnapshot(stage, payload) {
    const timestamp = performance.now();
    if (this.transport) {
      try {
        const writer = this.transport.datagrams.writable.getWriter();
        const json = JSON.stringify({ stage, payload, timestamp });
        writer.write(new TextEncoder().encode(json))
          .catch((err) => {
            this.logger("Transport write failed: " + err.message);
          })
          .finally(() => {
            writer.releaseLock();
          });
      } catch (err) {
        this.logger("Transport write failed: " + err.message);
      }
    }
    if (this.dataChannel && this.dataChannel.readyState === "open") {
      try {
        this.dataChannel.send(JSON.stringify({ stage, payload, timestamp }));
      } catch (err) {
        this.logger("WebRTC send failed: " + err.message);
      }
    }
  }

  onHealthUpdate(cb) {
    this.healthListeners.add(cb);
    return () => this.healthListeners.delete(cb);
  }

  #recordLatency(channel, rawMessage) {
    try {
      const parsed = JSON.parse(rawMessage);
      if (!parsed.timestamp) {
        return;
      }
      const latency = performance.now() - parsed.timestamp;
      this.latencyHistory.push({ channel, latency });
      this.latencyHistory = this.latencyHistory.slice(-100);
      this.healthListeners.forEach((cb) => cb({ channel, latency }));
    } catch (err) {
      this.logger("Latency parsing failed: " + err.message);
    }
  }

  #monitorTransport() {
    if (!this.transport) {
      return;
    }
    const interval = setInterval(() => {
      if (!this.transport) {
        clearInterval(interval);
        return;
      }
      try {
        this.broadcastStageSnapshot("health", { timestamp: performance.now() });
      } catch (err) {
        this.logger("Health ping failed: " + err.message);
      }
    }, 5000);
  }

  dispose() {
    if (this.transport) {
      try {
        this.transport.close();
      } catch (_) {
      }
      this.transport = undefined;
    }
    if (this.dataChannel) {
      try {
        this.dataChannel.close();
      } catch (_) {
      }
      this.dataChannel = undefined;
    }
    if (this.rtcPeer) {
      try {
        this.rtcPeer.close();
      } catch (_) {
      }
      this.rtcPeer = undefined;
    }
    this.healthListeners.clear();
  }
}

/**
 * Wrapper to handle PNDM scheduler
 */
class TVMPNDMScheduler {
  constructor(schedulerConsts, latentShape, tvm, device, vm) {
    this.timestep = [];
    this.sampleCoeff = [];
    this.alphaDiff = [];
    this.modelOutputDenomCoeff = [];
    this.ets = [];
    this.schedulerFunc = [];
    this.currSample = undefined;
    this.tvm = tvm;

    // prebuild constants
    // principle: always detach for class members
    // to avoid recycling output scope.
    function loadConsts(output, dtype, input) {
      for (let t = 0; t < input.length; ++t) {
        output.push(
          tvm.detachFromCurrentScope(
            tvm.empty([], dtype, device).copyFrom([input[t]])
          )
        );
      }
    }
    loadConsts(this.timestep, "int32", schedulerConsts["timesteps"]);
    loadConsts(this.sampleCoeff, "float32", schedulerConsts["sample_coeff"]);
    loadConsts(this.alphaDiff, "float32", schedulerConsts["alpha_diff"]);
    loadConsts(
      this.modelOutputDenomCoeff, "float32",
      schedulerConsts["model_output_denom_coeff"]);

    for (let i = 0; i < 4; ++i) {
      this.ets.push(
        this.tvm.detachFromCurrentScope(
          this.tvm.empty(latentShape, "float32", device)
        )
      );
    }

    for (let i = 0; i < 5; ++i) {
      this.schedulerFunc.push(
        tvm.detachFromCurrentScope(
          vm.getFunction("pndm_scheduler_step_" + i.toString())
        )
      );
    }
  }

  dispose() {
    for (let t = 0; t < this.timestep.length; ++t) {
      this.timestep[t].dispose();
      this.sampleCoeff[t].dispose();
      this.alphaDiff[t].dispose();
      this.modelOutputDenomCoeff[t].dispose();
    }

    for (let i = 0; i < this.schedulerFunc.length; ++i) {
      this.schedulerFunc[i].dispose();
    }

    if (this.currSample) {
      this.currSample.dispose();
    }
    for (let i = 0; i < this.ets.length; ++i) {
      this.ets[i].dispose();
    }
  }

  step(modelOutput, sample, counter) {
    // keep running history of last four inputs
    if (counter != 1) {
      this.ets.shift();
      this.ets.push(this.tvm.detachFromCurrentScope(
        modelOutput
      ));
    }
    if (counter == 0) {
      this.currSample = this.tvm.detachFromCurrentScope(
        sample
      );
    } else if (counter == 1) {
      sample = this.tvm.attachToCurrentScope(this.currSample);
      this.currSample = undefined;
    }

    const findex = counter < 4 ? counter : 4;
    const prevLatents = this.schedulerFunc[findex](
      sample,
      modelOutput,
      this.sampleCoeff[counter],
      this.alphaDiff[counter],
      this.modelOutputDenomCoeff[counter],
      this.ets[0],
      this.ets[1],
      this.ets[2],
      this.ets[3]
    );
    return prevLatents;
  }

  prefetch(counter) {
    return {
      timestep: this.timestep[counter],
      sampleCoeff: this.sampleCoeff[counter],
      alphaDiff: this.alphaDiff[counter],
      modelOutputDenomCoeff: this.modelOutputDenomCoeff[counter]
    };
  }
}

/**
 * Wrapper to handle multistep DPM-solver scheduler
 */
class TVMDPMSolverMultistepScheduler {
  constructor(schedulerConsts, latentShape, tvm, device, vm) {
    this.timestep = [];
    this.alpha = [];
    this.sigma = [];
    this.c0 = [];
    this.c1 = [];
    this.c2 = [];
    this.lastModelOutput = undefined;
    this.convertModelOutputFunc = undefined;
    this.stepFunc = undefined;
    this.tvm = tvm;

    // prebuild constants
    // principle: always detach for class members
    // to avoid recycling output scope.
    function loadConsts(output, dtype, input) {
      for (let t = 0; t < input.length; ++t) {
        output.push(
          tvm.detachFromCurrentScope(
            tvm.empty([], dtype, device).copyFrom([input[t]])
          )
        );
      }
    }
    loadConsts(this.timestep, "int32", schedulerConsts["timesteps"]);
    loadConsts(this.alpha, "float32", schedulerConsts["alpha"]);
    loadConsts(this.sigma, "float32", schedulerConsts["sigma"]);
    loadConsts(this.c0, "float32", schedulerConsts["c0"]);
    loadConsts(this.c1, "float32", schedulerConsts["c1"]);
    loadConsts(this.c2, "float32", schedulerConsts["c2"]);

    this.lastModelOutput = this.tvm.detachFromCurrentScope(
      this.tvm.empty(latentShape, "float32", device)
    )
    this.convertModelOutputFunc = tvm.detachFromCurrentScope(
      vm.getFunction("dpm_solver_multistep_scheduler_convert_model_output")
    )
    this.stepFunc = tvm.detachFromCurrentScope(
      vm.getFunction("dpm_solver_multistep_scheduler_step")
    )
  }

  dispose() {
    for (let t = 0; t < this.timestep.length; ++t) {
      this.timestep[t].dispose();
      this.alpha[t].dispose();
      this.sigma[t].dispose();
      this.c0[t].dispose();
      this.c1[t].dispose();
      this.c2[t].dispose();
    }

    this.lastModelOutput.dispose();
    this.convertModelOutputFunc.dispose();
    this.stepFunc.dispose();
  }

  step(modelOutput, sample, counter) {
    modelOutput = this.convertModelOutputFunc(sample, modelOutput, this.alpha[counter], this.sigma[counter])
    const prevLatents = this.stepFunc(
      sample,
      modelOutput,
      this.lastModelOutput,
      this.c0[counter],
      this.c1[counter],
      this.c2[counter],
    );
    this.lastModelOutput = this.tvm.detachFromCurrentScope(
      modelOutput
    );

    return prevLatents;
  }

  prefetch(counter) {
    return {
      timestep: this.timestep[counter],
      alpha: this.alpha[counter],
      sigma: this.sigma[counter],
      c0: this.c0[counter],
      c1: this.c1[counter],
      c2: this.c2[counter]
    };
  }
}

class EulerDiscreteScheduler {
  constructor(schedulerConsts, latentShape, tvm, device, vm) {
    this.timestep = [];
    this.sigma = [];
    this.lastModelOutput = undefined;
    this.ScaleModelInputFunc = undefined;
    this.stepFunc = undefined;
    this.tvm = tvm;

    // prebuild constants
    // principle: always detach for class members
    // to avoid recycling output scope.
    function loadConsts(output, dtype, input) {
      for (let t = 0; t < input.length; ++t) {
        output.push(
          tvm.detachFromCurrentScope(
            tvm.empty([], dtype, device).copyFrom([input[t]])
          )
        );
      }
    }
    loadConsts(this.timestep, "int32", schedulerConsts["timesteps"]);
    loadConsts(this.sigma, "float32", schedulerConsts["sigma"]);

    this.lastModelOutput = this.tvm.detachFromCurrentScope(
      this.tvm.empty(latentShape, "float32", device)
    )
    this.ScaleModelInputFunc = tvm.detachFromCurrentScope(
      vm.getFunction("euler_discrete_scheduler_scale")
    )
    this.stepFunc = tvm.detachFromCurrentScope(
      vm.getFunction("euler_discrete_scheduler_step")
    )
  }

  dispose() {
    for (let t = 0; t < this.timestep.length; ++t) {
      this.timestep[t].dispose();
      this.sigma[t].dispose();
    }

    this.lastModelOutput.dispose();
    this.ScaleModelInputFunc.dispose();
    this.stepFunc.dispose();
  }

  step(modelOutput, sample, counter) {
    const prevLatents = this.stepFunc(
      sample,
      modelOutput,
      this.sigma[counter],
      this.sigma[counter+1],
    );

    return prevLatents;
  }

  prefetch(counter) {
    return {
      timestep: this.timestep[counter],
      sigma: this.sigma[counter]
    };
  }

  scaleModelInput(sample, counter) {
    const result = this.ScaleModelInputFunc(
      sample,
      this.sigma[counter],
    );

    return result;
  }
}

class StableDiffusionPipeline {
  constructor(tvm, tokenizer, schedulerConsts, cacheMetadata) {
    if (cacheMetadata == undefined) {
      throw Error("Expect cacheMetadata");
    }
    this.tvm = tvm;
    this.tokenizer = tokenizer;
    this.maxTokenLength = 77;

    this.device = this.tvm.webgpu();
    this.tvm.bindCanvas(document.getElementById("canvas"));
    // VM functions
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device)
    );

    this.schedulerConsts = schedulerConsts;
    this.clipToTextEmbeddings = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("clip")
    );
    this.clipParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("clip", cacheMetadata.clipParamSize)
    );
    this.unetLatentsToNoisePred = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("unet")
    );
    this.unetParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("unet", cacheMetadata.unetParamSize)
    );
    this.vaeToImage = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("vae")
    );
    this.vaeParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("vae", cacheMetadata.vaeParamSize)
    );
    this.imageToRGBA = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("image_to_rgba")
    );
    this.concatEmbeddings = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("concat_embeddings")
    );
    this.workerPool = new StageWorkerPool(false);
    this.collaborationManager = undefined;
  }

  #convertIdsMessageToNDArray(message) {
    return TransferableNDArray.fromMessage(this.tvm, this.device, message);
  }

  #encodePromptFromWorker(payload) {
    return this.tvm.withNewScope(() => {
      const ids = this.#convertIdsMessageToNDArray(payload.inputMessage);
      const result = this.tvm.detachFromCurrentScope(
        this.clipToTextEmbeddings(ids, this.clipParams)
      );
      const transferable = new TransferableNDArray(result).toMessage();
      transferable.transferList = [transferable.buffer];
      result.dispose();
      return transferable;
    });
  }

  #runUnetFromWorker(payload) {
    const { latentsMessage, timestepHandle, embeddingsMessage, scheduler, counter } = payload;
    return this.tvm.withNewScope(() => {
      const latents = TransferableNDArray.fromMessage(this.tvm, this.device, latentsMessage);
      const embeddings = TransferableNDArray.fromMessage(this.tvm, this.device, embeddingsMessage);
      const timestep = timestepHandle ? timestepHandle.timestep : scheduler.timestep[counter];
      const noisePred = this.unetLatentsToNoisePred(
        latents,
        timestep,
        embeddings,
        this.unetParams
      );
      const nextLatents = scheduler.step(noisePred, latents, counter);
      const detached = this.tvm.detachFromCurrentScope(nextLatents);
      const transferable = new TransferableNDArray(detached).toMessage();
      transferable.transferList = [transferable.buffer];
      detached.dispose();
      return transferable;
    });
  }

  #runVaeFromWorker(payload) {
    return this.tvm.withNewScope(() => {
      const latents = TransferableNDArray.fromMessage(this.tvm, this.device, payload.latentsMessage);
      const image = this.vaeToImage(latents, this.vaeParams);
      const rgba = this.imageToRGBA(image);
      const detached = this.tvm.detachFromCurrentScope(rgba);
      const transferable = new TransferableNDArray(detached).toMessage();
      transferable.transferList = [transferable.buffer];
      detached.dispose();
      return transferable;
    });
  }

  dispose() {
    // note: tvm instance is not owned by this class
    this.concatEmbeddings.dispose();
    this.imageToRGBA.dispose()
    this.vaeParams.dispose();
    this.vaeToImage.dispose();
    this.unetParams.dispose();
    this.unetLatentsToNoisePred.dispose();
    this.clipParams.dispose();
    this.clipToTextEmbeddings.dispose();
    this.vm.dispose();
    this.workerPool.dispose();
  }

  setCollaborativeManager(manager) {
    this.collaborationManager = manager;
  }

  /**
   * Tokenize the prompt to TVMNDArray.
   * @param prompt Input prompt
   * @returns The text id NDArray.
   */
  tokenize(prompt) {
    const encoded = this.tokenizer.encode(prompt, true).input_ids;
    const inputIDs = new Int32Array(this.maxTokenLength);

    if (encoded.length < this.maxTokenLength) {
      inputIDs.set(encoded);
      const lastTok = encoded[encoded.length - 1];
      inputIDs.fill(lastTok, encoded.length, inputIDs.length);
    } else {
      inputIDs.set(encoded.slice(0, this.maxTokenLength));
    }
    return this.tvm.empty([1, this.maxTokenLength], "int32", this.device).copyFrom(inputIDs);
  }

  /**
   * async preload webgpu pipelines when possible.
   */
  async asyncLoadWebGPUPipelines() {
    await this.tvm.asyncLoadWebGPUPipelines(this.vm.getInternalModule());
  }

  /**
   * Run generation pipeline.
   *
   * @param prompt Input prompt.
   * @param negPrompt Input negative prompt.
   * @param progressCallback Callback to check progress.
   * @param schedulerId The integer ID of the scheduler to use.
   * - 0 for multi-step DPM solver,
   * - 1 for PNDM solver.
   * @param vaeCycle optionally draw VAE result every cycle iterations.
   * @param beginRenderVae Begin rendering VAE after skipping these warmup runs.
   */
  async generate(
    prompt,
    negPrompt = "",
    progressCallback = undefined,
    schedulerId = 0,
    vaeCycle = -1,
    beginRenderVae = 10
  ) {
    // Principle: beginScope/endScope in synchronized blocks,
    // this helps to recycle intermediate memories
    // detach states that needs to go across async boundaries.
    //--------------------------
    // Stage 0: CLIP
    //--------------------------
    this.tvm.beginScope();
    // get latents
    const latentShape = [1, 4, 64, 64];

    let scheduler;
    var unetNumSteps;
    if (schedulerId == 0) {
      scheduler = new TVMDPMSolverMultistepScheduler(
        this.schedulerConsts[0], latentShape, this.tvm, this.device, this.vm);
      unetNumSteps = this.schedulerConsts[0]["num_steps"];
    } else {
      scheduler = new TVMPNDMScheduler(
        this.schedulerConsts[1], latentShape, this.tvm, this.device, this.vm);
      unetNumSteps = this.schedulerConsts[1]["num_steps"];
    }
    const totalNumSteps = unetNumSteps + 2;

    this.workerPool.init(
      (payload) => this.#encodePromptFromWorker(payload),
      (payload) => this.#runUnetFromWorker(payload),
      (payload) => this.#runVaeFromWorker(payload)
    );

    if (progressCallback !== undefined) {
      progressCallback("clip", 0, 1, totalNumSteps);
    }

    const clipQueue = new CommandQueue(2);
    const posInputIDs = this.tokenize(prompt);
    const negInputIDs = this.tokenize(negPrompt);
    await this.device.sync();
    const posMessage = new TransferableNDArray(posInputIDs).toMessage();
    const negMessage = new TransferableNDArray(negInputIDs).toMessage();
    posMessage.transferList = [posMessage.buffer];
    negMessage.transferList = [negMessage.buffer];

    const [posEmbeddingMessage, negEmbeddingMessage] = await Promise.all([
      clipQueue.enqueue(() => this.workerPool.clipWorker.run({
        stage: "clip",
        negative: false,
        inputMessage: posMessage,
        transferList: posMessage.transferList
      })),
      clipQueue.enqueue(() => this.workerPool.clipWorker.run({
        stage: "clip",
        negative: true,
        inputMessage: negMessage,
        transferList: negMessage.transferList
      }))
    ]);

    posInputIDs.dispose();
    negInputIDs.dispose();

    const embeddings = this.tvm.withNewScope(() => {
      const posEmbeddings = TransferableNDArray.fromMessage(
        this.tvm,
        this.device,
        posEmbeddingMessage
      );
      const negEmbeddings = TransferableNDArray.fromMessage(
        this.tvm,
        this.device,
        negEmbeddingMessage
      );
      return this.tvm.detachFromCurrentScope(
        this.concatEmbeddings(negEmbeddings, posEmbeddings)
      );
    });

    if (this.collaborationManager) {
      this.collaborationManager.broadcastStageSnapshot("clip", {
        promptLength: prompt.length,
        negativePromptLength: negPrompt.length
      });
    }

    // use uniform distribution with same variance as normal(0, 1)
    const scale = Math.sqrt(12) / 2;
    let latents = this.tvm.detachFromCurrentScope(
      this.tvm.uniform(latentShape, -scale, scale, this.tvm.webgpu())
    );
    this.tvm.endScope();


    // const image = this.tvm.withNewScope(() => {
    //   return this.tvm.detachFromCurrentScope(
    //     this.vaeToImage(latents, this.vaeParams)
    //   )
    // });

    // await this.device.sync();

    //---------------------------
    // Stage 1: UNet + Scheduler
    //---------------------------
    const embeddingsMessage = new TransferableNDArray(embeddings).toMessage();
    embeddingsMessage.transferList = [embeddingsMessage.buffer];
    let latentsMessage = new TransferableNDArray(latents).toMessage();
    latentsMessage.transferList = [latentsMessage.buffer];
    const unetQueue = new CommandQueue(2);
    const schedulerPrefetchCache = new Map();
    const acquireSchedulerHandle = (index) => {
      if (!schedulerPrefetchCache.has(index)) {
        schedulerPrefetchCache.set(index, scheduler.prefetch(index));
      }
      return schedulerPrefetchCache.get(index);
    };
    const collaboration = this.collaborationManager;

    if (vaeCycle != -1) {
      this.tvm.withNewScope(() => {
        const image = this.vaeToImage(latents, this.vaeParams);
        this.tvm.showImage(this.imageToRGBA(image));
      });
      await this.device.sync();
    }
    vaeCycle = vaeCycle == -1 ? unetNumSteps : vaeCycle;
    let lastSync = undefined;

    for (let counter = 0; counter < unetNumSteps; ++counter) {
      if (progressCallback !== undefined) {
        progressCallback("unet", counter, unetNumSteps, totalNumSteps);
      }
      const timestepHandle = acquireSchedulerHandle(counter);
      const resultMessage = await unetQueue.enqueue(() => this.workerPool.unetWorker.run({
        stage: "unet",
        latentsMessage,
        embeddingsMessage,
        scheduler,
        counter,
        timestepHandle,
        transferList: latentsMessage.transferList
      }));

      latents.dispose();
      latents = TransferableNDArray.fromMessage(this.tvm, this.device, resultMessage);
      latentsMessage = new TransferableNDArray(latents).toMessage();
      latentsMessage.transferList = [latentsMessage.buffer];

      if (collaboration) {
        collaboration.broadcastStageSnapshot("unet", {
          counter,
          total: unetNumSteps
        });
      }

      if (lastSync !== undefined) {
        await lastSync;
      }
      lastSync = this.device.sync();

      if ((counter + 1) < unetNumSteps) {
        unetQueue.enqueue(() => acquireSchedulerHandle(counter + 1));
      }

      if ((counter + 1) % vaeCycle == 0 &&
        (counter + 1) != unetNumSteps &&
        counter >= beginRenderVae) {
        this.tvm.withNewScope(() => {
          const image = this.vaeToImage(latents, this.vaeParams);
          this.tvm.showImage(this.imageToRGBA(image));
        });
        await this.device.sync();
      }
    }
    scheduler.dispose();
    embeddings.dispose();

    await this.device.sync();
    
    // allocate a gpu arr and async copy to it.
    const cpu_arr = this.tvm.withNewScope(() => {
      return this.tvm.detachFromCurrentScope(
        this.tvm.empty(latents.shape, latents.dtype, this.tvm.cpu(0))
      )
    });
    console.log("empty arr" + cpu_arr.toArray());

    cpu_arr.copyFrom(latents);
    await this.tvm.webgpu().sync();

    console.log("final latents" + cpu_arr.toArray());


    //-----------------------------
    // Stage 2: VAE and draw image
    //-----------------------------
    if (progressCallback !== undefined) {
      progressCallback("vae", 0, 1, totalNumSteps);
    }
    const vaeResultMessage = await this.workerPool.vaeWorker.run({
      stage: "vae",
      latentsMessage,
      transferList: latentsMessage.transferList
    });
    this.tvm.withNewScope(() => {
      const rgba = TransferableNDArray.fromMessage(this.tvm, this.device, vaeResultMessage);
      this.tvm.showImage(rgba);
    });
    if (this.collaborationManager) {
      this.collaborationManager.broadcastStageSnapshot("vae", {
        status: "complete"
      });
    }
    latents.dispose();
    await this.device.sync();
    if (progressCallback !== undefined) {
      progressCallback("vae", 1, 1, totalNumSteps);
    }
  }

  clearCanvas() {
    this.tvm.clearCanvas();
  }
};

class DiffusionXLPipeline {
  constructor(tvm, tokenizer1, tokenizer2, schedulerConsts, cacheMetadata) {
    if (cacheMetadata == undefined) {
      throw Error("Expect cacheMetadata");
    }
    this.tvm = tvm;
    this.tokenizer1 = tokenizer1;
    this.tokenizer2 = tokenizer2;
    this.maxTokenLength = 77;

    this.device = this.tvm.webgpu();
    this.tvm.bindCanvas(document.getElementById("canvas"));
    // VM functions
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device)
    );

    this.schedulerConsts = schedulerConsts;
    this.clipToTextEmbeddings1 = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("clip")
    );
    this.clipParams1 = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("clip", cacheMetadata.clipParamSize)
    );
    this.clipToTextEmbeddings2 = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("clip2")
    );
    this.clipParams2 = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("clip2", cacheMetadata.clip2ParamSize)
    );
    this.unetLatentsToNoisePred = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("unet")
    );
    this.unetParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("unet", cacheMetadata.unetParamSize)
    );
    this.vaeToImage = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("vae")
    );
    this.vaeParams = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("vae", cacheMetadata.vaeParamSize)
    );
    this.imageToRGBA = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("image_to_rgba")
    );
    this.concatEmbeddings = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("concat_embeddings")
    );
    this.concatPoolEmbeddings = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("concat_pool_embeddings")
    );
    this.catLatents = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("cat_latents")
    );
    this.concatEncoderOutputs = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("concat_enocder_outputs")
    );
    this.workerPool = new StageWorkerPool(false);
    this.collaborationManager = undefined;
  }

  dispose() {
    // note: tvm instance is not owned by this class
    this.concatEmbeddings.dispose();
    this.imageToRGBA.dispose()
    this.concatPoolEmbeddings.dispose();
    this.catLatents.dispose();
    this.concatEncoderOutputs.dispose();
    this.vaeParams.dispose();
    this.vaeToImage.dispose();
    this.unetParams.dispose();
    this.unetLatentsToNoisePred.dispose();
    this.clipParams1.dispose();
    this.clipToTextEmbeddings1.dispose();
    this.clipParams2.dispose();
    this.clipToTextEmbeddings2.dispose();
    this.vm.dispose();
  }

  setCollaborativeManager(manager) {
    this.collaborationManager = manager;
  }

  tokenize(prompt, tokenizer) {
    const encoded = tokenizer.encode(prompt, true).input_ids;
    const inputIDs = new Int32Array(this.maxTokenLength);

    if (encoded.length < this.maxTokenLength) {
      inputIDs.set(encoded);
      const lastTok = encoded[encoded.length - 1];
      inputIDs.fill(lastTok, encoded.length, inputIDs.length);
    } else {
      inputIDs.set(encoded.slice(0, this.maxTokenLength));
    }
    return this.tvm.empty([1, this.maxTokenLength], "int32", this.device).copyFrom(inputIDs);
  }

  async asyncLoadWebGPUPipelines() {
    await this.tvm.asyncLoadWebGPUPipelines(this.vm.getInternalModule());
  }

  async generate(
    prompt,
    negPrompt = "",
    progressCallback = undefined,
    schedulerId = 0,
    vaeCycle = -1,
    beginRenderVae = 10
  ) {
    this.tvm.beginScope();
    // get latents
    const latentShape = [1, 4, 128, 128];

    var unetNumSteps;
    if (schedulerId == 2) {
      scheduler = new EulerDiscreteScheduler(
        this.schedulerConsts[2], latentShape, this.tvm, this.device, this.vm);
      unetNumSteps = this.schedulerConsts[2]["num_steps"];
    } else {
      //raise error
      throw Error("not supported scheduler");
    }
    const totalNumSteps = unetNumSteps + 2;

    if (progressCallback !== undefined) {
      progressCallback("clip", 0, 1, totalNumSteps);
    }

    const [embeddings, pool_embeddings] = this.tvm.withNewScope(() => {
      let posInputIDs1 = this.tokenize(prompt, this.tokenizer1);
      let posInputIDs2 = this.tokenize(prompt, this.tokenizer2);
      const posEmbeddings1 = this.clipToTextEmbeddings1(
        posInputIDs1, this.clipParams1).get(0);
      const posTemp = this.clipToTextEmbeddings2(
        posInputIDs2, this.clipParams2);
      const posEmbeddings2 = posTemp.get(0);
      const poolPosEmbeddings = posTemp.get(1);

      let negInputIDs1 = this.tokenize(negPrompt, this.tokenizer1);
      let negInputIDs2 = this.tokenize(negPrompt, this.tokenizer2);
      const negEmbeddings1 = this.clipToTextEmbeddings1(
        negInputIDs1, this.clipParams1).get(0);
      const negTemp = this.clipToTextEmbeddings2(
        negInputIDs2, this.clipParams2);
      const negEmbeddings2 = negTemp.get(0);
      const poolNegEmbeddings = negTemp.get(1);
      console.log("posEmbeddings1")
      console.log(posEmbeddings1)
      const posEmbeddings = this.concatEncoderOutputs(posEmbeddings1, posEmbeddings2);
      const negEmbeddings = this.concatEncoderOutputs(negEmbeddings1, negEmbeddings2);

      // maintain new latents
      //TODO: zero out embeddings when negPrompt is empty
      return [this.tvm.detachFromCurrentScope(
        this.concatEmbeddings(negEmbeddings, posEmbeddings)
      ),
      this.tvm.detachFromCurrentScope(
        this.concatPoolEmbeddings(poolNegEmbeddings, poolPosEmbeddings)
      )
      ];
    });
    // use uniform distribution with same variance as normal(0, 1)
    const scale = Math.sqrt(12) / 2 * 13.1585;
    let latents = this.tvm.detachFromCurrentScope(
      this.tvm.uniform(latentShape, -scale, scale, this.tvm.webgpu())
    );
    this.tvm.endScope();
    //---------------------------
    // Stage 1: UNet + Scheduler
    //---------------------------
    if (vaeCycle != -1) {
      // show first frame
      this.tvm.withNewScope(() => {
        const image = this.vaeToImage(latents, this.vaeParams);
        this.tvm.showImage(this.imageToRGBA(image));
      });
      await this.device.sync();
    }
    vaeCycle = vaeCycle == -1 ? unetNumSteps : vaeCycle;
    let lastSync = undefined;

    for (let counter = 0; counter < unetNumSteps; ++counter) {
      if (progressCallback !== undefined) {
        progressCallback("unet", counter, unetNumSteps, totalNumSteps);
      }
      const timestep = scheduler.timestep[counter];
      // recycle noisePred, track latents manually
      const newLatents = this.tvm.withNewScope(() => {
        this.tvm.attachToCurrentScope(latents);
        const latent_model_input = this.catLatents(latents);
        const scaled_latent_model_input = scheduler.scaleModelInput(latent_model_input, counter)
        const array_id = [1024., 1024., 0., 0., 1024., 1024., 1024., 1024., 0., 0., 1024., 1024.]
        let add_time_ids = this.tvm.empty([2, 6], "float32", this.tvm.webgpu()).copyFrom(array_id);
        const noisePred = this.unetLatentsToNoisePred(
          scaled_latent_model_input, timestep, embeddings, 
            pool_embeddings, add_time_ids, this.unetParams);
        // maintain new latents
        return this.tvm.detachFromCurrentScope(
          scheduler.step(noisePred, latents, counter)
        );
      });
      latents = newLatents;
      // use skip one sync, although likely not as useful.
      if (lastSync !== undefined) {
        await lastSync;
      }
      // async event checker
      lastSync = this.device.sync();

      // Optionally, we can draw intermediate result of VAE.
      if ((counter + 1) % vaeCycle == 0 &&
        (counter + 1) != unetNumSteps &&
        counter >= beginRenderVae) {
        this.tvm.withNewScope(() => {
          const image = this.vaeToImage(latents, this.vaeParams);
          this.tvm.showImage(this.imageToRGBA(image));
        });
        await this.device.sync();
      }
    }
    scheduler.dispose();
    embeddings.dispose();
    //-----------------------------
    // Stage 2: VAE and draw image
    //-----------------------------
    if (progressCallback !== undefined) {
      progressCallback("vae", 0, 1, totalNumSteps);
    }
    this.tvm.withNewScope(() => {
      const image = this.vaeToImage(latents, this.vaeParams);
      this.tvm.showImage(this.imageToRGBA(image));
    });
    if (this.collaborationManager) {
      this.collaborationManager.broadcastStageSnapshot("vae", {
        status: "complete"
      });
    }
    latents.dispose();
    await this.device.sync();
    if (progressCallback !== undefined) {
      progressCallback("vae", 1, 1, totalNumSteps);
    }
  }

  clearCanvas() {
    this.tvm.clearCanvas();
  }
};

/**
 * A instance that can be used to facilitate deployment.
 */
class StableDiffusionInstance {
  constructor() {
    this.tvm = undefined;
    this.pipeline = undefined;
    this.config = undefined;
    this.generateInProgress = false;
    this.logger = console.log;
    this.collaborationManager = new CollaborativeGenerationManager((message) => this.logger(message));
    this.model = "Stable-Diffusion-XL"
  }
  /**
   * Initialize TVM
   * @param wasmUrl URL to wasm source.
   * @param cacheUrl URL to NDArray cache.
   * @param logger Custom logger.
   */
  async #asyncInitTVM(wasmUrl, cacheUrl) {
    if (this.tvm !== undefined) {
      return;
    }

    if (document.getElementById("log") !== undefined) {
      this.logger = function (message) {
        console.log(message);
        const d = document.createElement("div");
        d.innerHTML = message;
        document.getElementById("log").appendChild(d);
      };
      if (this.collaborationManager) {
        this.collaborationManager.logger = this.logger;
      }
    }

    const wasmSource = await (
      await fetch(wasmUrl)
    ).arrayBuffer();
    const tvm = await tvmjs.instantiate(
      new Uint8Array(wasmSource),
      new EmccWASI(),
      this.logger
    );
    // initialize WebGPU
    try {
      const output = await tvmjs.detectGPUDevice();
      if (output !== undefined) {
        var label = "WebGPU";
        if (output.adapterInfo.description.length != 0) {
          label += " - " + output.adapterInfo.description;
        } else {
          label += " - " + output.adapterInfo.vendor;
        }
        document.getElementById(
          "gpu-tracker-label").innerHTML = ("Initialize GPU device: " + label);
        tvm.initWebGPU(output.device);
      } else {
        document.getElementById(
          "gpu-tracker-label").innerHTML = "This browser env do not support WebGPU";
        this.reset();
        throw Error("This browser env do not support WebGPU");
      }
    } catch (err) {
      document.getElementById("gpu-tracker-label").innerHTML = (
        "Find an error initializing the WebGPU device " + err.toString()
      );
      console.log(err.stack);
      this.reset();
      throw Error("Find an error initializing WebGPU: " + err.toString());
    }

    this.tvm = tvm;
    function initProgressCallback(report) {
      document.getElementById("progress-tracker-label").innerHTML = report.text;
      document.getElementById("progress-tracker-progress").value = report.progress * 100;
    }
    tvm.registerInitProgressCallback(initProgressCallback);
    if (!cacheUrl.startsWith("http")) {
      cacheUrl = new URL(cacheUrl, document.URL).href;
    }
    await tvm.fetchNDArrayCache(cacheUrl, tvm.webgpu());
  }

  /**
   * Initialize the pipeline
   *
   * @param schedulerConstUrl The scheduler constant.
   * @param tokenizerName The name of the tokenizer.
   */
  async #asyncInitPipeline(schedulerConstUrl, tokenizerName, tokenizerName2) {
    if (this.tvm == undefined) {
      throw Error("asyncInitTVM is not called");
    }
    if (this.pipeline !== undefined) return;
    var schedulerConst = []
    for (let i = 0; i < schedulerConstUrl.length; ++i) {
      schedulerConst.push(await (await fetch(schedulerConstUrl[i])).json())
    }
    if (this.model == "Stable-Diffusion-XL") {
      const tokenizer1 = await tvmjsGlobalEnv.getTokenizer(tokenizerName);
      const tokenizer2 = await tvmjsGlobalEnv.getTokenizer(tokenizerName2);
      this.pipeline = this.tvm.withNewScope(() => {
        return new DiffusionXLPipeline(this.tvm, tokenizer1, tokenizer2, schedulerConst, this.tvm.cacheMetadata);
      });
      if (this.pipeline.setCollaborativeManager !== undefined) {
        this.pipeline.setCollaborativeManager(this.collaborationManager);
      }
      await this.pipeline.asyncLoadWebGPUPipelines();
    }
    else {
      console.log("entered SD pipeline")
      const tokenizer = await tvmjsGlobalEnv.getTokenizer(tokenizerName);
      this.pipeline = this.tvm.withNewScope(() => {
        return new StableDiffusionPipeline(this.tvm, tokenizer, schedulerConst, this.tvm.cacheMetadata);
      });
      this.pipeline.setCollaborativeManager(this.collaborationManager);
      await this.pipeline.asyncLoadWebGPUPipelines();
    }
  }

  /**
   * Async initialize config
   */
  async #asyncInitConfig() {
    if (this.config !== undefined) return;
    this.config = await (await fetch("stable-diffusion-config.json")).json();
  
    var model_param_url = undefined;
    var model_lib_url = undefined;
    if (this.config.model_lib_map[this.model] !== undefined) {
      model_param_url = this.config.param_dict[this.model];
      model_lib_url = this.config.model_lib_map[this.model];
    }
    else{
      throw Error("Model not found");
    }
    this.config.wasmUrl = model_lib_url;
    this.config.cacheUrl = model_param_url;
  }

  /**
   * Function to create progress callback tracker.
   * @returns A progress callback tracker.
   */
  #getProgressCallback() {
    const tstart = performance.now();
    function progressCallback(stage, counter, numSteps, totalNumSteps) {
      const timeElapsed = (performance.now() - tstart) / 1000;
      let text = "Generating ... at stage " + stage;
      if (stage == "unet") {
        counter += 1;
        text += " step [" + counter + "/" + numSteps + "]"
      }
      if (stage == "vae") {
        counter = totalNumSteps;
      }
      text += ", " + Math.ceil(timeElapsed) + " secs elapsed.";
      document.getElementById("progress-tracker-label").innerHTML = text;
      document.getElementById("progress-tracker-progress").value = (counter / totalNumSteps) * 100;
    }
    return progressCallback;
  }

  /**
   * Async initialize instance.
   */
  async asyncInit() {
    if (this.pipeline !== undefined) return;
    await this.#asyncInitConfig();
    await this.#asyncInitTVM(this.config.wasmUrl, this.config.cacheUrl);
    await this.#asyncInitPipeline(this.config.schedulerConstUrl, this.config.tokenizer, this.config.tokenizer2);
    await this.#setupCollaboration();
  }

  async #setupCollaboration() {
    if (!this.collaborationManager || !this.config) {
      return;
    }
    const settings = this.config.collaboration;
    if (!settings) {
      return;
    }
    if (settings.webtransportUrl) {
      await this.collaborationManager.ensureTransport(settings.webtransportUrl);
    }
    if (settings.webrtcConfig) {
      await this.collaborationManager.ensurePeerConnection(settings.webrtcConfig);
    }
    const healthLabelId = settings.healthElementId || "network-health-label";
    if (healthLabelId) {
      const label = document.getElementById(healthLabelId);
      if (label) {
        this.collaborationManager.onHealthUpdate(({ channel, latency }) => {
          label.innerText = `Network ${channel}: ${latency.toFixed(1)}ms`;
        });
      }
    }
  }

  /**
   * Async initialize
   *
   * @param tvm The tvm instance.
   */
  async asyncInitOnRPCServerLoad(tvmInstance) {
    if (this.tvm !== undefined) {
      throw Error("Cannot reuse a loaded instance for rpc");
    }
    this.tvm = tvmInstance;

    this.tvm.beginScope();
    this.tvm.registerAsyncServerFunc("generate", async (prompt, schedulerId, vaeCycle) => {
      document.getElementById("inputPrompt").value = prompt;
      const negPrompt = "";
      document.getElementById("negativePrompt").value = "";
      await this.pipeline.generate(prompt, negPrompt, this.#getProgressCallback(), schedulerId, vaeCycle);
    });
    this.tvm.registerAsyncServerFunc("clearCanvas", async () => {
      this.tvm.clearCanvas();
    });
    this.tvm.registerAsyncServerFunc("showImage", async (data) => {
      this.tvm.showImage(data);
    });
    this.tvm.endScope();
  }

  /**
   * Run generate
   */
  async generate() {
    if (this.requestInProgress) {
      this.logger("Request in progress, generate request ignored");
      return;
    }
    this.requestInProgress = true;
    try {
      await this.asyncInit();
      const prompt = document.getElementById("inputPrompt").value;
      const negPrompt = document.getElementById("negativePrompt").value;
      const schedulerId = document.getElementById("schedulerId").value;
      const vaeCycle = document.getElementById("vaeCycle").value;
      await this.pipeline.generate(prompt, negPrompt, this.#getProgressCallback(), schedulerId, vaeCycle);
    } catch (err) {
      this.logger("Generate error, " + err.toString());
      console.log(err.stack);
      this.reset();
    }
    this.requestInProgress = false;
  }

  /**
   * Reset the instance;
   */
  reset() {
    if (this.pipeline !== undefined) {
      this.pipeline.dispose();
    }
    this.pipeline = undefined;
    if (this.tvm !== undefined) {
      this.tvm.dispose();
      this.tvm = undefined;
    }
    this.config = undefined;
    if (this.collaborationManager) {
      this.collaborationManager.dispose();
      this.collaborationManager = new CollaborativeGenerationManager((message) => this.logger(message));
    }
  }
}

localStableDiffusionInst = new StableDiffusionInstance();

tvmjsGlobalEnv.asyncOnGenerate = async function () {
  await localStableDiffusionInst.generate();
};

tvmjsGlobalEnv.asyncOnRPCServerLoad = async function (tvm) {
  const inst = new StableDiffusionInstance();
  await inst.asyncInitOnRPCServerLoad(tvm);
};

function handle_model_change() {
  var e = document.getElementById("modelId");
  function onChange() {
    localStableDiffusionInst.reset();
    localStableDiffusionInst.model = e.value;
    localStableDiffusionInst.logger("model changed to " + e.value)
  }
  e.onchange = onChange;
}

handle_model_change()