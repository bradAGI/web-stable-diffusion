function callUIHook(name, ...args) {
  try {
    if (typeof tvmjsGlobalEnv !== "undefined") {
      const fn = tvmjsGlobalEnv[name];
      if (typeof fn === "function") {
        return fn(...args);
      }
    }
  } catch (err) {
    console.warn("UI hook execution error", name, err);
  }
  return undefined;
}

function setElementText(id, text) {
  if (typeof document === "undefined") return;
  const el = document.getElementById(id);
  if (el !== null && el !== undefined) {
    el.innerHTML = text;
  }
}

function setProgressValue(id, value) {
  if (typeof document === "undefined") return;
  const el = document.getElementById(id);
  if (el !== null && el !== undefined) {
    el.value = value;
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

    if (progressCallback !== undefined) {
      progressCallback("clip", 0, 1, totalNumSteps);
    }

    const embeddings = this.tvm.withNewScope(() => {
      let posInputIDs = this.tokenize(prompt);
      let negInputIDs = this.tokenize(negPrompt);
      const posEmbeddings = this.clipToTextEmbeddings(
        posInputIDs, this.clipParams);
      const negEmbeddings = this.clipToTextEmbeddings(
        negInputIDs, this.clipParams);
      // maintain new latents
      return this.tvm.detachFromCurrentScope(
        this.concatEmbeddings(negEmbeddings, posEmbeddings)
      );
    });
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
        const noisePred = this.unetLatentsToNoisePred(
          latents, timestep, embeddings, this.unetParams);
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
    this.tvm.withNewScope(() => {
      const image = this.vaeToImage(latents, this.vaeParams);
      this.tvm.showImage(this.imageToRGBA(image));
    });
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
    this.requestInProgress = false;
    this.logger = console.log;
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

    const logContainer = (typeof document !== "undefined") ? document.getElementById("log") : undefined;
    this.logger = (message) => {
      console.log(message);
      const handled = callUIHook("logMessage", message);
      if (handled === undefined && logContainer !== undefined && logContainer !== null) {
        const d = document.createElement("div");
        d.innerHTML = message;
        logContainer.appendChild(d);
      }
    };

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
        const status = "Initialize GPU device: " + label;
        if (callUIHook("updateGpuStatus", status) === undefined) {
          setElementText("gpu-tracker-label", status);
        }
        tvm.initWebGPU(output.device);
      } else {
        const status = "This browser env do not support WebGPU";
        if (callUIHook("updateGpuStatus", status) === undefined) {
          setElementText("gpu-tracker-label", status);
        }
        this.reset();
        throw Error("This browser env do not support WebGPU");
      }
    } catch (err) {
      const status = "Find an error initializing the WebGPU device " + err.toString();
      if (callUIHook("updateGpuStatus", status) === undefined) {
        setElementText("gpu-tracker-label", status);
      }
      console.log(err.stack);
      this.reset();
      throw Error("Find an error initializing WebGPU: " + err.toString());
    }

    this.tvm = tvm;
    function initProgressCallback(report) {
      const handled = callUIHook("updateProgress", report);
      if (handled === undefined) {
        setElementText("progress-tracker-label", report.text);
        setProgressValue("progress-tracker-progress", report.progress * 100);
      }
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
      await this.pipeline.asyncLoadWebGPUPipelines();
    }
    else {
      console.log("entered SD pipeline")
      const tokenizer = await tvmjsGlobalEnv.getTokenizer(tokenizerName);
      this.pipeline = this.tvm.withNewScope(() => {
        return new StableDiffusionPipeline(this.tvm, tokenizer, schedulerConst, this.tvm.cacheMetadata);
      });
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

  #getInputValue(elementId) {
    if (typeof document === "undefined") {
      return "";
    }
    const el = document.getElementById(elementId);
    if (el === null || el === undefined) {
      return "";
    }
    return el.value;
  }

  #reportValidationError(field, message) {
    callUIHook("reportValidationError", { field: field, message: message });
    this.logger(message);
  }

  #validatePrompt(prompt) {
    if (prompt === undefined || prompt === null || prompt.trim().length === 0) {
      const error = new Error("Prompt cannot be empty.");
      error.validation = true;
      this.#reportValidationError("prompt", "Prompt cannot be empty.");
      throw error;
    }
    if (prompt.length > 1000) {
      const error = new Error("Prompt too long.");
      error.validation = true;
      this.#reportValidationError("prompt", "Prompt is limited to 1000 characters.");
      throw error;
    }
    return prompt;
  }

  #validateNegativePrompt(prompt) {
    if (prompt !== undefined && prompt !== null && prompt.length > 1000) {
      const error = new Error("Negative prompt too long.");
      error.validation = true;
      this.#reportValidationError("negativePrompt", "Negative prompt is limited to 1000 characters.");
      throw error;
    }
    return prompt ?? "";
  }

  #resolveSchedulerId(schedulerId) {
    const schedulerSupport = {
      "Stable-Diffusion-XL": ["2"],
      "Stable-Diffusion-1.5": ["0", "1"],
    };
    const supported = schedulerSupport[this.model] ?? ["0"];
    if (supported.includes(schedulerId)) {
      return schedulerId;
    }
    const fallback = supported.length > 0 ? supported[0] : schedulerId ?? "0";
    const payload = {
      previous: schedulerId,
      fallback: fallback,
      model: this.model,
    };
    callUIHook("onSchedulerFallback", payload);
    this.logger(
      "Scheduler " + schedulerId + " not valid for model " + this.model + ", falling back to " + fallback
    );
    callUIHook("setSchedulerId", fallback);
    if (typeof document !== "undefined") {
      const select = document.getElementById("schedulerId");
      if (select !== null && select !== undefined) {
        select.value = fallback;
      }
    }
    return fallback;
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
      const report = {
        text: text,
        progress: counter / totalNumSteps,
        stage: stage,
        counter: counter,
        numSteps: numSteps,
        totalNumSteps: totalNumSteps,
      };
      const handled = callUIHook("updateProgress", report);
      if (handled === undefined) {
        setElementText("progress-tracker-label", text);
        setProgressValue("progress-tracker-progress", (counter / totalNumSteps) * 100);
      }
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
      callUIHook("setPrompts", { prompt: prompt, negativePrompt: "" });
      if (typeof document !== "undefined") {
        const promptEl = document.getElementById("inputPrompt");
        if (promptEl !== null && promptEl !== undefined) {
          promptEl.value = prompt;
        }
        const negPromptEl = document.getElementById("negativePrompt");
        if (negPromptEl !== null && negPromptEl !== undefined) {
          negPromptEl.value = "";
        }
      }
      await this.pipeline.generate(prompt, "", this.#getProgressCallback(), schedulerId, vaeCycle);
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
    callUIHook("onGenerationLifecycle", "start");
    callUIHook("clearValidationErrors");
    try {
      await this.asyncInit();
      const promptCandidate = callUIHook("getPrompt");
      const negPromptCandidate = callUIHook("getNegativePrompt");
      const schedulerCandidate = callUIHook("getSchedulerId");
      const vaeCandidate = callUIHook("getVaeCycle");
      const prompt = this.#validatePrompt(
        promptCandidate !== undefined ? promptCandidate : this.#getInputValue("inputPrompt")
      );
      const negPrompt = this.#validateNegativePrompt(
        negPromptCandidate !== undefined ? negPromptCandidate : this.#getInputValue("negativePrompt")
      );
      const schedulerId = this.#resolveSchedulerId(
        schedulerCandidate !== undefined ? schedulerCandidate : this.#getInputValue("schedulerId")
      );
      const vaeCycle =
        vaeCandidate !== undefined ? vaeCandidate : this.#getInputValue("vaeCycle");
      await this.pipeline.generate(prompt, negPrompt, this.#getProgressCallback(), schedulerId, vaeCycle);
      callUIHook("onGenerationLifecycle", "complete");
    } catch (err) {
      if (!(err !== undefined && err !== null && err.validation === true)) {
        this.logger("Generate error, " + err.toString());
        if (err !== undefined && err !== null && err.stack !== undefined) {
          console.log(err.stack);
        }
        this.reset();
        callUIHook("onGenerationLifecycle", "error");
      } else {
        callUIHook("onGenerationLifecycle", "validation-error");
      }
    }
    this.requestInProgress = false;
    callUIHook("onGenerationLifecycle", "end");
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
  function notifyModelChange(value) {
    if (value === undefined) {
      return;
    }
    localStableDiffusionInst.reset();
    localStableDiffusionInst.model = value;
    localStableDiffusionInst.logger("model changed to " + value);
  }
  if (typeof tvmjsGlobalEnv !== "undefined") {
    tvmjsGlobalEnv.notifyModelSelection = notifyModelChange;
  }
  notifyModelChange(localStableDiffusionInst.model);
  if (typeof document !== "undefined") {
    const e = document.getElementById("modelId");
    if (e !== null && e !== undefined) {
      e.onchange = function () {
        notifyModelChange(e.value);
      };
    }
  }
}

handle_model_change();
