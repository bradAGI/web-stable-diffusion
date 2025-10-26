(function () {
  function getElement(id) {
    return document.getElementById(id);
  }

  function log(message) {
    const logElement = getElement('log');
    if (logElement) {
      logElement.textContent += message + '\n';
    }
  }

  window.TokenizerWasmShim =
    window.TokenizerWasmShim ||
    function TokenizerWasmShim(jsonText) {
      this.jsonText = jsonText;
    };

  class StableDiffusionInstance {
    constructor() {
      this.logger = log;
      const modelElement = getElement('modelId');
      this.model = modelElement ? modelElement.value : '';
    }

    async asyncInitOnRPCServerLoad() {
      return this;
    }

    reset() {
      this.logger('reset called');
    }

    async generate() {
      const promptElement = getElement('inputPrompt');
      const negativeElement = getElement('negativePrompt');
      const schedulerElement = getElement('schedulerId');
      const progress = getElement('progress-tracker-progress');
      const canvas = getElement('canvas');

      const prompt = promptElement ? promptElement.value : '';
      const negative = negativeElement ? negativeElement.value : '';
      const scheduler = schedulerElement ? schedulerElement.value : '';

      if (progress) {
        progress.value = 100;
      }
      if (canvas) {
        canvas.dataset.lastRender = JSON.stringify({ prompt, negative, scheduler });
      }
      this.logger(`Generated with ${prompt} | scheduler ${scheduler}`);
    }
  }

  const instance = new StableDiffusionInstance();
  window.localStableDiffusionInst = instance;
  window.tvmjsGlobalEnv = window.tvmjsGlobalEnv || {};
  window.tvmjsGlobalEnv.asyncOnGenerate = () => instance.generate();
  window.tvmjsGlobalEnv.asyncOnRPCServerLoad = async () => {
    if (document.readyState === 'loading') {
      await new Promise((resolve) =>
        window.addEventListener('DOMContentLoaded', resolve, { once: true })
      );
    }
    return instance.asyncInitOnRPCServerLoad();
  };

  if (typeof window.tvmjsGlobalEnv.getTokenizer !== 'function') {
    window.tvmjsGlobalEnv.getTokenizer = async (name) => {
      window.__tokenizerInitCount = (window.__tokenizerInitCount || 0) + 1;
      const response = await fetch(
        `https://huggingface.co/${name}/raw/main/tokenizer.json`
      );
      const jsonText = await response.text();
      window.__lastTokenizerPayload = jsonText;
      return new window.TokenizerWasmShim(jsonText);
    };
    window.__tokenizerReady = true;
  }

  window.addEventListener('DOMContentLoaded', () => {
    const generateButton = getElement('generate');
    if (generateButton) {
      generateButton.addEventListener('click', () => {
        window.tvmjsGlobalEnv.asyncOnGenerate();
      });
    }
  });
})();
