(function () {
  function getElement(id) {
    return document.getElementById(id);
  }

  function log(message) {
    const logElement = getElement('log');
    if (logElement) {
      logElement.textContent += message + '\n';
    }
    window.__logEntries = window.__logEntries || [];
    window.__logEntries.push(message);
  }

  window.TokenizerWasmShim =
    window.TokenizerWasmShim ||
    function TokenizerWasmShim(jsonText) {
      this.jsonText = jsonText;
    };

  const STREAM_STAGES = [
    { stage: 'seed', progress: 0.25 },
    { stage: 'denoise', progress: 0.6 },
    { stage: 'final', progress: 1.0 },
  ];

  function broadcastFrame(canvas, frame) {
    window.__streamFrames = window.__streamFrames || [];
    window.__streamFrames.push(frame);
    canvas.dataset.lastStream = JSON.stringify(frame);
    canvas.dispatchEvent(new CustomEvent('mock-stream', { detail: frame }));
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    window.__themeChanges = window.__themeChanges || [];
    window.__themeChanges.push({ theme, timestamp: Date.now() });

    document.querySelectorAll('[data-theme-button]').forEach((button) => {
      const isActive = button.dataset.theme === theme;
      button.setAttribute('aria-pressed', String(isActive));
    });
  }

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
        progress.setAttribute('aria-valuenow', '100');
        progress.setAttribute('aria-valuetext', 'Generation complete');
      }
      if (canvas) {
        canvas.dataset.lastRender = JSON.stringify({ prompt, negative, scheduler });
        const theme = document.documentElement.dataset.theme || 'light';
        STREAM_STAGES.forEach((stage, index) => {
          setTimeout(() => {
            broadcastFrame(canvas, {
              ...stage,
              theme,
              timestamp: Date.now(),
            });
          }, index * 15);
        });
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

    document
      .querySelectorAll('[data-theme-button]')
      .forEach((button) =>
        button.addEventListener('click', () => applyTheme(button.dataset.theme))
      );

    if (!document.documentElement.dataset.theme) {
      applyTheme('light');
    }
  });
})();
