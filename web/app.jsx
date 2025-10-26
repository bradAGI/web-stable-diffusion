const { useState, useEffect, useMemo, useCallback, useRef } = React;

const PROMPT_MAX = 1000;
const NEGATIVE_MAX = 1000;

const translations = {
  en: {
    title: "Web Stable Diffusion",
    subtitle: "Generate images with accessible controls and live guidance.",
    promptLabel: "Prompt",
    promptPlaceholder: "Describe the image you want to generate.",
    promptHelp: "Provide a descriptive prompt to guide the image generation.",
    promptRequired: "Prompt is required.",
    promptTooLong: (max) => `Prompt must be ${max} characters or fewer.`,
    promptLimitInfo: (max) => `Prompts support up to ${max} characters.`,
    charCount: (current, max) => `${current}/${max} characters`,
    negativePromptLabel: "Negative prompt",
    negativePromptPlaceholder: "Optional: elements you want to avoid.",
    negativePromptHelp: "Add optional instructions for what the model should avoid.",
    negativePromptTooLong: (max) => `Negative prompt must be ${max} characters or fewer.`,
    negativePromptLimitInfo: (max) => `Negative prompts support up to ${max} characters.`,
    negativeCharCount: (current, max) => `${current}/${max} characters`,
    modelLabel: "Model",
    modelNames: {
      "Stable-Diffusion-XL": "Stable Diffusion XL",
      "Stable-Diffusion-1.5": "Stable Diffusion 1.5",
    },
    schedulerLabel: "Scheduler",
    schedulerHelp: "Schedulers control the denoising strategy. Availability depends on the selected model.",
    schedulerMultistep: "Multi-step DPM Solver (20 steps)",
    schedulerPNDM: "PNDM (50 steps)",
    schedulerEuler: "Euler Discrete Scheduler",
    schedulerFallbackMessage: (previous, fallback, model) => `Scheduler ${previous} is not available for ${model}. Switched to ${fallback}.`,
    vaeLabel: "Render intermediate steps",
    vaeNone: "No",
    vaeEveryTwo: "Run VAE every two UNet steps after step 10",
    vaeHelp: "Rendering intermediate steps may slow the process but provides more feedback during generation.",
    themeLabel: "Theme",
    lightTheme: "Light",
    darkTheme: "Dark",
    customTheme: "Custom",
    paletteInstructions: "Adjust the colors when using the custom theme.",
    paletteHelp: "Pick colors that match your preferences or accessibility needs.",
    accentLabel: "Accent color",
    backgroundLabel: "Background color",
    surfaceLabel: "Surface color",
    languageLabel: "Language",
    localizationHelp: "Switch languages without reloading the page.",
    help: "Help",
    openHelp: "Open help dialog",
    close: "Close",
    helpTitle: "Keyboard shortcuts & tips",
    helpShortcutsDescription: "Use these shortcuts to work efficiently.",
    helpKeyboard: "Keyboard shortcuts",
    helpShortcutGenerate: "Generate image",
    helpShortcutHelp: "Toggle help",
    helpShortcutTheme: "Cycle theme",
    helpAccessibility: "All controls are accessible via keyboard and screen readers.",
    helpPrompt: "Describe what you want to see in the generated image. Be specific for best results.",
    helpNegativePrompt: "List items or qualities you want to avoid in the generated image.",
    helpContextualTips: "Open the contextual help button next to each section to learn more.",
    generate: "Generate",
    generating: "Generating…",
    generatorReady: "Ready to generate images.",
    gpuStatus: "GPU status",
    progressStatus: "Progress",
    progressPreparing: "Preparing pipeline…",
    fallbackLabel: "Compatibility update",
    fallbackAnnouncement: "Scheduler updated for compatibility.",
    validationSummary: "Validation feedback",
    logsTitle: "Activity log",
    logsEmpty: "No log entries yet. Actions and status messages will appear here.",
    clearLogs: "Clear log",
    customizingPalette: "Customize palette",
    progressAria: (percent) => `Generation ${Math.round(percent)} percent complete.`,
  },
  es: {
    title: "Difusión Estable en la Web",
    subtitle: "Genera imágenes con controles accesibles y guía en vivo.",
    promptLabel: "Indicaciones",
    promptPlaceholder: "Describe la imagen que deseas generar.",
    promptHelp: "Escribe una indicación descriptiva para guiar la generación de la imagen.",
    promptRequired: "La indicación es obligatoria.",
    promptTooLong: (max) => `La indicación debe tener ${max} caracteres o menos.`,
    promptLimitInfo: (max) => `Las indicaciones admiten hasta ${max} caracteres.`,
    charCount: (current, max) => `${current}/${max} caracteres`,
    negativePromptLabel: "Indicaciones negativas",
    negativePromptPlaceholder: "Opcional: elementos que quieres evitar.",
    negativePromptHelp: "Agrega instrucciones opcionales sobre lo que el modelo debe evitar.",
    negativePromptTooLong: (max) => `La indicación negativa debe tener ${max} caracteres o menos.`,
    negativePromptLimitInfo: (max) => `Las indicaciones negativas admiten hasta ${max} caracteres.`,
    negativeCharCount: (current, max) => `${current}/${max} caracteres`,
    modelLabel: "Modelo",
    modelNames: {
      "Stable-Diffusion-XL": "Stable Diffusion XL",
      "Stable-Diffusion-1.5": "Stable Diffusion 1.5",
    },
    schedulerLabel: "Programador",
    schedulerHelp: "Los programadores controlan la estrategia de eliminación de ruido. Su disponibilidad depende del modelo seleccionado.",
    schedulerMultistep: "Solucionador DPM multietapa (20 pasos)",
    schedulerPNDM: "PNDM (50 pasos)",
    schedulerEuler: "Programador discreto Euler",
    schedulerFallbackMessage: (previous, fallback, model) => `El programador ${previous} no está disponible para ${model}. Se cambió a ${fallback}.`,
    vaeLabel: "Renderizar pasos intermedios",
    vaeNone: "No",
    vaeEveryTwo: "Ejecutar VAE cada dos pasos de UNet después del paso 10",
    vaeHelp: "Renderizar pasos intermedios puede ralentizar la generación, pero ofrece más retroalimentación.",
    themeLabel: "Tema",
    lightTheme: "Claro",
    darkTheme: "Oscuro",
    customTheme: "Personalizado",
    paletteInstructions: "Ajusta los colores cuando uses el tema personalizado.",
    paletteHelp: "Elige colores que se adapten a tus preferencias o necesidades de accesibilidad.",
    accentLabel: "Color de acento",
    backgroundLabel: "Color de fondo",
    surfaceLabel: "Color de superficie",
    languageLabel: "Idioma",
    localizationHelp: "Cambia de idioma sin recargar la página.",
    help: "Ayuda",
    openHelp: "Abrir diálogo de ayuda",
    close: "Cerrar",
    helpTitle: "Atajos de teclado y consejos",
    helpShortcutsDescription: "Usa estos atajos para trabajar con rapidez.",
    helpKeyboard: "Atajos de teclado",
    helpShortcutGenerate: "Generar imagen",
    helpShortcutHelp: "Mostrar/ocultar ayuda",
    helpShortcutTheme: "Cambiar tema",
    helpAccessibility: "Todos los controles son accesibles mediante teclado y lectores de pantalla.",
    helpPrompt: "Describe lo que quieres ver en la imagen generada. Sé específico para obtener mejores resultados.",
    helpNegativePrompt: "Enumera los elementos o cualidades que deseas evitar en la imagen generada.",
    helpContextualTips: "Abre el botón de ayuda contextual junto a cada sección para obtener más información.",
    generate: "Generar",
    generating: "Generando…",
    generatorReady: "Listo para generar imágenes.",
    gpuStatus: "Estado de la GPU",
    progressStatus: "Progreso",
    progressPreparing: "Preparando la canalización…",
    fallbackLabel: "Actualización de compatibilidad",
    fallbackAnnouncement: "Programador actualizado por compatibilidad.",
    validationSummary: "Comentarios de validación",
    logsTitle: "Registro de actividad",
    logsEmpty: "Aún no hay entradas. Aquí aparecerán las acciones y los mensajes de estado.",
    clearLogs: "Borrar registro",
    customizingPalette: "Personalizar paleta",
    progressAria: (percent) => `Generación completada al ${Math.round(percent)} por ciento.`,
  },
};

const schedulerDefinitions = [
  { value: "0", models: ["Stable-Diffusion-1.5"], labelKey: "schedulerMultistep" },
  { value: "1", models: ["Stable-Diffusion-1.5"], labelKey: "schedulerPNDM" },
  { value: "2", models: ["Stable-Diffusion-XL"], labelKey: "schedulerEuler" },
];

const vaeDefinitions = [
  { value: "-1", labelKey: "vaeNone" },
  { value: "2", labelKey: "vaeEveryTwo" },
];

const modelOptions = ["Stable-Diffusion-XL", "Stable-Diffusion-1.5"];

function ContextualHelpButton({ label, onClick }) {
  return (
    <button
      type="button"
      className="help-button"
      onClick={onClick}
      aria-label={label}
    >
      <span aria-hidden="true">?</span>
    </button>
  );
}

function HelpModal({ isOpen, onClose, content, i18n }) {
  const closeRef = useRef(null);
  useEffect(() => {
    if (isOpen && closeRef.current) {
      closeRef.current.focus();
    }
  }, [isOpen]);

  if (!isOpen || !content) {
    return null;
  }

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="help-modal-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h2 id="help-modal-title">{content.title || i18n.helpTitle}</h2>
          <button
            type="button"
            className="icon-button"
            onClick={onClose}
            aria-label={i18n.close}
            ref={closeRef}
          >
            ×
          </button>
        </div>
        <div className="modal-body">
          {content.description && <p>{content.description}</p>}
          {content.items && content.items.map((item) => (
            <section key={item.title} className="modal-section">
              <h3>{item.title}</h3>
              {item.content}
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

function App() {
  const [language, setLanguage] = useState(() => {
    try {
      return window.localStorage.getItem("wsd-language") || "en";
    } catch (err) {
      return "en";
    }
  });
  const i18n = useMemo(() => translations[language] || translations.en, [language]);

  const [prompt, setPrompt] = useState("A photo of an astronaut riding a horse on mars");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [model, setModel] = useState("Stable-Diffusion-XL");
  const [scheduler, setScheduler] = useState("2");
  const [vaeCycle, setVaeCycle] = useState("-1");
  const [theme, setTheme] = useState(() => {
    try {
      return window.localStorage.getItem("wsd-theme") || "dark";
    } catch (err) {
      return "dark";
    }
  });
  const [accentColor, setAccentColor] = useState(() => {
    try {
      return window.localStorage.getItem("wsd-accent") || "#6366f1";
    } catch (err) {
      return "#6366f1";
    }
  });
  const [customBackground, setCustomBackground] = useState(() => {
    try {
      return window.localStorage.getItem("wsd-custom-background") || "#111827";
    } catch (err) {
      return "#111827";
    }
  });
  const [customSurface, setCustomSurface] = useState(() => {
    try {
      return window.localStorage.getItem("wsd-custom-surface") || "#1f2937";
    } catch (err) {
      return "#1f2937";
    }
  });
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [helpContext, setHelpContext] = useState(null);
  const [progressState, setProgressState] = useState({ text: "", progress: 0, stage: "" });
  const [gpuStatus, setGpuStatus] = useState("");
  const [statusMessage, setStatusMessage] = useState(i18n.generatorReady);
  const [logs, setLogs] = useState([]);
  const [validationErrors, setValidationErrors] = useState({});
  const [schedulerFallback, setSchedulerFallback] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const promptRef = useRef(prompt);
  const negativePromptRef = useRef(negativePrompt);
  const schedulerRef = useRef(scheduler);
  const vaeRef = useRef(vaeCycle);

  useEffect(() => {
    promptRef.current = prompt;
  }, [prompt]);
  useEffect(() => {
    negativePromptRef.current = negativePrompt;
  }, [negativePrompt]);
  useEffect(() => {
    schedulerRef.current = scheduler;
  }, [scheduler]);
  useEffect(() => {
    vaeRef.current = vaeCycle;
  }, [vaeCycle]);

  useEffect(() => {
    document.documentElement.lang = language;
    try {
      window.localStorage.setItem("wsd-language", language);
    } catch (err) {
      // ignore storage errors
    }
  }, [language]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.setProperty("--custom-accent-color", accentColor);
    document.documentElement.style.setProperty("--custom-background-color", customBackground);
    document.documentElement.style.setProperty("--custom-surface-color", customSurface);
    try {
      window.localStorage.setItem("wsd-theme", theme);
      window.localStorage.setItem("wsd-accent", accentColor);
      window.localStorage.setItem("wsd-custom-background", customBackground);
      window.localStorage.setItem("wsd-custom-surface", customSurface);
    } catch (err) {
      // ignore storage errors
    }
  }, [theme, accentColor, customBackground, customSurface]);

  useEffect(() => {
    setStatusMessage(i18n.generatorReady);
  }, [i18n]);

  const addLogEntry = useCallback((message, level = "info") => {
    setLogs((previous) => {
      const timestamp = Date.now();
      const entry = { id: `${timestamp}-${Math.random()}`, message, level, timestamp };
      const next = [...previous.slice(-199), entry];
      return next;
    });
  }, []);

  const getSchedulerLabel = useCallback((value) => {
    const def = schedulerDefinitions.find((option) => option.value === value);
    if (!def) {
      return value;
    }
    return i18n[def.labelKey] || value;
  }, [i18n]);

  const availableSchedulers = useMemo(
    () => schedulerDefinitions.filter((option) => option.models.includes(model)),
    [model]
  );

  useEffect(() => {
    if (availableSchedulers.length > 0 && !availableSchedulers.some((item) => item.value === scheduler)) {
      const fallback = availableSchedulers[0];
      const previousLabel = getSchedulerLabel(scheduler);
      const fallbackLabel = getSchedulerLabel(fallback.value);
      setScheduler(fallback.value);
      schedulerRef.current = fallback.value;
      setSchedulerFallback({
        previous: previousLabel,
        fallback: fallbackLabel,
        model: i18n.modelNames[model] || model,
        timestamp: Date.now(),
      });
      setStatusMessage(
        i18n.schedulerFallbackMessage(
          previousLabel,
          fallbackLabel,
          i18n.modelNames[model] || model
        )
      );
    }
  }, [availableSchedulers, scheduler, getSchedulerLabel, i18n, model]);

  useEffect(() => {
    const env = window.tvmjsGlobalEnv || (window.tvmjsGlobalEnv = {});
    env.getPrompt = () => promptRef.current;
    env.getNegativePrompt = () => negativePromptRef.current;
    env.getSchedulerId = () => schedulerRef.current;
    env.getVaeCycle = () => vaeRef.current;
    env.setPrompts = (payload) => {
      if (payload && typeof payload.prompt === "string") {
        setPrompt(payload.prompt);
      }
      if (payload && typeof payload.negativePrompt === "string") {
        setNegativePrompt(payload.negativePrompt);
      }
      return true;
    };
    env.setSchedulerId = (value) => {
      if (typeof value === "string") {
        setScheduler(value);
        schedulerRef.current = value;
      }
      return true;
    };
    env.updateProgress = (report) => {
      setProgressState(report || { text: "", progress: 0, stage: "" });
      if (report && typeof report.progress === "number") {
        setIsGenerating(report.progress < 1);
      }
      if (report && report.text) {
        setStatusMessage(report.text);
      }
      return true;
    };
    env.updateGpuStatus = (message) => {
      setGpuStatus(message);
      if (message) {
        setStatusMessage(message);
      }
      return true;
    };
    env.logMessage = (message, level = "info") => {
      addLogEntry(message, level);
      return true;
    };
    env.onSchedulerFallback = (payload) => {
      if (!payload) {
        return true;
      }
      const previousLabel = getSchedulerLabel(payload.previous);
      const fallbackLabel = getSchedulerLabel(payload.fallback);
      setSchedulerFallback({
        previous: previousLabel,
        fallback: fallbackLabel,
        model: i18n.modelNames[payload.model] || payload.model,
        timestamp: Date.now(),
      });
      setStatusMessage(
        i18n.schedulerFallbackMessage(
          previousLabel,
          fallbackLabel,
          i18n.modelNames[payload.model] || payload.model
        )
      );
      return true;
    };
    env.reportValidationError = (payload) => {
      if (payload && payload.field) {
        setValidationErrors((previous) => ({ ...previous, [payload.field]: payload.message }));
      }
      return true;
    };
    env.clearValidationErrors = () => {
      setValidationErrors({});
      return true;
    };
    env.onGenerationLifecycle = (stage) => {
      if (stage === "start") {
        setIsGenerating(true);
        setStatusMessage(i18n.progressPreparing);
        setProgressState((previous) => ({ ...previous, text: i18n.progressPreparing, progress: 0 }));
      } else if (stage === "complete") {
        setStatusMessage(i18n.generatorReady);
      } else if (stage === "error") {
        setIsGenerating(false);
      } else if (stage === "validation-error") {
        setIsGenerating(false);
        setStatusMessage(i18n.validationSummary);
      } else if (stage === "end") {
        setIsGenerating(false);
      }
      return true;
    };
    if (typeof env.notifyModelSelection === "function") {
      env.notifyModelSelection(model);
    }
    return () => {
      env.getPrompt = undefined;
      env.getNegativePrompt = undefined;
      env.getSchedulerId = undefined;
      env.getVaeCycle = undefined;
      env.setPrompts = undefined;
      env.setSchedulerId = undefined;
      env.updateProgress = undefined;
      env.updateGpuStatus = undefined;
      env.logMessage = undefined;
      env.onSchedulerFallback = undefined;
      env.reportValidationError = undefined;
      env.clearValidationErrors = undefined;
      env.onGenerationLifecycle = undefined;
    };
  }, [addLogEntry, getSchedulerLabel, i18n, model]);

  const previousModelRef = useRef(model);
  useEffect(() => {
    const env = window.tvmjsGlobalEnv;
    if (!env || typeof env.notifyModelSelection !== "function") {
      previousModelRef.current = model;
      return;
    }
    if (previousModelRef.current !== model) {
      env.notifyModelSelection(model);
      previousModelRef.current = model;
    }
  }, [model]);

  const cycleTheme = useCallback(() => {
    setTheme((current) => {
      if (current === "light") return "dark";
      if (current === "dark") return "custom";
      return "light";
    });
  }, []);

  const handleGenerate = useCallback(
    async (event) => {
      if (event) {
        event.preventDefault();
      }
      setValidationErrors({});
      setSchedulerFallback(null);
      if (!prompt.trim()) {
        const message = i18n.promptRequired;
        setValidationErrors({ prompt: message });
        addLogEntry(message, "warn");
        setStatusMessage(i18n.validationSummary);
        return;
      }
      if (prompt.length > PROMPT_MAX) {
        const message = i18n.promptTooLong(PROMPT_MAX);
        setValidationErrors({ prompt: message });
        addLogEntry(message, "warn");
        setStatusMessage(i18n.validationSummary);
        return;
      }
      if (negativePrompt.length > NEGATIVE_MAX) {
        const message = i18n.negativePromptTooLong(NEGATIVE_MAX);
        setValidationErrors({ negativePrompt: message });
        addLogEntry(message, "warn");
        setStatusMessage(i18n.validationSummary);
        return;
      }
      try {
        setIsGenerating(true);
        setStatusMessage(i18n.progressPreparing);
        const env = window.tvmjsGlobalEnv;
        if (env && typeof env.asyncOnGenerate === "function") {
          await env.asyncOnGenerate();
        } else {
          const message = "Generation API not ready.";
          addLogEntry(message, "error");
          setStatusMessage(message);
          setIsGenerating(false);
        }
      } catch (error) {
        const message = error && error.message ? error.message : "Generation failed.";
        addLogEntry(message, "error");
        setStatusMessage(message);
        setIsGenerating(false);
      }
    },
    [prompt, negativePrompt, i18n, addLogEntry]
  );

  useEffect(() => {
    function handleKeyDown(event) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "enter") {
        event.preventDefault();
        handleGenerate();
      } else if (event.shiftKey && event.key === "?") {
        event.preventDefault();
        if (isHelpOpen && helpContext && helpContext.type === "general") {
          setIsHelpOpen(false);
        } else {
          setHelpContext({ type: "general" });
          setIsHelpOpen(true);
        }
      } else if (event.altKey && event.key.toLowerCase() === "t") {
        event.preventDefault();
        cycleTheme();
      } else if (event.key === "Escape" && isHelpOpen) {
        setIsHelpOpen(false);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleGenerate, cycleTheme, isHelpOpen, helpContext]);

  const helpContent = useMemo(() => {
    if (!helpContext) {
      return null;
    }
    switch (helpContext.type) {
      case "prompt":
        return {
          title: i18n.promptLabel,
          items: [
            { title: i18n.promptLabel, content: <p>{i18n.helpPrompt}</p> },
            { title: i18n.validationSummary, content: <p>{i18n.promptLimitInfo(PROMPT_MAX)}</p> },
          ],
        };
      case "negativePrompt":
        return {
          title: i18n.negativePromptLabel,
          items: [
            { title: i18n.negativePromptLabel, content: <p>{i18n.helpNegativePrompt}</p> },
            {
              title: i18n.validationSummary,
              content: <p>{i18n.negativePromptLimitInfo(NEGATIVE_MAX)}</p>,
            },
          ],
        };
      case "scheduler":
        return {
          title: i18n.schedulerLabel,
          items: [
            { title: i18n.schedulerLabel, content: <p>{i18n.schedulerHelp}</p> },
            { title: i18n.fallbackLabel, content: <p>{i18n.fallbackAnnouncement}</p> },
          ],
        };
      case "vae":
        return {
          title: i18n.vaeLabel,
          items: [{ title: i18n.vaeLabel, content: <p>{i18n.vaeHelp}</p> }],
        };
      case "theme":
        return {
          title: i18n.themeLabel,
          items: [
            { title: i18n.themeLabel, content: <p>{i18n.paletteHelp}</p> },
            { title: i18n.customizingPalette, content: <p>{i18n.paletteInstructions}</p> },
          ],
        };
      case "language":
        return {
          title: i18n.languageLabel,
          items: [{ title: i18n.languageLabel, content: <p>{i18n.localizationHelp}</p> }],
        };
      case "general":
        return {
          title: i18n.helpTitle,
          description: i18n.helpShortcutsDescription,
          items: [
            {
              title: i18n.helpKeyboard,
              content: (
                <ul className="shortcut-list">
                  <li>
                    <kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>Enter</kbd> — {i18n.helpShortcutGenerate}
                  </li>
                  <li>
                    <kbd>Shift</kbd> + <kbd>?</kbd> — {i18n.helpShortcutHelp}
                  </li>
                  <li>
                    <kbd>Alt</kbd> + <kbd>T</kbd> — {i18n.helpShortcutTheme}
                  </li>
                </ul>
              ),
            },
            { title: i18n.validationSummary, content: <p>{i18n.helpAccessibility}</p> },
            { title: i18n.help, content: <p>{i18n.helpContextualTips}</p> },
          ],
        };
      default:
        return null;
    }
  }, [helpContext, i18n]);

  const progressPercent = Math.max(0, Math.min(1, progressState.progress || 0)) * 100;
  const progressText = progressState.text || (isGenerating ? i18n.progressPreparing : i18n.generatorReady);
  const progressAriaLabel = i18n.progressAria(progressPercent);

  const timeFormatter = useMemo(() => {
    try {
      return new Intl.DateTimeFormat(language === "es" ? "es" : "en", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    } catch (err) {
      return {
        format: (date) => new Date(date).toLocaleTimeString(),
      };
    }
  }, [language]);

  const openHelp = useCallback((type) => {
    setHelpContext({ type });
    setIsHelpOpen(true);
  }, []);

  return (
    <div className="app" data-generating={isGenerating}>
      <header className="toolbar" role="banner">
        <div className="brand">
          <h1>{i18n.title}</h1>
          <p>{i18n.subtitle}</p>
        </div>
        <div className="toolbar-controls">
          <div className="language-field">
            <label htmlFor="language-select">{i18n.languageLabel}</label>
            <ContextualHelpButton
              label={`${i18n.openHelp}: ${i18n.languageLabel}`}
              onClick={() => openHelp("language")}
            />
            <select
              id="language-select"
              value={language}
              onChange={(event) => setLanguage(event.target.value)}
            >
              <option value="en">English</option>
              <option value="es">Español</option>
            </select>
          </div>
          <button
            type="button"
            className="primary ghost"
            onClick={() => {
              openHelp("general");
            }}
            aria-label={i18n.openHelp}
          >
            {i18n.help}
          </button>
        </div>
      </header>
      <main className="layout" aria-live="polite">
        <section className="control-panel">
          <form className="control-form" onSubmit={handleGenerate} aria-busy={isGenerating}>
            <fieldset className="field-group">
              <legend>
                {i18n.promptLabel}
                <ContextualHelpButton
                  label={`${i18n.openHelp}: ${i18n.promptLabel}`}
                  onClick={() => openHelp("prompt")}
                />
              </legend>
              <textarea
                id="inputPrompt"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder={i18n.promptPlaceholder}
                aria-describedby="prompt-helper"
                aria-invalid={Boolean(validationErrors.prompt)}
                maxLength={PROMPT_MAX}
              />
              <div className="field-helper" id="prompt-helper">
                <span>{i18n.promptHelp}</span>
                <span>{i18n.charCount(prompt.length, PROMPT_MAX)}</span>
              </div>
              {validationErrors.prompt && (
                <p className="field-error" role="alert">{validationErrors.prompt}</p>
              )}
            </fieldset>
            <fieldset className="field-group">
              <legend>
                {i18n.negativePromptLabel}
                <ContextualHelpButton
                  label={`${i18n.openHelp}: ${i18n.negativePromptLabel}`}
                  onClick={() => openHelp("negativePrompt")}
                />
              </legend>
              <textarea
                id="negativePrompt"
                value={negativePrompt}
                onChange={(event) => setNegativePrompt(event.target.value)}
                placeholder={i18n.negativePromptPlaceholder}
                aria-describedby="negative-helper"
                aria-invalid={Boolean(validationErrors.negativePrompt)}
                maxLength={NEGATIVE_MAX}
              />
              <div className="field-helper" id="negative-helper">
                <span>{i18n.negativePromptHelp}</span>
                <span>{i18n.negativeCharCount(negativePrompt.length, NEGATIVE_MAX)}</span>
              </div>
              {validationErrors.negativePrompt && (
                <p className="field-error" role="alert">{validationErrors.negativePrompt}</p>
              )}
            </fieldset>
            <div className="field-row">
              <div className="field-group compact">
                <label htmlFor="modelId">{i18n.modelLabel}</label>
                <select
                  id="modelId"
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                >
                  {modelOptions.map((option) => (
                    <option key={option} value={option}>
                      {i18n.modelNames[option] || option}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field-group compact">
                <label htmlFor="schedulerId">
                  {i18n.schedulerLabel}
                  <ContextualHelpButton
                    label={`${i18n.openHelp}: ${i18n.schedulerLabel}`}
                    onClick={() => openHelp("scheduler")}
                  />
                </label>
                <select
                  id="schedulerId"
                  value={scheduler}
                  onChange={(event) => setScheduler(event.target.value)}
                >
                  {availableSchedulers.map((option) => (
                    <option key={option.value} value={option.value}>
                      {i18n[option.labelKey]}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="field-group compact">
              <label htmlFor="vaeCycle">
                {i18n.vaeLabel}
                <ContextualHelpButton
                  label={`${i18n.openHelp}: ${i18n.vaeLabel}`}
                  onClick={() => openHelp("vae")}
                />
              </label>
              <select
                id="vaeCycle"
                value={vaeCycle}
                onChange={(event) => setVaeCycle(event.target.value)}
              >
                {vaeDefinitions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {i18n[option.labelKey]}
                  </option>
                ))}
              </select>
            </div>
            <fieldset className="field-group">
              <legend>
                {i18n.themeLabel}
                <ContextualHelpButton
                  label={`${i18n.openHelp}: ${i18n.themeLabel}`}
                  onClick={() => openHelp("theme")}
                />
              </legend>
              <div className="theme-options" role="radiogroup" aria-label={i18n.themeLabel}>
                <label>
                  <input
                    type="radio"
                    name="theme"
                    value="light"
                    checked={theme === "light"}
                    onChange={(event) => setTheme(event.target.value)}
                  />
                  {i18n.lightTheme}
                </label>
                <label>
                  <input
                    type="radio"
                    name="theme"
                    value="dark"
                    checked={theme === "dark"}
                    onChange={(event) => setTheme(event.target.value)}
                  />
                  {i18n.darkTheme}
                </label>
                <label>
                  <input
                    type="radio"
                    name="theme"
                    value="custom"
                    checked={theme === "custom"}
                    onChange={(event) => setTheme(event.target.value)}
                  />
                  {i18n.customTheme}
                </label>
              </div>
              {theme === "custom" && (
                <div className="palette-controls">
                  <div className="color-field">
                    <label htmlFor="accentColor">{i18n.accentLabel}</label>
                    <input
                      type="color"
                      id="accentColor"
                      value={accentColor}
                      onChange={(event) => setAccentColor(event.target.value)}
                    />
                  </div>
                  <div className="color-field">
                    <label htmlFor="backgroundColor">{i18n.backgroundLabel}</label>
                    <input
                      type="color"
                      id="backgroundColor"
                      value={customBackground}
                      onChange={(event) => setCustomBackground(event.target.value)}
                    />
                  </div>
                  <div className="color-field">
                    <label htmlFor="surfaceColor">{i18n.surfaceLabel}</label>
                    <input
                      type="color"
                      id="surfaceColor"
                      value={customSurface}
                      onChange={(event) => setCustomSurface(event.target.value)}
                    />
                  </div>
                  <p className="field-helper">{i18n.paletteInstructions}</p>
                </div>
              )}
            </fieldset>
            <div className="form-actions">
              <button type="submit" className="primary" disabled={isGenerating}>
                {isGenerating ? i18n.generating : i18n.generate}
              </button>
              <div className="status-announcer" role="status" aria-live="polite">
                {statusMessage}
              </div>
            </div>
          </form>
        </section>
        <section className="visual-panel">
          <div className="canvas-wrapper" aria-live="polite">
            <canvas
              id="canvas"
              width="512"
              height="512"
              aria-label="Stable Diffusion output"
            ></canvas>
          </div>
          <div className="progress-panel">
            <div className="progress-header">
              <h2>{i18n.progressStatus}</h2>
            </div>
            <div
              className={`progress-bar${isGenerating ? " active" : ""}`}
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progressPercent)}
              aria-label={progressAriaLabel}
            >
              <div className="progress-bar-fill" style={{ width: `${progressPercent}%` }}></div>
            </div>
            <div className="progress-details" aria-live="polite">
              <p>{progressText}</p>
              {isGenerating && <div className="spinner" aria-hidden="true"></div>}
            </div>
          </div>
          <div className="telemetry-panel">
            <h2>{i18n.gpuStatus}</h2>
            <p aria-live="polite">{gpuStatus || i18n.generatorReady}</p>
            {schedulerFallback && (
              <div className="fallback-notice" role="status" aria-live="assertive">
                <strong>{i18n.fallbackLabel}:</strong> {i18n.schedulerFallbackMessage(
                  schedulerFallback.previous,
                  schedulerFallback.fallback,
                  schedulerFallback.model
                )}
              </div>
            )}
          </div>
          <section className="log-panel" aria-label={i18n.logsTitle}>
            <div className="log-header">
              <h2>{i18n.logsTitle}</h2>
              <button type="button" className="icon-button" onClick={() => setLogs([])}>
                {i18n.clearLogs}
              </button>
            </div>
            <div className="log-entries" role="log" aria-live="polite">
              {logs.length === 0 ? (
                <p>{i18n.logsEmpty}</p>
              ) : (
                logs.map((entry) => (
                  <div key={entry.id} className={`log-entry log-${entry.level}`}>
                    <span className="log-timestamp">
                      {timeFormatter.format(entry.timestamp)}
                    </span>
                    <span className="log-message">{entry.message}</span>
                  </div>
                ))
              )}
            </div>
          </section>
        </section>
      </main>
      <div className="sr-only" id="progress-tracker-label" aria-hidden="true"></div>
      <progress
        className="sr-only"
        id="progress-tracker-progress"
        max="100"
        value="0"
        aria-hidden="true"
      ></progress>
      <div className="sr-only" id="gpu-tracker-label" aria-hidden="true"></div>
      <div className="sr-only" id="log" aria-hidden="true"></div>
      <HelpModal
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
        content={helpContent}
        i18n={i18n}
      />
    </div>
  );
}

const rootElement = document.getElementById("root");
if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(<App />);
}
