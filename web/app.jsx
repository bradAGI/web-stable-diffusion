const { useState, useEffect, useMemo, useCallback, useRef } = React;

function hexToHsl(hex) {
  if (!hex || typeof hex !== "string") {
    return { h: 240, s: 85, l: 60 };
  }
  let normalized = hex.trim().replace(/^#/, "");
  if (normalized.length === 3) {
    normalized = normalized
      .split("")
      .map((char) => char + char)
      .join("");
  }
  if (normalized.length !== 6) {
    return { h: 240, s: 85, l: 60 };
  }
  const r = parseInt(normalized.slice(0, 2), 16) / 255;
  const g = parseInt(normalized.slice(2, 4), 16) / 255;
  const b = parseInt(normalized.slice(4, 6), 16) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let h = 0;
  if (delta !== 0) {
    if (max === r) {
      h = ((g - b) / delta) % 6;
    } else if (max === g) {
      h = (b - r) / delta + 2;
    } else {
      h = (r - g) / delta + 4;
    }
    h *= 60;
    if (h < 0) {
      h += 360;
    }
  }
  const l = (max + min) / 2;
  const s = delta === 0 ? 0 : delta / (1 - Math.abs(2 * l - 1));
  return {
    h: Math.round(h),
    s: Math.round(s * 100),
    l: Math.round(l * 100),
  };
}

const CONTROL_LAYOUT_SEQUENCE = ["classic", "compact", "adaptive"];

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
    helpShortcutLayout: "Cycle control layout",
    helpShortcutLogs: "Pause or resume logs",
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
    logsFilterLabel: "Filter",
    logsLevelAll: "All",
    logsLevelInfo: "Info",
    logsLevelWarn: "Warnings",
    logsLevelError: "Errors",
    logsSearchPlaceholder: "Search messages…",
    logsPause: "Pause",
    logsResume: "Resume",
    logsPaused: "Log stream paused.",
    logsResumed: "Log stream resumed.",
    logsFilteredEmpty: "No log entries match the current filters.",
    customizingPalette: "Customize palette",
    layoutLabel: "Control layout",
    layoutDescription: "Choose how form controls are arranged.",
    layoutAssistiveHint: "Layouts can also be cycled with Alt + L.",
    layoutClassic: "Classic",
    layoutCompact: "Compact",
    layoutAdaptive: "Adaptive",
    timelineTitle: "Generation timeline",
    timelineEmpty: "Timeline updates will appear when a generation starts.",
    timelineStarted: "Generation started",
    timelineCompleted: "Generation complete",
    timelineError: "Generation failed",
    timelineValidation: "Waiting for corrections",
    timelineElapsed: (time) => `${time} elapsed`,
    timelineStageLabel: "Stage",
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
    helpShortcutLayout: "Cambiar distribución",
    helpShortcutLogs: "Pausar o reanudar registro",
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
    logsFilterLabel: "Filtrar",
    logsLevelAll: "Todo",
    logsLevelInfo: "Info",
    logsLevelWarn: "Avisos",
    logsLevelError: "Errores",
    logsSearchPlaceholder: "Buscar mensajes…",
    logsPause: "Pausar",
    logsResume: "Reanudar",
    logsPaused: "Transmisión de registro en pausa.",
    logsResumed: "Transmisión de registro reanudada.",
    logsFilteredEmpty: "No hay entradas que coincidan con los filtros.",
    customizingPalette: "Personalizar paleta",
    layoutLabel: "Distribución de controles",
    layoutDescription: "Elige cómo se organizan los controles.",
    layoutAssistiveHint: "También puedes alternar con Alt + L.",
    layoutClassic: "Clásica",
    layoutCompact: "Compacta",
    layoutAdaptive: "Adaptable",
    timelineTitle: "Cronología de generación",
    timelineEmpty: "Los eventos aparecerán cuando inicie una generación.",
    timelineStarted: "Generación iniciada",
    timelineCompleted: "Generación completada",
    timelineError: "Generación fallida",
    timelineValidation: "Esperando correcciones",
    timelineElapsed: (time) => `${time} transcurridos`,
    timelineStageLabel: "Etapa",
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

function StatusTimeline({ entries, i18n }) {
  if (!entries || entries.length === 0) {
    return <p className="timeline-empty">{i18n.timelineEmpty}</p>;
  }

  return (
    <ol className="status-timeline" aria-live="polite">
      {entries.map((entry, index) => {
        const isLast = index === entries.length - 1;
        return (
          <li key={entry.id} className={`status-timeline-item${isLast ? " active" : ""}`}>
            <div className="status-timeline-marker" aria-hidden="true">
              <span className="status-timeline-dot"></span>
            </div>
            <div className="status-timeline-content">
              <div className="status-timeline-header">
                <span className="status-timeline-stage">{entry.label}</span>
                <time dateTime={new Date(entry.timestamp).toISOString()}>{entry.formattedTime}</time>
              </div>
              <div className="status-timeline-body">
                <p>{entry.message}</p>
                <div className="status-timeline-meta">
                  {entry.progress !== null && (
                    <span className="status-timeline-progress" aria-label={`${i18n.progressStatus}: ${Math.round(entry.progress * 100)}%`}>
                      {Math.round(entry.progress * 100)}%
                    </span>
                  )}
                  <span className="status-timeline-elapsed">{entry.elapsedLabel}</span>
                </div>
              </div>
            </div>
          </li>
        );
      })}
    </ol>
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
  const [controlLayout, setControlLayout] = useState(() => {
    try {
      return window.localStorage.getItem("wsd-control-layout") || "classic";
    } catch (err) {
      return "classic";
    }
  });
  const [timelineEntries, setTimelineEntries] = useState([]);
  const [isLogPaused, setIsLogPaused] = useState(false);
  const [logLevelFilter, setLogLevelFilter] = useState("all");
  const [logQuery, setLogQuery] = useState("");

  const promptRef = useRef(prompt);
  const negativePromptRef = useRef(negativePrompt);
  const schedulerRef = useRef(scheduler);
  const vaeRef = useRef(vaeCycle);
  const logContainerRef = useRef(null);
  const lastTimelineEventRef = useRef(null);
  const generationStartRef = useRef(null);

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

  const formatDuration = useCallback((durationMs) => {
    if (!durationMs || Number.isNaN(durationMs)) {
      return "0s";
    }
    const totalSeconds = Math.max(0, Math.round(durationMs / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    if (minutes > 0) {
      return `${minutes}:${seconds.toString().padStart(2, "0")} min`;
    }
    return `${seconds}s`;
  }, []);

  const cycleLayout = useCallback(() => {
    setControlLayout((current) => {
      const index = CONTROL_LAYOUT_SEQUENCE.indexOf(current);
      const nextIndex = index === -1 ? 0 : (index + 1) % CONTROL_LAYOUT_SEQUENCE.length;
      return CONTROL_LAYOUT_SEQUENCE[nextIndex];
    });
  }, []);

  const timelineStageLabels = useMemo(
    () => ({
      start: i18n.timelineStarted,
      complete: i18n.timelineCompleted,
      error: i18n.timelineError,
      "validation-error": i18n.timelineValidation,
    }),
    [i18n]
  );

  const recordTimelineEvent = useCallback(
    ({ stage, text, progress, timestamp: providedTimestamp }) => {
      const timestamp = providedTimestamp || Date.now();
      const normalizedStage = stage || "progress";
      const defaultLabel = timelineStageLabels[normalizedStage] || text || i18n.timelineStageLabel;
      const message = text || defaultLabel;
      const roundedProgress =
        typeof progress === "number" && !Number.isNaN(progress)
          ? Math.max(0, Math.min(1, progress))
          : null;
      const last = lastTimelineEventRef.current;
      if (
        last &&
        last.stage === normalizedStage &&
        last.text === message &&
        (roundedProgress === null || last.progress === roundedProgress)
      ) {
        return;
      }
      const elapsed = generationStartRef.current ? timestamp - generationStartRef.current : 0;
      const entry = {
        id: `${timestamp}-${Math.random()}`,
        stage: normalizedStage,
        label: defaultLabel,
        message,
        progress: roundedProgress,
        timestamp,
        formattedTime: timeFormatter.format(timestamp),
        elapsedLabel: i18n.timelineElapsed(formatDuration(elapsed)),
      };
      setTimelineEntries((previous) => [...previous.slice(-19), entry]);
      lastTimelineEventRef.current = {
        stage: normalizedStage,
        text: message,
        progress: roundedProgress,
      };
    },
    [formatDuration, i18n, timeFormatter, timelineStageLabels]
  );

  useEffect(() => {
    setTimelineEntries((previous) => {
      let didChange = false;
      const updated = previous.map((entry) => {
        const elapsed = generationStartRef.current ? entry.timestamp - generationStartRef.current : 0;
        const nextLabel = timelineStageLabels[entry.stage] || entry.message;
        const nextElapsed = i18n.timelineElapsed(formatDuration(elapsed));
        if (entry.label === nextLabel && entry.elapsedLabel === nextElapsed) {
          return entry;
        }
        didChange = true;
        return {
          ...entry,
          label: nextLabel,
          elapsedLabel: nextElapsed,
        };
      });
      return didChange ? updated : previous;
    });
  }, [formatDuration, i18n, timelineStageLabels]);

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
    const { h, s, l } = hexToHsl(accentColor);
    document.documentElement.style.setProperty("--accent-hue", `${h}deg`);
    document.documentElement.style.setProperty("--accent-saturation", `${s}%`);
    document.documentElement.style.setProperty("--accent-lightness", `${l}%`);
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
    document.documentElement.dataset.layout = controlLayout;
    try {
      window.localStorage.setItem("wsd-control-layout", controlLayout);
    } catch (err) {
      // ignore storage errors
    }
  }, [controlLayout]);

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

  const toggleLogPause = useCallback(() => {
    setIsLogPaused((previous) => {
      const next = !previous;
      addLogEntry(next ? i18n.logsPaused : i18n.logsResumed, "info");
      return next;
    });
  }, [addLogEntry, i18n]);

  const filteredLogs = useMemo(() => {
    const normalizedQuery = logQuery.trim().toLowerCase();
    return logs.filter((entry) => {
      const matchesLevel = logLevelFilter === "all" || entry.level === logLevelFilter;
      if (!matchesLevel) {
        return false;
      }
      if (!normalizedQuery) {
        return true;
      }
      return entry.message.toLowerCase().includes(normalizedQuery);
    });
  }, [logs, logLevelFilter, logQuery]);

  useEffect(() => {
    if (isLogPaused) {
      return;
    }
    const container = logContainerRef.current;
    if (!container) {
      return;
    }
    container.scrollTop = container.scrollHeight;
  }, [filteredLogs, isLogPaused]);

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
      if (!report) {
        setProgressState({ text: "", progress: 0, stage: "" });
        return true;
      }
      setProgressState((previous) => ({
        text: report.text ?? previous.text,
        progress:
          typeof report.progress === "number"
            ? report.progress
            : previous.progress,
        stage: report.stage ?? previous.stage,
      }));
      if (typeof report.progress === "number") {
        setIsGenerating(report.progress < 1);
      }
      if (report.text) {
        setStatusMessage(report.text);
      }
      recordTimelineEvent({
        stage: report.stage,
        text: report.text,
        progress: report.progress,
      });
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
        generationStartRef.current = Date.now();
        lastTimelineEventRef.current = null;
        setTimelineEntries([]);
        recordTimelineEvent({
          stage: "start",
          text: i18n.timelineStarted,
          progress: 0,
          timestamp: generationStartRef.current,
        });
        setIsGenerating(true);
        setStatusMessage(i18n.progressPreparing);
        setProgressState((previous) => ({
          ...previous,
          text: i18n.progressPreparing,
          progress: 0,
          stage: "start",
        }));
      } else if (stage === "complete") {
        recordTimelineEvent({ stage: "complete", text: i18n.timelineCompleted, progress: 1 });
        setStatusMessage(i18n.generatorReady);
        setIsGenerating(false);
        setProgressState((previous) => ({ ...previous, stage: "complete", progress: 1 }));
      } else if (stage === "error") {
        recordTimelineEvent({ stage: "error", text: i18n.timelineError });
        setIsGenerating(false);
        setStatusMessage(i18n.timelineError);
        setProgressState((previous) => ({ ...previous, stage: "error" }));
      } else if (stage === "validation-error") {
        recordTimelineEvent({ stage: "validation-error", text: i18n.timelineValidation });
        setIsGenerating(false);
        setStatusMessage(i18n.validationSummary);
        setProgressState((previous) => ({ ...previous, stage: "validation-error" }));
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
  }, [addLogEntry, getSchedulerLabel, i18n, model, recordTimelineEvent]);

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
      } else if (event.altKey && event.key.toLowerCase() === "l") {
        event.preventDefault();
        cycleLayout();
      } else if (event.altKey && (event.key === "`" || event.code === "Backquote")) {
        event.preventDefault();
        toggleLogPause();
      } else if (event.key === "Escape" && isHelpOpen) {
        setIsHelpOpen(false);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleGenerate, cycleTheme, cycleLayout, toggleLogPause, isHelpOpen, helpContext]);

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
      case "layout":
        return {
          title: i18n.layoutLabel,
          items: [
            { title: i18n.layoutLabel, content: <p>{i18n.layoutDescription}</p> },
            { title: i18n.help, content: <p>{i18n.layoutAssistiveHint}</p> },
          ],
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
                  <li>
                    <kbd>Alt</kbd> + <kbd>L</kbd> — {i18n.helpShortcutLayout}
                  </li>
                  <li>
                    <kbd>Alt</kbd> + <kbd>&#96;</kbd> — {i18n.helpShortcutLogs}
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

  const openHelp = useCallback((type) => {
    setHelpContext({ type });
    setIsHelpOpen(true);
  }, []);

  return (
    <div className={`app layout-${controlLayout}`} data-generating={isGenerating} data-layout={controlLayout}>
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
            <fieldset className="field-group">
              <legend>
                {i18n.layoutLabel}
                <ContextualHelpButton
                  label={`${i18n.openHelp}: ${i18n.layoutLabel}`}
                  onClick={() => openHelp("layout")}
                />
              </legend>
              <p className="field-helper">{i18n.layoutDescription}</p>
              <div className="layout-options" role="radiogroup" aria-label={i18n.layoutLabel}>
                <label>
                  <input
                    type="radio"
                    name="controlLayout"
                    value="classic"
                    checked={controlLayout === "classic"}
                    onChange={(event) => setControlLayout(event.target.value)}
                  />
                  {i18n.layoutClassic}
                </label>
                <label>
                  <input
                    type="radio"
                    name="controlLayout"
                    value="compact"
                    checked={controlLayout === "compact"}
                    onChange={(event) => setControlLayout(event.target.value)}
                  />
                  {i18n.layoutCompact}
                </label>
                <label>
                  <input
                    type="radio"
                    name="controlLayout"
                    value="adaptive"
                    checked={controlLayout === "adaptive"}
                    onChange={(event) => setControlLayout(event.target.value)}
                  />
                  {i18n.layoutAdaptive}
                </label>
              </div>
              <p className="field-helper hint">{i18n.layoutAssistiveHint}</p>
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
          <section className="timeline-panel" aria-label={i18n.timelineTitle}>
            <div className="timeline-header">
              <h2>{i18n.timelineTitle}</h2>
            </div>
            <StatusTimeline entries={timelineEntries} i18n={i18n} />
          </section>
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
          <section className="log-panel" aria-label={i18n.logsTitle} data-paused={isLogPaused}>
            <div className="log-header">
              <h2>{i18n.logsTitle}</h2>
              <div className="log-header-actions">
                <button type="button" className="icon-button" onClick={() => setLogs([])}>
                  {i18n.clearLogs}
                </button>
                <button
                  type="button"
                  className={`icon-button${isLogPaused ? " active" : ""}`}
                  onClick={toggleLogPause}
                  aria-pressed={isLogPaused}
                >
                  {isLogPaused ? i18n.logsResume : i18n.logsPause}
                </button>
              </div>
            </div>
            <div className="log-controls">
              <label className="log-filter-label" htmlFor="log-level">
                {i18n.logsFilterLabel}
              </label>
              <select
                id="log-level"
                value={logLevelFilter}
                onChange={(event) => setLogLevelFilter(event.target.value)}
              >
                <option value="all">{i18n.logsLevelAll}</option>
                <option value="info">{i18n.logsLevelInfo}</option>
                <option value="warn">{i18n.logsLevelWarn}</option>
                <option value="error">{i18n.logsLevelError}</option>
              </select>
              <label className="sr-only" htmlFor="log-search">
                {i18n.logsSearchPlaceholder}
              </label>
              <input
                id="log-search"
                type="search"
                value={logQuery}
                onChange={(event) => setLogQuery(event.target.value)}
                placeholder={i18n.logsSearchPlaceholder}
              />
            </div>
            <div
              className="log-entries"
              ref={logContainerRef}
              role="log"
              aria-live={isLogPaused ? "off" : "polite"}
            >
              {filteredLogs.length === 0 ? (
                <p>{logs.length === 0 ? i18n.logsEmpty : i18n.logsFilteredEmpty}</p>
              ) : (
                filteredLogs.map((entry) => (
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
