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
    omniTitle: "Omni-modal preview stack",
    omniSubtitle: "Live previews for image, audio, and video outputs.",
    imagePanelTitle: "Image preview",
    audioPanelTitle: "Audio playback",
    audioPanelEmpty: "Audio previews will appear here when available.",
    audioSlicesTitle: "Volume slicing",
    audioSlicesEmpty: "Slice data will appear as audio streams are decoded.",
    audioSliceLabel: (index, start, end) =>
      `Slice ${index + 1} (${start.toFixed(2)}s – ${end.toFixed(2)}s)`,
    audioSliceGain: "Gain",
    audioItemLabel: (index) => `Audio clip ${index + 1}`,
    videoPanelTitle: "Video playback",
    videoPanelEmpty: "Video previews will appear here when available.",
    videoShortcutHint: "Focus a video tile and use Space to play or pause, ←/→ to seek, and M to mute.",
    videoItemLabel: (index) => `Video clip ${index + 1}`,
    toggleLegacyLabel: "Legacy diffusion controls",
    showLegacy: "Show legacy controls",
    hideLegacy: "Hide legacy controls",
    helpShortcutVideo: "Video playback shortcuts",
    helpVideoShortcuts: "Focus a video preview and press Space to play/pause, ←/→ to seek five seconds, or M to toggle mute.",
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
    omniTitle: "Vista previa omni-modal",
    omniSubtitle: "Previsualizaciones en vivo para imágenes, audio y video.",
    imagePanelTitle: "Vista previa de imagen",
    audioPanelTitle: "Reproducción de audio",
    audioPanelEmpty: "Las previsualizaciones de audio aparecerán aquí cuando estén disponibles.",
    audioSlicesTitle: "Segmentación de volumen",
    audioSlicesEmpty: "Los datos de segmentos aparecerán a medida que se decodifique el audio.",
    audioSliceLabel: (index, start, end) =>
      `Segmento ${index + 1} (${start.toFixed(2)}s – ${end.toFixed(2)}s)`,
    audioSliceGain: "Ganancia",
    audioItemLabel: (index) => `Clip de audio ${index + 1}`,
    videoPanelTitle: "Reproducción de video",
    videoPanelEmpty: "Las previsualizaciones de video aparecerán aquí cuando estén disponibles.",
    videoShortcutHint:
      "Enfoca una tarjeta de video y usa Espacio para reproducir/pausar, ←/→ para avanzar o retroceder y M para silenciar.",
    videoItemLabel: (index) => `Clip de video ${index + 1}`,
    toggleLegacyLabel: "Controles clásicos de difusión",
    showLegacy: "Mostrar controles clásicos",
    hideLegacy: "Ocultar controles clásicos",
    helpShortcutVideo: "Atajos de reproducción de video",
    helpVideoShortcuts:
      "Enfoca una previsualización de video y presiona Espacio para reproducir/pausar, ←/→ para mover cinco segundos o M para silenciar.",
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

function base64ToUint8Array(input) {
  if (typeof input !== "string" || typeof window === "undefined" || typeof window.atob !== "function") {
    return null;
  }
  try {
    const normalized = input.replace(/-/g, "+").replace(/_/g, "/");
    const binary = window.atob(normalized);
    const length = binary.length;
    const bytes = new Uint8Array(length);
    for (let index = 0; index < length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  } catch (error) {
    console.warn("Failed to decode base64 payload", error);
    return null;
  }
}

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
  const [showLegacyControls, setShowLegacyControls] = useState(false);
  const [audioPreviews, setAudioPreviews] = useState([]);
  const [videoPreviews, setVideoPreviews] = useState([]);
  const [activeVideoId, setActiveVideoId] = useState(null);

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
  const canvasRef = useRef(null);
  const audioRefs = useRef(new Map());
  const videoRefs = useRef(new Map());
  const audioObjectUrls = useRef(new Map());
  const videoObjectUrls = useRef(new Map());

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

  const registerAudioRef = useCallback(
    (id) => (element) => {
      if (!audioRefs.current) {
        return;
      }
      if (element) {
        audioRefs.current.set(id, element);
      } else {
        audioRefs.current.delete(id);
      }
    },
    []
  );

  const registerVideoRef = useCallback(
    (id) => (element) => {
      if (!videoRefs.current) {
        return;
      }
      if (element) {
        videoRefs.current.set(id, element);
      } else {
        videoRefs.current.delete(id);
      }
    },
    []
  );

  const drawImageToCanvas = useCallback((source) => {
    const canvas = canvasRef.current;
    if (!canvas || !source) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    const image = new Image();
    image.onload = () => {
      context.clearRect(0, 0, canvas.width, canvas.height);
      const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
      const width = image.width * scale;
      const height = image.height * scale;
      const offsetX = (canvas.width - width) / 2;
      const offsetY = (canvas.height - height) / 2;
      context.drawImage(image, offsetX, offsetY, width, height);
    };
    image.src = source;
  }, []);

  const handleSliceVolumeChange = useCallback((audioId, sliceIndex, value) => {
    const numeric = Number(value);
    const normalized = Number.isFinite(numeric) ? Math.max(0, Math.min(100, numeric)) / 100 : 1;
    setAudioPreviews((previous) =>
      previous.map((preview) => {
        if (preview.id !== audioId) {
          return preview;
        }
        const nextSlices = (preview.slices || []).map((slice, index) =>
          index === sliceIndex ? { ...slice, gain: normalized } : slice
        );
        return { ...preview, slices: nextSlices };
      })
    );
  }, []);

  useEffect(() => {
    const cleanups = [];
    audioRefs.current.forEach((element, id) => {
      const preview = audioPreviews.find((item) => item.id === id);
      if (!preview || !element) {
        return;
      }
      const slices = preview.slices || [];
      const updateVolume = () => {
        if (!element || typeof element.currentTime !== "number") {
          return;
        }
        const current = element.currentTime;
        let applied = typeof preview.gain === "number" ? preview.gain : 1;
        for (const slice of slices) {
          if (
            typeof slice.start === "number" &&
            typeof slice.end === "number" &&
            current >= slice.start &&
            current <= slice.end
          ) {
            if (typeof slice.gain === "number") {
              applied = slice.gain;
            } else if (typeof slice.volume === "number") {
              applied = slice.volume;
            }
            break;
          }
        }
        element.volume = Math.max(0, Math.min(1, applied));
      };
      element.addEventListener("timeupdate", updateVolume);
      updateVolume();
      cleanups.push(() => {
        element.removeEventListener("timeupdate", updateVolume);
      });
    });
    return () => {
      cleanups.forEach((dispose) => dispose());
    };
  }, [audioPreviews]);

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

  useEffect(() => {
    return () => {
      if (typeof URL !== "undefined" && typeof URL.revokeObjectURL === "function") {
        audioObjectUrls.current.forEach((url) => {
          URL.revokeObjectURL(url);
        });
        videoObjectUrls.current.forEach((url) => {
          URL.revokeObjectURL(url);
        });
      }
      audioObjectUrls.current.clear();
      videoObjectUrls.current.clear();
    };
  }, []);

  const handleStreamEvent = useCallback(
    (event) => {
      if (!event || typeof event !== "object") {
        return;
      }
      if (event.type === "progress") {
        setProgressState((previous) => ({
          ...previous,
          text: event.message || event.text || previous.text,
          progress:
            typeof event.progress === "number" ? Math.max(0, Math.min(1, event.progress)) : previous.progress,
          stage: event.stage || previous.stage,
        }));
        if (typeof event.progress === "number") {
          setIsGenerating(event.progress < 1);
        }
        if (event.message || event.text) {
          setStatusMessage(event.message || event.text);
        }
        return;
      }
      if (event.type === "status") {
        if (event.message) {
          setStatusMessage(event.message);
          addLogEntry(event.message, event.level || "info");
        }
        if (typeof event.progress === "number") {
          setProgressState((previous) => ({ ...previous, progress: event.progress }));
          setIsGenerating(event.progress < 1);
        }
        return;
      }
      if (event.type === "gpu") {
        if (event.message) {
          setGpuStatus(event.message);
        }
        return;
      }
      if (event.type === "log") {
        if (event.message) {
          addLogEntry(event.message, event.level || "info");
        }
        return;
      }
      if (event.type === "lifecycle") {
        if (event.stage === "start") {
          setIsGenerating(true);
          setStatusMessage(i18n.progressPreparing);
          setProgressState((previous) => ({ ...previous, text: i18n.progressPreparing, progress: 0 }));
        } else if (event.stage === "complete") {
          setStatusMessage(i18n.generatorReady);
        } else if (event.stage === "error") {
          setIsGenerating(false);
        } else if (event.stage === "end") {
          setIsGenerating(false);
        }
        return;
      }
      if (event.type === "validation-error" && event.field) {
        setValidationErrors((previous) => ({ ...previous, [event.field]: event.message }));
        return;
      }
      if (event.type === "validation-clear") {
        setValidationErrors({});
        return;
      }
      if (event.type === "image") {
        if (event.url) {
          drawImageToCanvas(event.url);
          return;
        }
        if (event.data) {
          if (typeof event.data === "string") {
            drawImageToCanvas(`data:${event.mimeType || "image/png"};base64,${event.data}`);
          } else if (event.data instanceof ArrayBuffer) {
            const blob = new Blob([event.data], { type: event.mimeType || "image/png" });
            const url = URL.createObjectURL(blob);
            drawImageToCanvas(url);
            if (typeof URL.revokeObjectURL === "function") {
              setTimeout(() => URL.revokeObjectURL(url), 1000);
            }
          }
        }
        return;
      }
      if (event.type === "audio" || event.type === "audio-preview") {
        const id = event.id || `${Date.now()}-${Math.random()}`;
        let url = event.url;
        let shouldRevoke = false;
        if (!url) {
          if (event.data instanceof ArrayBuffer) {
            const blob = new Blob([event.data], { type: event.mimeType || "audio/mpeg" });
            url = URL.createObjectURL(blob);
            shouldRevoke = true;
          } else if (typeof event.data === "string") {
            const bytes = base64ToUint8Array(event.data);
            if (bytes) {
              const blob = new Blob([bytes.buffer], { type: event.mimeType || "audio/mpeg" });
              url = URL.createObjectURL(blob);
              shouldRevoke = true;
            }
          }
        }
        if (url) {
          if (shouldRevoke) {
            const previous = audioObjectUrls.current.get(id);
            if (previous && typeof URL.revokeObjectURL === "function") {
              URL.revokeObjectURL(previous);
            }
            audioObjectUrls.current.set(id, url);
          }
          setAudioPreviews((previous) => {
            const next = previous.filter((item) => item.id !== id);
            const label = event.label || i18n.audioItemLabel(next.length);
            return [
              ...next,
              {
                id,
                url,
                label,
                mimeType: event.mimeType || "audio/mpeg",
                slices: Array.isArray(event.slices) ? event.slices : [],
              },
            ];
          });
        }
        return;
      }
      if (event.type === "audio-slices" || event.type === "audioSlices") {
        if (!event.id) {
          return;
        }
        setAudioPreviews((previous) =>
          previous.map((preview) =>
            preview.id === event.id
              ? { ...preview, slices: Array.isArray(event.slices) ? event.slices : preview.slices }
              : preview
          )
        );
        return;
      }
      if (event.type === "video" || event.type === "video-preview") {
        const id = event.id || `${Date.now()}-${Math.random()}`;
        let url = event.url;
        let shouldRevoke = false;
        if (!url) {
          if (event.data instanceof ArrayBuffer) {
            const blob = new Blob([event.data], { type: event.mimeType || "video/mp4" });
            url = URL.createObjectURL(blob);
            shouldRevoke = true;
          } else if (typeof event.data === "string") {
            const bytes = base64ToUint8Array(event.data);
            if (bytes) {
              const blob = new Blob([bytes.buffer], { type: event.mimeType || "video/mp4" });
              url = URL.createObjectURL(blob);
              shouldRevoke = true;
            }
          }
        }
        if (url) {
          if (shouldRevoke) {
            const previous = videoObjectUrls.current.get(id);
            if (previous && typeof URL.revokeObjectURL === "function") {
              URL.revokeObjectURL(previous);
            }
            videoObjectUrls.current.set(id, url);
          }
          setVideoPreviews((previous) => {
            const next = previous.filter((item) => item.id !== id);
            const label = event.label || i18n.videoItemLabel(next.length);
            return [
              ...next,
              {
                id,
                url,
                label,
                mimeType: event.mimeType || "video/mp4",
                poster: event.poster || null,
              },
            ];
          });
        }
        return;
      }
      if (event.type === "clear") {
        if (event.scope === "audio" || event.scope === "all") {
          audioObjectUrls.current.forEach((value) => {
            if (typeof URL.revokeObjectURL === "function") {
              URL.revokeObjectURL(value);
            }
          });
          audioObjectUrls.current.clear();
          setAudioPreviews([]);
        }
        if (event.scope === "video" || event.scope === "all") {
          videoObjectUrls.current.forEach((value) => {
            if (typeof URL.revokeObjectURL === "function") {
              URL.revokeObjectURL(value);
            }
          });
          videoObjectUrls.current.clear();
          setVideoPreviews([]);
        }
        return;
      }
    },
    [addLogEntry, drawImageToCanvas, i18n]
  );

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.WebSocket === "undefined") {
      return () => {};
    }
    let isActive = true;
    let socket;
    let buffer = "";

    const processBuffer = (payload) => {
      buffer += payload;
      const lines = buffer.split(/\n+/);
      buffer = lines.pop() || "";
      lines.forEach((line) => {
        const trimmed = line.trim();
        if (!trimmed) {
          return;
        }
        try {
          const parsed = JSON.parse(trimmed);
          handleStreamEvent(parsed);
        } catch (error) {
          console.warn("Failed to parse stream payload", error, trimmed);
        }
      });
    };

    const connect = () => {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const url = `${protocol}://${window.location.host}/generate`;
      socket = new WebSocket(url);
      socket.binaryType = "arraybuffer";
      socket.addEventListener("open", () => {
        addLogEntry("Connected to generation stream.");
      });
      socket.addEventListener("message", (event) => {
        if (!isActive) {
          return;
        }
        const data = event.data;
        if (typeof data === "string") {
          processBuffer(data);
        } else if (data instanceof ArrayBuffer) {
          const decoder = new TextDecoder();
          processBuffer(decoder.decode(new Uint8Array(data)));
        } else if (data instanceof Blob && typeof data.text === "function") {
          data
            .text()
            .then(processBuffer)
            .catch((error) => {
              console.warn("Failed to decode blob payload", error);
            });
        }
      });
      socket.addEventListener("error", (error) => {
        console.error("WebSocket error", error);
        addLogEntry("WebSocket error while reading generation stream.", "error");
      });
      socket.addEventListener("close", () => {
        if (!isActive) {
          return;
        }
        addLogEntry("Disconnected from generation stream.", "warn");
        setTimeout(() => {
          if (isActive) {
            connect();
          }
        }, 1500);
      });
    };

    connect();

    return () => {
      isActive = false;
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    };
  }, [addLogEntry, handleStreamEvent]);

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

  useEffect(() => {
    if (activeVideoId && !videoPreviews.some((preview) => preview.id === activeVideoId)) {
      setActiveVideoId(null);
    }
  }, [activeVideoId, videoPreviews]);

  useEffect(() => {
    function handleVideoKeys(event) {
      if (!activeVideoId) {
        return;
      }
      if (!event || !event.target) {
        return;
      }
      const targetTag = event.target.tagName;
      if (targetTag === "INPUT" || targetTag === "TEXTAREA" || targetTag === "SELECT") {
        return;
      }
      if (event.target.isContentEditable) {
        return;
      }
      const element = videoRefs.current.get(activeVideoId);
      if (!element) {
        return;
      }
      if (event.code === "Space") {
        event.preventDefault();
        if (element.paused) {
          element.play().catch(() => {});
        } else {
          element.pause();
        }
      } else if (event.code === "ArrowRight") {
        event.preventDefault();
        element.currentTime = Math.min(element.duration || element.currentTime + 5, element.currentTime + 5);
      } else if (event.code === "ArrowLeft") {
        event.preventDefault();
        element.currentTime = Math.max(0, element.currentTime - 5);
      } else if (event.code === "KeyM") {
        event.preventDefault();
        element.muted = !element.muted;
      }
    }
    window.addEventListener("keydown", handleVideoKeys);
    return () => window.removeEventListener("keydown", handleVideoKeys);
  }, [activeVideoId]);

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
                  <li>{i18n.helpVideoShortcuts}</li>
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
        <section className="visual-panel omni-stack">
          <div className="visual-panel-header">
            <div>
              <h2>{i18n.omniTitle}</h2>
              <p>{i18n.omniSubtitle}</p>
            </div>
            <div className="omni-actions">
              <span className="legacy-label">{i18n.toggleLegacyLabel}</span>
              <button
                type="button"
                className="primary ghost"
                onClick={() => setShowLegacyControls((current) => !current)}
                aria-pressed={showLegacyControls}
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
                {showLegacyControls ? i18n.hideLegacy : i18n.showLegacy}
              </button>
            </div>
          </div>
          <section className="image-panel" aria-label={i18n.imagePanelTitle}>
            <h3>{i18n.imagePanelTitle}</h3>
            <div className="canvas-wrapper" aria-live="polite">
              <canvas
                id="canvas"
                width="512"
                height="512"
                aria-label="Stable Diffusion output"
                ref={canvasRef}
              ></canvas>
            </div>
          </section>
          <section className="audio-panel" aria-label={i18n.audioPanelTitle}>
            <h3>{i18n.audioPanelTitle}</h3>
            {audioPreviews.length === 0 ? (
              <p>{i18n.audioPanelEmpty}</p>
            ) : (
              audioPreviews.map((preview, index) => (
                <div key={preview.id} className="audio-preview">
                  <h4>{preview.label || i18n.audioItemLabel(index)}</h4>
                  <audio
                    controls
                    src={preview.url}
                    ref={registerAudioRef(preview.id)}
                    data-preview-id={preview.id}
                  >
                    Your browser does not support the audio element.
                  </audio>
                </div>
              ))
            )}
          </section>
          <section className="volume-panel" aria-label={i18n.audioSlicesTitle}>
            <h3>{i18n.audioSlicesTitle}</h3>
            {audioPreviews.length === 0 ? (
              <p>{i18n.audioSlicesEmpty}</p>
            ) : (
              audioPreviews.map((preview) => (
                <div key={`${preview.id}-slices`} className="volume-preview">
                  <h4>{preview.label}</h4>
                  {preview.slices && preview.slices.length > 0 ? (
                    preview.slices.map((slice, index) => {
                      const start = typeof slice.start === "number" ? slice.start : 0;
                      const end = typeof slice.end === "number" ? slice.end : start;
                      const gainValue =
                        typeof slice.gain === "number"
                          ? slice.gain
                          : typeof slice.volume === "number"
                          ? slice.volume
                          : 1;
                      return (
                        <label key={`${preview.id}-${index}`} className="slice-control">
                          <span>{i18n.audioSliceLabel(index, start, end)}</span>
                          <input
                            type="range"
                            min="0"
                            max="100"
                            step="1"
                            value={Math.round(gainValue * 100)}
                            onChange={(event) =>
                              handleSliceVolumeChange(preview.id, index, event.target.value)
                            }
                            aria-label={`${preview.label} ${i18n.audioSliceGain} ${index + 1}`}
                          />
                        </label>
                      );
                    })
                  ) : (
                    <p>{i18n.audioSlicesEmpty}</p>
                  )}
                </div>
              ))
            )}
          </section>
          <section className="video-panel" aria-label={i18n.videoPanelTitle}>
            <h3>{i18n.videoPanelTitle}</h3>
            <p className="panel-helper">{i18n.videoShortcutHint}</p>
            {videoPreviews.length === 0 ? (
              <p>{i18n.videoPanelEmpty}</p>
            ) : (
              videoPreviews.map((preview, index) => (
                <div
                  key={preview.id}
                  className={`video-preview${activeVideoId === preview.id ? " active" : ""}`}
                  tabIndex={0}
                  onFocus={() => setActiveVideoId(preview.id)}
                  onMouseEnter={() => setActiveVideoId(preview.id)}
                  onBlur={(event) => {
                    if (!event.currentTarget.contains(event.relatedTarget)) {
                      setActiveVideoId((current) => (current === preview.id ? null : current));
                    }
                  }}
                >
                  <h4>{preview.label || i18n.videoItemLabel(index)}</h4>
                  <video
                    controls
                    src={preview.url}
                    poster={preview.poster || undefined}
                    ref={registerVideoRef(preview.id)}
                    onPlay={() => setActiveVideoId(preview.id)}
                    onClick={() => setActiveVideoId(preview.id)}
                  >
                    Your browser does not support the video element.
                  </video>
                </div>
              ))
            )}
          </section>
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
                    <span className="log-timestamp">{timeFormatter.format(entry.timestamp)}</span>
                    <span className="log-message">{entry.message}</span>
                  </div>
                ))
              )}
            </div>
          </section>
        </section>
        {showLegacyControls && (
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
        )}
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
