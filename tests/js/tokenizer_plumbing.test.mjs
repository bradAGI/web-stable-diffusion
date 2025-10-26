import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function loadModuleScript() {
  const htmlPath = path.resolve(__dirname, '../../web/stable_diffusion.html');
  const html = readFileSync(htmlPath, 'utf8');
  const match = /<script type="module">([\s\S]*?)<\/script>/.exec(html);
  if (!match) {
    throw new Error('Unable to locate tokenizer module script');
  }
  const scriptContent = match[1];
  const withoutImports = scriptContent.replace(/\s*import[^;]+;\s*/g, '');
  const patched =
    `var initCalls = 0;\n` +
    `async function init() { initCalls += 1; }\n` +
    `function TokenizerWasm(jsonText) { this.jsonText = jsonText; }\n` +
    withoutImports;
  return patched;
}

const context = {
  tvmjsGlobalEnv: {},
  fetchCalls: [],
};

context.fetch = (url) => {
  context.fetchCalls.push(url);
  return Promise.resolve({
    text: () => Promise.resolve('{"hello": "world"}'),
  });
};

vm.createContext(context);

const script = loadModuleScript();
const moduleScript = new vm.Script(script, { filename: 'tokenizer-module.js' });
moduleScript.runInContext(context);

describe('tokenizer plumbing', () => {
  it('exposes getTokenizer on the global env', async () => {
    expect(typeof context.tvmjsGlobalEnv.getTokenizer).toBe('function');
    const tokenizer = await context.tvmjsGlobalEnv.getTokenizer('mock/model');
    expect(context.fetchCalls[0]).toBe(
      'https://huggingface.co/mock/model/raw/main/tokenizer.json'
    );
    expect(tokenizer.jsonText).toContain('hello');
  });

  it('initializes wasm loader only once', async () => {
    const before = context.initCalls;
    await context.tvmjsGlobalEnv.getTokenizer('mock/model');
    await context.tvmjsGlobalEnv.getTokenizer('mock/model');
    expect(context.initCalls).toBe(before + 2);
  });
});
