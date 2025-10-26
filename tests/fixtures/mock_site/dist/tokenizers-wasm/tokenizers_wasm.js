export default async function init() {
  window.__tokenizerInitCount = (window.__tokenizerInitCount || 0) + 1;
}

export class TokenizerWasm {
  constructor(jsonText) {
    this.jsonText = jsonText;
    window.__lastTokenizerPayload = jsonText;
  }
}
