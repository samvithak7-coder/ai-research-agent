import os
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Config
# Primary model first, then automatic fallbacks if it's overloaded (503/429).
# Override the primary with a GEMINI_MODEL env var on Vercel if you like.
# (gemini-2.5-* models shut down Oct 16, 2026 - don't use them.)
# ---------------------------------------------------------------------------
PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
MODELS = [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m != PRIMARY_MODEL]

ATTEMPTS_PER_MODEL = 2
RETRY_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_MS = 40000
TOTAL_BUDGET_SECONDS = 50
RETRYABLE_CODES = {429, 500, 502, 503, 504}

# Shorter reports = faster. Users pick this in the UI.
LENGTHS = {
    "quick": "about 400 words",
    "standard": "about 800 words",
    "deep": "about 1500 words",
}

HTML_UI = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autonomous AI Research Agent</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.6/purify.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
<style>
  :root {
    --bg: #070b16; --card: rgba(30,41,59,.65); --border: rgba(148,163,184,.18);
    --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8; --accent2: #818cf8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; color: var(--text); padding: 48px 16px 80px;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background:
      radial-gradient(900px 500px at 15% -10%, rgba(56,189,248,.18), transparent 60%),
      radial-gradient(800px 500px at 90% 0%, rgba(129,140,248,.18), transparent 60%),
      var(--bg);
  }
  .wrap { max-width: 820px; margin: auto; }
  .badge {
    display: inline-flex; align-items: center; gap: 8px; font-size: 12px; color: var(--accent);
    background: rgba(56,189,248,.1); border: 1px solid rgba(56,189,248,.25);
    padding: 5px 12px; border-radius: 999px; margin-bottom: 18px;
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 8px #22c55e; }
  h1 {
    font-size: clamp(28px, 5vw, 40px); margin: 0 0 10px; line-height: 1.15; letter-spacing: -.02em;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .sub { color: var(--muted); font-size: 15px; margin: 0 0 28px; }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 22px;
    backdrop-filter: blur(12px); box-shadow: 0 20px 50px rgba(0,0,0,.35);
  }
  .row { display: flex; gap: 10px; flex-wrap: wrap; }
  input[type=text] {
    flex: 1 1 280px; background: #0b1224; color: #fff; border: 1px solid #334155; border-radius: 10px;
    padding: 14px 16px; font-size: 15px; outline: none; transition: border .2s, box-shadow .2s;
  }
  input[type=text]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(56,189,248,.15); }
  select {
    background: #0b1224; color: #fff; border: 1px solid #334155; border-radius: 10px;
    padding: 0 12px; font-size: 14px; height: 48px; outline: none;
  }
  .btn {
    border: none; cursor: pointer; font-weight: 600; font-size: 15px; border-radius: 10px; padding: 0 22px; height: 48px;
    color: #fff; background: linear-gradient(135deg, #0284c7, #6366f1); transition: transform .15s, opacity .2s, box-shadow .2s;
    box-shadow: 0 6px 20px rgba(99,102,241,.35);
  }
  .btn:hover { transform: translateY(-1px); }
  .btn:disabled { opacity: .55; cursor: not-allowed; transform: none; }
  .btn.ghost {
    background: rgba(148,163,184,.12); border: 1px solid var(--border); box-shadow: none; height: 38px;
    font-size: 13px; padding: 0 14px; color: var(--text);
  }
  .btn.ghost:hover { background: rgba(148,163,184,.22); }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
  .chip {
    font-size: 12px; color: var(--muted); background: rgba(148,163,184,.08); border: 1px solid var(--border);
    padding: 6px 12px; border-radius: 999px; cursor: pointer; transition: all .2s;
  }
  .chip:hover { color: #fff; border-color: var(--accent); }

  #status { display: none; align-items: center; gap: 12px; margin-top: 18px; color: #fbbf24; font-size: 14px; }
  .spinner {
    width: 18px; height: 18px; border-radius: 50%; border: 2px solid rgba(251,191,36,.25);
    border-top-color: #fbbf24; animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #result { display: none; margin-top: 22px; }
  .toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
  .toolbar .grow { flex: 1; font-size: 12px; color: #64748b; }
  #report {
    background: #0b1224; border: 1px solid var(--border); border-radius: 14px; padding: 8px 30px 26px;
    line-height: 1.75; font-size: 15px; overflow-wrap: anywhere;
  }
  #report h1, #report h2, #report h3 { color: #f1f5f9; -webkit-text-fill-color: initial; background: none; }
  #report h1 { font-size: 26px; margin: 24px 0 8px; }
  #report h2 { font-size: 19px; margin: 28px 0 8px; color: var(--accent); padding-bottom: 6px; border-bottom: 1px solid var(--border); }
  #report h3 { font-size: 16px; margin: 20px 0 6px; }
  #report ul, #report ol { padding-left: 22px; }
  #report li { margin: 5px 0; }
  #report strong { color: #fff; }
  #report code { background: rgba(148,163,184,.15); padding: 2px 6px; border-radius: 5px; font-size: 13px; }
  #report table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 14px; }
  #report th, #report td { border: 1px solid #334155; padding: 8px 12px; text-align: left; }
  #report th { background: rgba(56,189,248,.1); }
  .cursor::after { content: "▍"; color: var(--accent); animation: blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  .error { border-color: #b91c1c !important; color: #fca5a5; white-space: pre-wrap; padding: 22px !important; }
  footer { text-align: center; color: #475569; font-size: 12px; margin-top: 30px; }

  /* Light, print-friendly style used only for the PDF */
  .pdf-doc { color: #111827; background: #fff; font-family: Georgia, "Times New Roman", serif; font-size: 12.5px; line-height: 1.65; padding: 6px 4px; }
  .pdf-doc .pdf-head { border-bottom: 3px solid #0284c7; padding-bottom: 10px; margin-bottom: 16px; }
  .pdf-doc .pdf-head h1 { font-size: 24px; margin: 0 0 4px; color: #0c4a6e; }
  .pdf-doc .pdf-head div { font-size: 11px; color: #6b7280; font-family: Arial, sans-serif; }
  .pdf-doc h1 { font-size: 20px; color: #0c4a6e; margin: 18px 0 6px; }
  .pdf-doc h2 { font-size: 16px; color: #0369a1; margin: 18px 0 6px; border-bottom: 1px solid #e5e7eb; padding-bottom: 3px; }
  .pdf-doc h3 { font-size: 13.5px; color: #111827; margin: 12px 0 4px; }
  .pdf-doc p { margin: 6px 0; }
  .pdf-doc ul, .pdf-doc ol { padding-left: 20px; margin: 6px 0; }
  .pdf-doc table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 11.5px; }
  .pdf-doc th, .pdf-doc td { border: 1px solid #d1d5db; padding: 5px 8px; text-align: left; }
  .pdf-doc th { background: #e0f2fe; }
  .pdf-doc code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="badge"><span class="dot"></span> Powered by Gemini</div>
  <h1>Autonomous AI Research Agent</h1>
  <p class="sub">Enter any technical topic and get a structured intelligence report, streamed live. Download it as PDF when you're done.</p>

  <div class="card">
    <div class="row">
      <input type="text" id="topic" placeholder="e.g., Edge AI & Vision-Language Models" autocomplete="off">
      <select id="depth" title="Report length">
        <option value="quick">Quick (~400 words)</option>
        <option value="standard" selected>Standard (~800 words)</option>
        <option value="deep">Deep (~1500 words)</option>
      </select>
      <button class="btn" id="runBtn">Generate</button>
    </div>
    <div class="chips" id="chips">
      <span class="chip">Edge AI</span>
      <span class="chip">Retrieval-Augmented Generation</span>
      <span class="chip">Quantum Computing</span>
      <span class="chip">Vision-Language Models</span>
      <span class="chip">Zero-Trust Security</span>
    </div>
    <div id="status"><div class="spinner"></div><span id="statusText"></span></div>
  </div>

  <div id="result">
    <div class="toolbar">
      <span class="grow" id="meta"></span>
      <button class="btn ghost" id="copyBtn">Copy</button>
      <button class="btn ghost" id="mdBtn">Download .md</button>
      <button class="btn ghost" id="pdfBtn">Download PDF</button>
    </div>
    <div id="report"></div>
  </div>

  <footer>Autonomous AI Research Agent</footer>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  let rawMd = "", currentTopic = "", timer = null, renderQueued = false;

  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const slug = (s) => (s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "report").slice(0, 50);

  function renderMd(streaming) {
    $("report").classList.remove("error");
    $("report").innerHTML = DOMPurify.sanitize(marked.parse(rawMd));
    $("report").classList.toggle("cursor", !!streaming);
  }
  function queueRender() {
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(() => { renderQueued = false; renderMd(true); });
  }
  function setBusy(busy, text) {
    $("runBtn").disabled = busy;
    $("status").style.display = busy ? "flex" : "none";
    if (text) $("statusText").textContent = text;
  }
  function showError(msg) {
    $("result").style.display = "block";
    $("report").className = "error";
    $("report").textContent = msg;
    $("meta").textContent = "";
    ["copyBtn", "mdBtn", "pdfBtn"].forEach((b) => ($(b).style.display = "none"));
  }

  async function generate() {
    const topic = $("topic").value.trim();
    if (!topic) { $("topic").focus(); return; }
    currentTopic = topic; rawMd = "";
    ["copyBtn", "mdBtn", "pdfBtn"].forEach((b) => ($(b).style.display = ""));
    $("result").style.display = "none";
    setBusy(true, "Connecting to Gemini...");
    const t0 = performance.now();
    clearInterval(timer);
    timer = setInterval(() => {
      const s = ((performance.now() - t0) / 1000).toFixed(1);
      $("statusText").textContent = (rawMd ? "Writing report... " : "Connecting to Gemini... ") + s + "s";
    }, 100);

    try {
      const res = await fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, depth: $("depth").value }),
      });

      if (!res.ok) {
        let msg = "Something went wrong.";
        try { msg = (await res.json()).error || msg; } catch (e) {}
        showError(msg);
        return;
      }

      const model = res.headers.get("X-Model") || "";
      const attempts = parseInt(res.headers.get("X-Attempts") || "1", 10);
      $("result").style.display = "block";
      $("pdfBtn").disabled = $("copyBtn").disabled = $("mdBtn").disabled = true;

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        rawMd += decoder.decode(value, { stream: true });
        queueRender();
      }
      rawMd += decoder.decode();
      renderMd(false);

      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      $("meta").textContent = (model ? model + " · " : "") + secs + "s" + (attempts > 1 ? " · " + attempts + " attempts" : "");
      $("pdfBtn").disabled = $("copyBtn").disabled = $("mdBtn").disabled = false;
    } catch (e) {
      showError("Could not reach the server. Check your connection and try again.");
    } finally {
      clearInterval(timer);
      setBusy(false);
    }
  }

  function downloadPdf() {
    const doc = document.createElement("div");
    doc.className = "pdf-doc";
    const date = new Date().toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
    doc.innerHTML =
      '<div class="pdf-head"><h1>' + esc(currentTopic) + "</h1><div>AI Research Report · " + date + "</div></div>" +
      DOMPurify.sanitize(marked.parse(rawMd));
    html2pdf().set({
      margin: [14, 14, 14, 14],
      filename: slug(currentTopic) + "-report.pdf",
      image: { type: "jpeg", quality: 0.95 },
      html2canvas: { scale: 2, useCORS: true },
      jsPDF: { unit: "mm", format: "a4", orientation: "portrait" },
      pagebreak: { mode: ["css", "legacy"], avoid: ["h2", "h3", "tr", "li"] },
    }).from(doc).save();
  }

  async function copyMd() {
    try {
      await navigator.clipboard.writeText(rawMd);
      $("copyBtn").textContent = "Copied!";
      setTimeout(() => ($("copyBtn").textContent = "Copy"), 1500);
    } catch (e) { alert("Copy failed. Select the text manually."); }
  }
  function downloadMd() {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([rawMd], { type: "text/markdown" }));
    a.download = slug(currentTopic) + "-report.md";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  $("runBtn").addEventListener("click", generate);
  $("pdfBtn").addEventListener("click", downloadPdf);
  $("copyBtn").addEventListener("click", copyMd);
  $("mdBtn").addEventListener("click", downloadMd);
  $("topic").addEventListener("keydown", (e) => { if (e.key === "Enter") generate(); });
  $("chips").addEventListener("click", (e) => {
    if (e.target.classList.contains("chip")) { $("topic").value = e.target.textContent; generate(); }
  });
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return Response(HTML_UI, mimetype='text/html')


def get_error_code(exc):
    """Pull an HTTP-style status code out of a google-genai exception."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    text = str(exc)
    for c in (503, 429, 500, 502, 504, 404, 403, 401, 400):
        if str(c) in text:
            return c
    return None


def text_chunks(stream):
    for chunk in stream:
        try:
            t = chunk.text
        except Exception:
            t = None
        if t:
            yield t


def open_stream(client, types, model, prompt, state):
    """Start a streaming call and wait for the first piece of text.
    Uses low 'thinking' for speed; silently drops it if the model rejects it."""
    options = [True, False] if state["thinking"] else [False]
    for use_thinking in options:
        config = None
        if use_thinking:
            try:
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                )
            except Exception:
                config = None
        try:
            stream = client.models.generate_content_stream(
                model=model, contents=prompt, config=config
            )
            gen = text_chunks(stream)
            try:
                first = next(gen)
            except StopIteration:
                raise RuntimeError("empty response")
            return gen, first
        except Exception as e:
            if use_thinking and get_error_code(e) == 400 and "think" in str(e).lower():
                state["thinking"] = False
                continue
            raise
    raise RuntimeError("could not open stream")


@app.route('/api/research', methods=['POST'])
def research():
    data = request.get_json(silent=True) or {}
    topic = (data.get('topic') or '').strip() or 'Artificial Intelligence'
    topic = topic[:300]
    length = LENGTHS.get(data.get('depth'), LENGTHS["standard"])

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({
            "error": "GEMINI_API_KEY is not set. Add it in Vercel > Project > "
                     "Settings > Environment Variables, then redeploy."
        }), 500

    try:
        from google import genai
        # pyrefly: ignore [missing-import]
        from google.genai import types
    except ImportError:
        return jsonify({
            "error": "The 'google-genai' package is missing. Add it to "
                     "requirements.txt and redeploy."
        }), 500

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )

    prompt = (
        f"Write a professional technical report in Markdown on: {topic}.\n"
        f"Length: {length}. Be dense and specific, no filler.\n"
        "Structure: a '# ' title, then '## Executive Summary', "
        "'## Technical Architecture', '## Key Use Cases', "
        "'## Challenges & Risks', '## Future Outlook'. "
        "Use bullet points, and a table where it helps. "
        "Start directly with the title, no preamble."
    )

    start = time.time()
    attempts = 0
    last_error = "Unknown error"
    state = {"thinking": True}

    for model in MODELS:
        delay = RETRY_DELAY_SECONDS
        for attempt in range(ATTEMPTS_PER_MODEL):
            if time.time() - start > TOTAL_BUDGET_SECONDS:
                last_error = "Timed out while waiting for Gemini."
                break
            attempts += 1
            try:
                gen, first = open_stream(client, types, model, prompt, state)

                def body(gen=gen, first=first):
                    try:
                        yield first
                        for piece in gen:
                            yield piece
                    except Exception as e:
                        yield f"\n\n---\n**Stream interrupted:** {e}"

                return Response(body(), mimetype='text/plain; charset=utf-8', headers={
                    "X-Model": model,
                    "X-Attempts": str(attempts),
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                })

            except Exception as e:
                code = get_error_code(e)
                last_error = f"{model}: {e}"

                if code in (400, 401, 403):
                    return jsonify({
                        "error": "Gemini rejected the request (check that "
                                 "GEMINI_API_KEY is valid and the Gemini API is "
                                 f"enabled for it).\nDetails: {e}"
                    }), 502

                if code == 404:
                    break  # model name not available -> next model

                if code in RETRYABLE_CODES or code is None:
                    if attempt < ATTEMPTS_PER_MODEL - 1:
                        time.sleep(delay)
                        delay *= 2
                    continue
                break

    return jsonify({
        "error": "All Gemini models are busy or unavailable right now. "
                 "This is a temporary capacity issue on Google's side, not a "
                 "problem with your key. Please wait a minute and try again.\n\n"
                 f"Last error: {last_error}"
    }), 503


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)