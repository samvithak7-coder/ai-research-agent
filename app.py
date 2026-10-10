import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

try:
    from google import genai
    # pyrefly: ignore [missing-import]
    from google.genai import types
    IMPORT_ERROR = None
except Exception as _e:
    genai = types = None
    IMPORT_ERROR = str(_e)

app = Flask(__name__)
CORS(app)
POOL = ThreadPoolExecutor(max_workers=8)

# ---------------------------------------------------------------------------
# Config - fastest model first, automatic fallback if busy/slow.
# Override the primary with a GEMINI_MODEL env var on Vercel.
# ---------------------------------------------------------------------------
PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash"]
MODELS = [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m != PRIMARY_MODEL]

ATTEMPTS_PER_MODEL = 2
RETRY_DELAY_SECONDS = 0.5
PER_MODEL_TIMEOUT = 25
TOTAL_BUDGET_SECONDS = 50
MAX_QUESTIONS = 5
MAX_MESSAGES = 24
MAX_CHARS = 1200
RATE_LIMIT_PER_HOUR = 40  # best effort only (serverless memory isn't shared)

URGENCY_LEVELS = {"emergency", "urgent", "doctor", "self_care"}

SYSTEM_PROMPT = """You are a careful, kind symptom-triage assistant. You help people decide HOW URGENTLY they need medical care. You do NOT diagnose and you are not a doctor.

How to behave:
- Ask ONE short, focused follow-up question at a time. Cover the things that change urgency: how long, how severe (1-10), other symptoms, age-related risk, existing conditions, medicines, pregnancy where relevant, and any worsening.
- Never ask more than %d questions in total. Stop earlier once you have enough to decide.
- When uncertain, choose the MORE cautious urgency level.
- Never state a single definitive diagnosis. "possible_causes" must be general, common possibilities, phrased as possibilities.
- Do not give prescription drug names with doses. For common over-the-counter relief, only mention general options and tell the person to follow the label or ask a pharmacist.
- Reply in the same language the user writes in.
- Use plain, calm language. No scary wording, no jargon.

Urgency levels:
- "emergency": call local emergency services now (e.g. chest pain with breathlessness, stroke signs, severe bleeding, trouble breathing, confusion, fainting, severe allergic reaction, fever in a baby under 3 months, thoughts of self-harm).
- "urgent": needs medical care today (urgent care / same-day doctor / ER if it worsens).
- "doctor": book a doctor visit within the next few days.
- "self_care": can likely be managed at home, with clear signs for when to escalate.

Output ONLY valid JSON, in one of these two shapes:

1) A follow-up question:
{"type":"question","message":"<one short question>","options":["<short answer>","<short answer>","<short answer>"]}
(options: 2 to 5 short, tappable answers; the user may also type their own.)

2) The final result:
{"type":"result","urgency":"emergency|urgent|doctor|self_care","headline":"<one line>","timeframe":"<e.g. Call emergency services now / See a doctor today / Within 2-3 days / Home care>","summary":"<2-3 sentences explaining why, based on what the person said>","possible_causes":["<general possibility>","..."],"what_to_do":["<clear step>","..."],"warning_signs":["<sign that means go to emergency care>","..."],"questions_for_doctor":["<useful question>","..."]}
""" % MAX_QUESTIONS

# ---------------------------------------------------------------------------
# Safety pre-screen: obvious emergencies skip the AI entirely (instant + safe)
# ---------------------------------------------------------------------------
SELF_HARM = re.compile(
    r"suicid\w*|kill myself|end my life|want to die|wanna die|don'?t want to live|"
    r"self[- ]?harm|hurt myself|harm myself|cut myself|take my own life", re.I)

EMERGENCY = re.compile(
    r"can'?t breathe|cannot breathe|can not breathe|struggling to breathe|trouble breathing|"
    r"difficulty breathing|gasping|chest pain|chest tightness|chest pressure|pressure in (my )?chest|"
    r"face (is )?drooping|slurred speech|sudden(ly)? (weak|numb)|one side of (my )?(body|face)|"
    r"worst headache|unconscious|unresponsive|not responding|passed out|seizure|convulsion|"
    r"severe bleeding|won'?t stop bleeding|vomiting blood|coughing (up )?blood|blood in (my )?(vomit|stool)|"
    r"overdose|swallowed (poison|bleach)|throat (is )?(closing|swelling)|swollen (tongue|throat)|anaphyla", re.I)


def prescreen(text):
    if SELF_HARM.search(text):
        return {
            "type": "result", "urgency": "emergency", "crisis": True,
            "headline": "You're not alone. Please reach out for support right now.",
            "timeframe": "Talk to someone now",
            "summary": "What you wrote worries me, and your safety matters. This tool can't help with this, "
                       "but people who can are available right now.",
            "possible_causes": [],
            "what_to_do": [
                "If you might act on these thoughts, call your local emergency number now (112 in India and the EU, 911 in the US, 999 in the UK).",
                "Crisis support: Tele-MANAS 14416 (India), 988 call or text (US), Samaritans 116 123 (UK). If you're elsewhere, search for your country's suicide crisis line.",
                "Tell someone you trust where you are and how you feel, and try not to be alone.",
                "If you can, move away from anything you could use to hurt yourself.",
            ],
            "warning_signs": [],
            "questions_for_doctor": [],
        }
    if EMERGENCY.search(text):
        return {
            "type": "result", "urgency": "emergency", "crisis": False,
            "headline": "These symptoms can be serious. Get emergency help now.",
            "timeframe": "Call emergency services now",
            "summary": "Some of what you described can be a sign of a serious or life-threatening problem. "
                       "It's safest to get checked immediately rather than wait or keep answering questions.",
            "possible_causes": [],
            "what_to_do": [
                "Call your local emergency number now (112 in India and the EU, 108 for an ambulance in India, 911 in the US, 999 in the UK).",
                "Don't drive yourself. Ask someone nearby to stay with you.",
                "Stop eating and drinking, and sit or lie in the most comfortable position.",
                "If someone is unresponsive and not breathing normally, start CPR if you know how and the dispatcher can guide you.",
            ],
            "warning_signs": [],
            "questions_for_doctor": [],
        }
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
HITS = {}


def rate_limited(ip):
    now = time.time()
    recent = [t for t in HITS.get(ip, []) if now - t < 3600]
    if len(recent) >= RATE_LIMIT_PER_HOUR:
        HITS[ip] = recent
        return True
    recent.append(now)
    HITS[ip] = recent
    return False


def get_error_code(exc):
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    text = str(exc)
    for c in (503, 429, 500, 502, 504, 404, 403, 401, 400):
        if str(c) in text:
            return c
    return None


def make_client(api_key):
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=PER_MODEL_TIMEOUT * 1000),
    )


def call_model(client, model, system, contents):
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        temperature=0.3,
    )
    try:
        resp = client.models.generate_content(model=model, contents=contents, config=cfg)
    except Exception as e:
        if get_error_code(e) == 400 and "api key" not in str(e).lower():
            # some model may not accept the JSON-mode config: retry plain
            plain = types.GenerateContentConfig(system_instruction=system, temperature=0.3)
            resp = client.models.generate_content(model=model, contents=contents, config=plain)
        else:
            raise
    return (resp.text or "").strip()


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON found")
    return json.loads(text[start:end + 1])


def str_list(v, limit=8):
    if not isinstance(v, list):
        return []
    return [str(x).strip()[:400] for x in v if str(x).strip()][:limit]


def normalize(obj):
    if obj.get("type") == "result" and obj.get("urgency") in URGENCY_LEVELS:
        return {
            "type": "result",
            "urgency": obj["urgency"],
            "crisis": False,
            "headline": str(obj.get("headline", "")).strip()[:200],
            "timeframe": str(obj.get("timeframe", "")).strip()[:120],
            "summary": str(obj.get("summary", "")).strip()[:900],
            "possible_causes": str_list(obj.get("possible_causes")),
            "what_to_do": str_list(obj.get("what_to_do")),
            "warning_signs": str_list(obj.get("warning_signs")),
            "questions_for_doctor": str_list(obj.get("questions_for_doctor")),
        }
    if obj.get("type") == "question" and str(obj.get("message", "")).strip():
        return {
            "type": "question",
            "message": str(obj["message"]).strip()[:400],
            "options": str_list(obj.get("options"), 5),
        }
    raise ValueError("unexpected shape")


def setup_error():
    if not os.environ.get("GEMINI_API_KEY"):
        return ("GEMINI_API_KEY is not set. Add it in Vercel > Project > "
                "Settings > Environment Variables, then redeploy.")
    if IMPORT_ERROR:
        return "The 'google-genai' package failed to load: " + IMPORT_ERROR
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return Response(HTML_UI, mimetype='text/html')


@app.route('/api/triage', methods=['POST'])
def triage():
    data = request.get_json(silent=True) or {}
    raw = data.get("messages")
    if not isinstance(raw, list) or not raw:
        return jsonify({"error": "Please describe your symptoms first."}), 400

    messages = []
    for m in raw[-MAX_MESSAGES:]:
        if not isinstance(m, dict):
            continue
        role = "user" if m.get("role") == "user" else "model"
        text = str(m.get("text", "")).strip()[:MAX_CHARS]
        if text:
            messages.append((role, text))
    if not messages or messages[-1][0] != "user":
        return jsonify({"error": "Invalid conversation."}), 400

    # 1) Instant safety screen on the newest user message
    early = prescreen(messages[-1][1])
    if early:
        return jsonify(early), 200

    # 2) Basic abuse protection
    ip = (request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
          or request.remote_addr or "unknown")
    if rate_limited(ip):
        return jsonify({"error": "Too many requests. Please try again in a while."}), 429

    err = setup_error()
    if err:
        return jsonify({"error": err}), 500

    profile = data.get("profile") or {}
    age = str(profile.get("age", "")).strip()[:3]
    sex = str(profile.get("sex", "")).strip()[:20]
    prefix = ""
    if age or sex:
        prefix = "[Patient profile: " + ", ".join(
            x for x in (f"age {age}" if age else "", f"sex {sex}" if sex else "") if x) + "]\n"

    asked = sum(1 for r, _ in messages if r == "model")
    system = SYSTEM_PROMPT + f"\nQuestions asked so far: {asked} of max {MAX_QUESTIONS}."
    if asked >= MAX_QUESTIONS:
        system += " You MUST now give the final result (type 'result'). Do not ask another question."

    contents = []
    for i, (role, text) in enumerate(messages):
        if i == 0 and role == "user":
            text = prefix + text
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

    client = make_client(os.environ["GEMINI_API_KEY"])
    start = time.time()
    log = []
    out_of_time = False

    for model in MODELS:
        if out_of_time:
            break
        for attempt in range(ATTEMPTS_PER_MODEL):
            remaining = TOTAL_BUDGET_SECONDS - (time.time() - start)
            if remaining < 5:
                out_of_time = True
                log.append("Ran out of time budget.")
                break
            wait = min(PER_MODEL_TIMEOUT, remaining)
            fut = POOL.submit(call_model, client, model, system, contents)
            try:
                text = fut.result(timeout=wait)
                result = normalize(parse_json(text))
                if asked >= MAX_QUESTIONS and result["type"] == "question":
                    log.append(f"{model}: asked another question after limit")
                    continue
                return jsonify(result), 200
            except FutTimeout:
                log.append(f"{model}: no answer within {int(wait)}s")
                break
            except (ValueError, json.JSONDecodeError) as e:
                log.append(f"{model}: unreadable reply ({e})")
                continue
            except Exception as e:
                code = get_error_code(e)
                msg = str(e).replace("\n", " ")[:200]
                log.append(f"{model}: {msg}")
                if code in (401, 403) or "api key" in msg.lower():
                    return jsonify({"error": "Gemini rejected the API key. Check GEMINI_API_KEY in Vercel."}), 502
                if code in (404, 400):
                    break
                if attempt < ATTEMPTS_PER_MODEL - 1:
                    time.sleep(RETRY_DELAY_SECONDS)

    return jsonify({
        "error": "The AI service is busy right now. Please try again in a minute. "
                 "If you feel very unwell, don't wait: contact a doctor or emergency services.",
        "details": log,
    }), 503


@app.route('/api/diag')
def diag():
    err = setup_error()
    info = {"key_present": bool(os.environ.get("GEMINI_API_KEY")),
            "import_error": IMPORT_ERROR, "models": MODELS}
    if err:
        info["error"] = err
        return jsonify(info), 500
    client = make_client(os.environ["GEMINI_API_KEY"])
    t0 = time.time()
    futs = {m: POOL.submit(lambda mm=m: client.models.generate_content(
        model=mm, contents="Reply with just: OK").text) for m in MODELS}
    results = {}
    for m, f in futs.items():
        try:
            out = f.result(timeout=max(1, 20 - (time.time() - t0)))
            results[m] = {"ok": True, "reply": (out or "")[:20]}
        except FutTimeout:
            results[m] = {"ok": False, "error": "no answer within 20s"}
        except Exception as e:
            results[m] = {"ok": False, "error": str(e)[:300]}
    info["results"] = results
    return jsonify(info), 200


HTML_UI = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Symptom Triage Assistant</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
<style>
  :root { --bg:#06121a; --card:rgba(18,36,48,.7); --border:rgba(148,163,184,.18); --text:#e2e8f0; --muted:#94a3b8; --teal:#2dd4bf; --blue:#38bdf8; }
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; color:var(--text); padding:28px 14px 60px;
    font-family: Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background: radial-gradient(800px 480px at 10% -10%, rgba(45,212,191,.18), transparent 60%),
                radial-gradient(800px 480px at 95% 0%, rgba(56,189,248,.16), transparent 60%), var(--bg); }
  .wrap { max-width: 780px; margin: auto; }
  .badge { display:inline-flex; gap:8px; align-items:center; font-size:12px; color:var(--teal);
    background:rgba(45,212,191,.1); border:1px solid rgba(45,212,191,.25); padding:5px 12px; border-radius:999px; margin-bottom:14px; }
  h1 { font-size:clamp(26px,5vw,36px); margin:0 0 8px; letter-spacing:-.02em;
    background:linear-gradient(90deg,var(--teal),var(--blue)); -webkit-background-clip:text; background-clip:text; color:transparent; }
  .sub { color:var(--muted); font-size:14px; margin:0 0 18px; }
  .warn { background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.35); color:#fecaca; border-radius:12px;
    padding:12px 14px; font-size:13px; line-height:1.55; margin-bottom:18px; }
  .warn b { color:#fff; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:20px; backdrop-filter:blur(10px); box-shadow:0 20px 50px rgba(0,0,0,.35); }
  label { display:block; font-size:12px; color:var(--muted); margin-bottom:6px; }
  input,select,textarea { width:100%; background:#08171f; color:#fff; border:1px solid #334155; border-radius:10px; padding:12px 14px; font-size:15px; outline:none; font-family:inherit; }
  textarea { resize:vertical; min-height:96px; }
  input:focus,select:focus,textarea:focus { border-color:var(--teal); box-shadow:0 0 0 3px rgba(45,212,191,.15); }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px; }
  .btn { border:none; cursor:pointer; font-weight:600; font-size:15px; border-radius:10px; padding:0 22px; height:46px; color:#04201c;
    background:linear-gradient(135deg,var(--teal),var(--blue)); transition:transform .15s,opacity .2s; }
  .btn:hover { transform:translateY(-1px); } .btn:disabled { opacity:.5; cursor:not-allowed; transform:none; }
  .btn.full { width:100%; margin-top:14px; }
  .btn.ghost { background:rgba(148,163,184,.12); border:1px solid var(--border); color:var(--text); height:38px; font-size:13px; padding:0 14px; }
  .chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
  .chip { font-size:12.5px; color:var(--muted); background:rgba(148,163,184,.08); border:1px solid var(--border); padding:7px 13px; border-radius:999px; cursor:pointer; transition:.2s; }
  .chip:hover { color:#fff; border-color:var(--teal); }

  #chat { display:none; }
  #log { display:flex; flex-direction:column; gap:12px; max-height:52vh; overflow-y:auto; padding:4px 2px 10px; }
  .msg { max-width:86%; padding:11px 15px; border-radius:16px; font-size:15px; line-height:1.55; white-space:pre-wrap; overflow-wrap:anywhere; }
  .msg.user { align-self:flex-end; background:linear-gradient(135deg,#0d9488,#0284c7); color:#fff; border-bottom-right-radius:4px; }
  .msg.ai { align-self:flex-start; background:#0d2230; border:1px solid var(--border); border-bottom-left-radius:4px; }
  .msg.err { align-self:flex-start; background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.4); color:#fecaca; }
  .typing { align-self:flex-start; display:flex; gap:5px; padding:14px 16px; background:#0d2230; border:1px solid var(--border); border-radius:16px; }
  .typing i { width:7px; height:7px; border-radius:50%; background:var(--teal); animation:b 1.2s infinite; }
  .typing i:nth-child(2){animation-delay:.2s} .typing i:nth-child(3){animation-delay:.4s}
  @keyframes b { 0%,60%,100%{opacity:.25;transform:translateY(0)} 30%{opacity:1;transform:translateY(-4px)} }
  #opts { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 4px; }
  .opt { cursor:pointer; font-size:14px; padding:9px 15px; border-radius:999px; color:var(--teal); background:rgba(45,212,191,.08); border:1px solid rgba(45,212,191,.35); transition:.2s; }
  .opt:hover { background:rgba(45,212,191,.2); }
  #composer { display:flex; gap:10px; margin-top:12px; }
  #composer input { flex:1; }
  #composer .btn { height:46px; }
  .progress { font-size:12px; color:var(--muted); margin-bottom:10px; }

  #result { display:none; margin-top:18px; }
  .banner { border-radius:14px; padding:18px 20px; margin-bottom:14px; border:1px solid; }
  .banner .lvl { font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:6px; }
  .banner h2 { margin:0 0 6px; font-size:20px; color:#fff; }
  .banner .tf { font-size:14px; opacity:.95; }
  .u-emergency { background:rgba(239,68,68,.15); border-color:#ef4444; color:#fecaca; }
  .u-urgent { background:rgba(249,115,22,.15); border-color:#f97316; color:#fed7aa; }
  .u-doctor { background:rgba(234,179,8,.13); border-color:#eab308; color:#fef08a; }
  .u-self_care { background:rgba(34,197,94,.13); border-color:#22c55e; color:#bbf7d0; }
  .sec { background:#08171f; border:1px solid var(--border); border-radius:12px; padding:14px 18px; margin-bottom:12px; }
  .sec h3 { margin:0 0 8px; font-size:14px; color:var(--teal); }
  .sec p { margin:0; line-height:1.65; font-size:14.5px; }
  .sec ul { margin:0; padding-left:20px; } .sec li { margin:6px 0; line-height:1.55; font-size:14.5px; }
  .actions { display:flex; gap:10px; flex-wrap:wrap; margin-top:6px; }
  .foot { font-size:12px; color:#64748b; margin-top:16px; line-height:1.55; }

  .pdf-doc { color:#111827; background:#fff; font-family:Arial,sans-serif; font-size:12px; line-height:1.6; padding:4px; }
  .pdf-doc h1 { font-size:20px; margin:0 0 2px; color:#0f766e; } .pdf-doc .d { color:#6b7280; font-size:10.5px; margin-bottom:12px; }
  .pdf-doc .b { border:2px solid; border-radius:8px; padding:10px 14px; margin-bottom:12px; }
  .pdf-doc .b h2 { margin:0 0 3px; font-size:15px; } .pdf-doc h3 { font-size:13px; color:#0f766e; margin:12px 0 4px; }
  .pdf-doc ul { margin:4px 0; padding-left:18px; } .pdf-doc .disc { margin-top:16px; padding:8px 10px; background:#fef2f2; border:1px solid #fecaca; border-radius:6px; font-size:10.5px; color:#7f1d1d; }
  .pdf-doc .qa { font-size:11px; color:#374151; }
</style>
</head>
<body>
<div class="wrap">
  <h1>AI Symptom Triage Assistant</h1>
  <p class="sub">Describe how you feel. I'll ask a few questions, then suggest how urgently you should get care.</p>

  <div class="warn"><b>Not medical advice.</b> This tool gives general information only and cannot diagnose anything.
    <b>In an emergency, call your local emergency number immediately</b> (112 India/EU, 108 ambulance India, 911 US, 999 UK).</div>

  <div class="card" id="start">
    <div class="grid">
      <div><label for="age">Age (optional)</label><input id="age" type="number" min="0" max="120" placeholder="e.g. 29"></div>
      <div><label for="sex">Sex (optional)</label>
        <select id="sex"><option value="">Prefer not to say</option><option>Female</option><option>Male</option><option>Other</option></select></div>
    </div>
    <label for="first">What symptoms are you having?</label>
    <textarea id="first" placeholder="e.g. I've had a sore throat and fever since yesterday..."></textarea>
    <div class="chips" id="examples">
      <span class="chip">Headache for 2 days</span>
      <span class="chip">Fever and body aches</span>
      <span class="chip">Stomach pain after eating</span>
      <span class="chip">Cough and sore throat</span>
    </div>
    <button class="btn full" id="startBtn">Start assessment</button>
  </div>

  <div class="card" id="chat">
    <div class="progress" id="progress"></div>
    <div id="log"></div>
    <div id="opts"></div>
    <div id="composer">
      <input id="reply" type="text" placeholder="Type your answer..." autocomplete="off">
      <button class="btn" id="sendBtn">Send</button>
    </div>
  </div>

  <div id="result">
    <div id="banner"></div>
    <div id="sections"></div>
    <div class="actions">
      <button class="btn ghost" id="pdfBtn">Download PDF summary</button>
      <button class="btn ghost" id="resetBtn">Start over</button>
    </div>
    <div class="foot">This is general information, not a diagnosis or a substitute for a professional. If symptoms get worse or you're worried, seek medical care. Nothing you type is stored by this app.</div>
  </div>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  const MAXQ = 5;
  let messages = [], profile = {}, busy = false, lastResult = null, started = null;
  const LABEL = { emergency:"Emergency", urgent:"Urgent: get care today", doctor:"See a doctor soon", self_care:"Likely manageable at home" };
  const COLOR = { emergency:"#fecaca", urgent:"#fed7aa", doctor:"#fef08a", self_care:"#bbf7d0" };
  const BORDER = { emergency:"#ef4444", urgent:"#f97316", doctor:"#eab308", self_care:"#22c55e" };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

  function addMsg(text, cls) {
    const d = document.createElement("div");
    d.className = "msg " + cls; d.textContent = text;
    $("log").appendChild(d); $("log").scrollTop = $("log").scrollHeight; return d;
  }
  function typing(on) {
    const old = document.getElementById("typing"); if (old) old.remove();
    if (!on) return;
    const t = document.createElement("div"); t.className = "typing"; t.id = "typing";
    t.innerHTML = "<i></i><i></i><i></i>"; $("log").appendChild(t); $("log").scrollTop = $("log").scrollHeight;
  }
  function setBusy(b) { busy = b; $("sendBtn").disabled = b; $("reply").disabled = b; typing(b); if (b) $("opts").innerHTML = ""; }
  function updateProgress() {
    const q = messages.filter((m) => m.role === "assistant").length;
    $("progress").textContent = "Question " + Math.min(q + 1, MAXQ) + " of up to " + MAXQ + " · I'll stop sooner if I have enough information";
  }

  async function ask() {
    setBusy(true);
    let data;
    try {
      const res = await fetch("/api/triage", { method:"POST", headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({ messages, profile }) });
      data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");
    } catch (e) {
      setBusy(false);
      const d = addMsg((e.message || "Network error") + " ", "err");
      const b = document.createElement("a"); b.textContent = "Retry"; b.href = "#"; b.style.color = "#fff";
      b.onclick = (ev) => { ev.preventDefault(); d.remove(); ask(); };
      d.appendChild(b); return;
    }
    setBusy(false);
    if (data.type === "question") {
      messages.push({ role:"assistant", text:data.message });
      addMsg(data.message, "ai"); updateProgress();
      (data.options || []).forEach((o) => {
        const s = document.createElement("span"); s.className = "opt"; s.textContent = o;
        s.onclick = () => send(o); $("opts").appendChild(s);
      });
      $("reply").focus();
    } else { showResult(data); }
  }

  function send(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    messages.push({ role:"user", text }); addMsg(text, "user");
    $("reply").value = ""; $("opts").innerHTML = ""; ask();
  }

  function sec(title, items, isList) {
    if (!items || (Array.isArray(items) && !items.length) || items === "") return null;
    const d = document.createElement("div"); d.className = "sec";
    const h = document.createElement("h3"); h.textContent = title; d.appendChild(h);
    if (isList) { const ul = document.createElement("ul");
      items.forEach((i) => { const li = document.createElement("li"); li.textContent = i; ul.appendChild(li); }); d.appendChild(ul);
    } else { const p = document.createElement("p"); p.textContent = items; d.appendChild(p); }
    return d;
  }

  function showResult(r) {
    lastResult = r; $("chat").style.display = "none"; $("result").style.display = "block";
    const b = $("banner"); b.innerHTML = "";
    const box = document.createElement("div"); box.className = "banner u-" + r.urgency;
    const l = document.createElement("div"); l.className = "lvl"; l.textContent = LABEL[r.urgency] || r.urgency;
    const h = document.createElement("h2"); h.textContent = r.headline || LABEL[r.urgency];
    const t = document.createElement("div"); t.className = "tf"; t.textContent = r.timeframe || "";
    box.append(l, h, t); b.appendChild(box);
    const s = $("sections"); s.innerHTML = "";
    [sec("Why", r.summary, false), sec("What to do now", r.what_to_do, true),
     sec("Possible causes (not a diagnosis)", r.possible_causes, true),
     sec("Go to emergency care if...", r.warning_signs, true),
     sec("Questions to ask your doctor", r.questions_for_doctor, true)]
      .filter(Boolean).forEach((x) => s.appendChild(x));
    $("result").scrollIntoView({ behavior:"smooth" });
  }

  function li(a) { return (a && a.length) ? "<ul>" + a.map((x) => "<li>" + esc(x) + "</li>").join("") + "</ul>" : ""; }
  function pdf() {
    const r = lastResult; if (!r) return;
    const date = new Date().toLocaleDateString(undefined, { year:"numeric", month:"long", day:"numeric" });
    const info = [profile.age ? "Age " + profile.age : "", profile.sex || ""].filter(Boolean).join(", ");
    const qa = messages.map((m) => "<div>" + (m.role === "user" ? "<b>You:</b> " : "<b>Assistant:</b> ") + esc(m.text) + "</div>").join("");
    const doc = document.createElement("div"); doc.className = "pdf-doc";
    doc.innerHTML =
      "<h1>Symptom Triage Summary</h1><div class='d'>" + date + (info ? " · " + esc(info) : "") + "</div>" +
      "<div class='b' style='border-color:" + BORDER[r.urgency] + ";background:" + COLOR[r.urgency] + "33'>" +
      "<div style='font-size:10px;font-weight:700;text-transform:uppercase'>" + esc(LABEL[r.urgency] || "") + "</div>" +
      "<h2>" + esc(r.headline) + "</h2><div>" + esc(r.timeframe) + "</div></div>" +
      (r.summary ? "<h3>Why</h3><p>" + esc(r.summary) + "</p>" : "") +
      (r.what_to_do.length ? "<h3>What to do now</h3>" + li(r.what_to_do) : "") +
      (r.possible_causes.length ? "<h3>Possible causes (not a diagnosis)</h3>" + li(r.possible_causes) : "") +
      (r.warning_signs.length ? "<h3>Go to emergency care if...</h3>" + li(r.warning_signs) : "") +
      (r.questions_for_doctor.length ? "<h3>Questions to ask your doctor</h3>" + li(r.questions_for_doctor) : "") +
      "<h3>Your conversation</h3><div class='qa'>" + qa + "</div>" +
      "<div class='disc'><b>Disclaimer:</b> This summary is general information generated by AI. It is not a diagnosis or medical advice. In an emergency call your local emergency number.</div>";
    html2pdf().set({ margin:[12,12,12,12], filename:"symptom-triage-summary.pdf", image:{type:"jpeg",quality:.95},
      html2canvas:{scale:2}, jsPDF:{unit:"mm",format:"a4"}, pagebreak:{mode:["css","legacy"],avoid:["h3","li"]} }).from(doc).save();
  }

  function begin() {
    const text = $("first").value.trim();
    if (!text) { $("first").focus(); return; }
    profile = { age: $("age").value.trim(), sex: $("sex").value };
    messages = []; $("log").innerHTML = "";
    $("start").style.display = "none"; $("chat").style.display = "block";
    messages.push({ role:"user", text }); addMsg(text, "user"); updateProgress(); ask();
  }
  function reset() {
    messages = []; lastResult = null; $("result").style.display = "none"; $("chat").style.display = "none";
    $("start").style.display = "block"; $("first").value = ""; window.scrollTo({ top:0, behavior:"smooth" });
  }

  $("startBtn").onclick = begin;
  $("sendBtn").onclick = () => send($("reply").value);
  $("reply").addEventListener("keydown", (e) => { if (e.key === "Enter") send($("reply").value); });
  $("pdfBtn").onclick = pdf; $("resetBtn").onclick = reset;
  $("examples").onclick = (e) => { if (e.target.classList.contains("chip")) { $("first").value = e.target.textContent; $("first").focus(); } };
</script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)