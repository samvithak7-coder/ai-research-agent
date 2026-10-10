import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Autonomous AI Research Agent</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 40px; margin: 0; }
        .container { max-width: 750px; margin: auto; background: #1e293b; padding: 35px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.4); border: 1px solid #334155; }
        h1 { color: #38bdf8; font-size: 24px; margin-bottom: 8px; }
        p { color: #94a3b8; font-size: 13px; margin-bottom: 25px; }
        input { width: 100%; background: #0f172a; color: #fff; border: 1px solid #475569; border-radius: 8px; padding: 14px; font-size: 15px; margin-bottom: 20px; box-sizing: border-box; }
        button { background: #0284c7; color: white; border: none; padding: 14px 20px; font-size: 16px; font-weight: bold; border-radius: 8px; cursor: pointer; width: 100%; transition: background 0.2s; }
        button:hover { background: #0369a1; }
        .status { color: #fbbf24; font-size: 13px; font-weight: 500; margin-bottom: 15px; display: none; }
        .output { margin-top: 20px; background: #0f172a; padding: 25px; border-radius: 8px; border: 1px solid #334155; display: none; font-size: 14px; line-height: 1.7; color: #e2e8f0; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Autonomous AI Research Agent</h1>
        <p>Enter any technical topic to generate a structured multi-section intelligence report.</p>
        
        <input type="text" id="topic" placeholder="e.g., Edge AI & Vision-Language Models">
        <button onclick="generateReport()">Run Agentic Pipeline</button>
        
        <div id="status" class="status"></div>
        <div id="output" class="output"></div>
    </div>

    <script>
        async function generateReport() {
            const topic = document.getElementById('topic').value;
            const status = document.getElementById('status');
            const output = document.getElementById('output');
            
            if(!topic.trim()) { alert('Please enter a topic.'); return; }

            status.style.display = 'block';
            output.style.display = 'none';
            status.innerText = 'Step 1/2: Synthesizing research parameters via Gemini API...';

            try {
                const res = await fetch('/api/research', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ topic })
                });
                const data = await res.json();
                
                status.style.display = 'none';
                output.style.display = 'block';
                output.innerText = data.report;
            } catch(e) {
                status.style.display = 'none';
                output.style.display = 'block';
                output.innerText = 'Error executing pipeline.';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_UI)

@app.route('/api/research', methods=['POST'])
def research():
    data = request.get_json(silent=True) or {}
    topic = data.get('topic', 'Artificial Intelligence')

    try:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        # Explicitly updated to gemini-3.8-flash as required by the API error
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=f"Generate a professional, structured technical report on: {topic}. Include sections for Executive Summary, Technical Architecture, and Future Outlook."
        )
        report = response.text.strip()
    except Exception as e:
        report = f"[API Error]: {str(e)}\nPlease check your GEMINI_API_KEY environment variable on Vercel."

    return jsonify({"report": report}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)