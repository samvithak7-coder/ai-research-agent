# AI Research Agent

A web app that turns any technical topic into a structured intelligence report in seconds. Enter a topic, pick a report length, and download the result as Markdown or PDF.

**Live Demo:** [https://ai-research-agent-amv8.vercel.app](https://ai-research-agent-amv8.vercel.app)

## Features

- Generate a structured technical report (executive summary, technical architecture and more) from any topic
- Choose the report length, for example Standard (~800 words)
- One-click topic suggestions such as Edge AI, Retrieval-Augmented Generation, Quantum Computing, Vision-Language Models and Zero-Trust Security
- Shows how long each report took to generate
- Copy the report to the clipboard
- Download as **Markdown (.md)** or **PDF**
- Deployed serverlessly on Vercel

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI | Google Gemini API |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Vercel |

## How It Works

1. The user enters a topic and selects a report length.
2. The frontend sends the request to the Python backend.
3. The backend prompts the Gemini API to write a structured report for that topic and length.
4. The report is rendered in the browser, with options to copy it or export it as Markdown or PDF.

## Project Structure

```
ai-research-agent/
├── app.py            # Backend and report generation logic
├── requirements.txt  # Python dependencies
└── vercel.json       # Vercel deployment configuration
```

## Getting Started

### Prerequisites

- Python 3.9+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### Installation

```bash
git clone https://github.com/samvithak7-coder/ai-research-agent.git
cd ai-research-agent
pip install -r requirements.txt
```

### Configuration

Set your API key as an environment variable:

```bash
export GEMINI_API_KEY=your_api_key_here
```

### Run locally

```bash
python app.py
```

Then open the local address shown in your terminal.

## Deployment

The app is deployed on Vercel using the included `vercel.json`. Add `GEMINI_API_KEY` as an environment variable in your Vercel project settings, and pushes to GitHub redeploy automatically.

## Author

**Kesapragada Venkata Prabha Sai Samvitha**
[Portfolio](https://samvitha-portfolio.vercel.app/) | [LinkedIn](https://www.linkedin.com/in/kesapragada-samvitha-977714376/) | [GitHub](https://github.com/samvithak7-coder)
