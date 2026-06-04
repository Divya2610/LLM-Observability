# 🧠 LLM Observability & Evaluation Platform

A production-grade LLM monitoring system that automatically evaluates every LLM response and visualizes metrics in real-time.


<img width="940" height="520" alt="image" src="https://github.com/user-attachments/assets/4074c516-1662-46a6-b0a0-98f65a55e412" />


## 🏗️ Architecture
LLM App (FastAPI) → Prometheus → Grafana Dashboard
↓
Evaluation Engine
(Relevance, Faithfulness, Toxicity, ROUGE-L)
↓
Langfuse Traces

## ✨ Features

- **Real-time monitoring** — latency, token usage, cost per query
- **Auto-evaluation** — every response scored for relevance, faithfulness, toxicity
- **Live dashboard** — Grafana with 4 panels tracking model health
- **Multi-provider** — supports OpenAI, Anthropic, Groq (Llama3)
- **Trace logging** — full request/response history in Langfuse

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Groq (Llama 3.3 70B) |
| Tracing | Langfuse |
| Metrics | Prometheus |
| Dashboard | Grafana |
| Evaluation | ROUGE-L + LLM-as-judge |
| Database | PostgreSQL |

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Python 3.11+
- Groq API key (free at console.groq.com)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/llm-observability.git
cd llm-observability
```

### 2. Set up environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Fill in your API keys
```

### 4. Start infrastructure
```bash
docker compose up -d
```

### 5. Start the API server
```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Run evaluation suite
```bash
python evals/eval_runner.py
```

## 📊 Dashboard

Open Grafana at http://localhost:3001 (admin/admin)

![Dashboard](docs/dashboard.png)

## 📁 Project Structure
llm-observability/
├── app/
│   ├── main.py              # FastAPI server + Prometheus metrics
│   └── llm_wrapper.py       # Instrumented LLM calls
├── evals/
│   ├── scorers.py           # Relevance, faithfulness, toxicity, ROUGE
│   └── eval_runner.py       # Evaluation suite
├── infra/
│   └── prometheus.yml       # Prometheus scrape config
├── docker-compose.yml
└── .env.example

## 🧪 Evaluation Metrics

| Metric | Description |
|---|---|
| Relevance | Does the response answer the question? |
| Faithfulness | Does it hallucinate facts? |
| Toxicity | Is the content harmful? |
| ROUGE-L | Text overlap with reference answer |

