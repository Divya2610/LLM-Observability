import os
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse
from app.llm_wrapper import call_openai, call_anthropic, call_groq
from evals.scorers import score_relevance, score_faithfulness, score_toxicity

app = FastAPI(title="LLM Observability Platform")

# ── Prometheus Metrics ──────────────────────────────────────
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model", "provider", "status"]
)
LATENCY = Histogram(
    "llm_latency_seconds",
    "LLM response latency in seconds",
    ["model", "provider"],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
)
INPUT_TOKENS = Counter(
    "llm_input_tokens_total",
    "Total input tokens used",
    ["model", "provider"]
)
OUTPUT_TOKENS = Counter(
    "llm_output_tokens_total",
    "Total output tokens used",
    ["model", "provider"]
)
RELEVANCE_SCORE = Gauge(
    "llm_relevance_score",
    "Latest relevance score",
    ["model"]
)
FAITHFULNESS_SCORE = Gauge(
    "llm_faithfulness_score",
    "Latest faithfulness score",
    ["model"]
)
TOXICITY_SCORE = Gauge(
    "llm_toxicity_score",
    "Latest toxicity score",
    ["model"]
)


# ── Request Model ───────────────────────────────────────────
class QueryRequest(BaseModel):
    prompt: str
    model: str = "llama-3.3-70b-versatile"
    provider: str = "groq"
    context: str = None


# ── Routes ──────────────────────────────────────────────────
@app.post("/query")
async def query_llm(request: QueryRequest):
    try:
        # Call the right provider
        if request.provider == "openai":
            result = call_openai(request.prompt, request.model)
        elif request.provider == "anthropic":
            result = call_anthropic(request.prompt, request.model)
        else:
            result = call_groq(request.prompt, request.model)

        output = result["output"]
        provider = request.provider
        model = result["model"]

        # Record Prometheus metrics
        REQUEST_COUNT.labels(model=model, provider=provider, status="success").inc()
        LATENCY.labels(model=model, provider=provider).observe(result["latency"])
        INPUT_TOKENS.labels(model=model, provider=provider).inc(result["input_tokens"])
        OUTPUT_TOKENS.labels(model=model, provider=provider).inc(result["output_tokens"])

        # Auto-evaluate every response
        relevance = score_relevance(request.prompt, output)
        toxicity = score_toxicity(output)
        faithfulness = score_faithfulness(
            request.context or request.prompt, output
        )

        # Push eval scores to Prometheus
        RELEVANCE_SCORE.labels(model=model).set(relevance)
        FAITHFULNESS_SCORE.labels(model=model).set(faithfulness)
        TOXICITY_SCORE.labels(model=model).set(toxicity)

        return {
            "status": "success",
            "output": output,
            "model": model,
            "provider": provider,
            "latency_seconds": round(result["latency"], 3),
            "tokens": {
                "input": result["input_tokens"],
                "output": result["output_tokens"]
            },
            "scores": {
                "relevance": relevance,
                "faithfulness": faithfulness,
                "toxicity": toxicity
            }
        }

    except Exception as e:
        REQUEST_COUNT.labels(
            model=request.model,
            provider=request.provider,
            status="error"
        ).inc()
        return {"status": "error", "message": str(e)}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {
        "message": "LLM Observability Platform",
        "endpoints": {
            "query": "POST /query",
            "metrics": "GET /metrics",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }