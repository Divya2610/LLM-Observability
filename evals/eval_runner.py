import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langfuse import Langfuse
from evals.scorers import score_relevance, score_faithfulness, score_toxicity, score_rouge
from app.llm_wrapper import call_groq

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# ✅ Test dataset
TEST_CASES = [
    {
        "prompt": "What is machine learning?",
        "context": "Machine learning is a subset of AI where systems learn from data to improve performance without being explicitly programmed.",
        "reference": "Machine learning is an AI technique where models learn patterns from data."
    },
    {
        "prompt": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital city is Paris, which is also its largest city.",
        "reference": "The capital of France is Paris."
    },
    {
        "prompt": "Explain neural networks simply.",
        "context": "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of layers of interconnected nodes.",
        "reference": "Neural networks are layered systems of nodes inspired by the human brain that process information."
    }
]


def evaluate_response(prompt, response, context=None, reference=None):
    """Run all scorers on a single response"""
    scores = {}

    print("    → Scoring relevance...")
    scores["relevance"] = score_relevance(prompt, response)

    if context:
        print("    → Scoring faithfulness...")
        scores["faithfulness"] = score_faithfulness(context, response)

    print("    → Scoring toxicity...")
    scores["toxicity"] = score_toxicity(response)

    if reference:
        print("    → Scoring ROUGE-L...")
        scores["rouge_l"] = score_rouge(reference, response)

    return scores


def run_eval_suite():
    print("\n" + "="*55)
    print("🧪  LLM EVALUATION SUITE STARTING")
    print("="*55)

    all_scores = []

    for i, test in enumerate(TEST_CASES):
        print(f"\n📌 Test {i+1}/{len(TEST_CASES)}: {test['prompt']}")
        print("  ⏳ Calling Groq (llama3)...")

        # Get LLM response
        result = call_groq(test["prompt"])
        response = result["output"]

        print(f"  💬 Response: {response[:100]}...")
        print(f"  ⚡ Latency : {round(result['latency'], 2)}s")
        print(f"  🔢 Tokens  : {result['input_tokens']} in / {result['output_tokens']} out")
        print("  📊 Evaluating...")

        # Score it
        scores = evaluate_response(
            prompt=test["prompt"],
            response=response,
            context=test["context"],
            reference=test["reference"]
        )

        # Push scores to Langfuse
        try:
            for score_name, score_value in scores.items():
                langfuse.create_event(
                    name=f"score-{score_name}",
                    metadata={
                        "score_name": score_name,
                        "score_value": score_value,
                        "prompt": test["prompt"]
                    }
                )
            langfuse.flush()
        except Exception as e:
            print(f"  [Langfuse score log skipped: {e}]")

        all_scores.append(scores)
        print(f"  ✅ Scores: {scores}")

    # Final summary
    print("\n" + "="*55)
    print("📊  FINAL EVALUATION SUMMARY")
    print("="*55)

    for metric in ["relevance", "faithfulness", "toxicity", "rouge_l"]:
        values = [s[metric] for s in all_scores if metric in s]
        if values:
            avg = round(sum(values) / len(values), 3)
            bar = "█" * int(avg * 20)
            print(f"  {metric:<15} {avg:.3f}  {bar}")

    print("\n✅ Done! Check Langfuse at http://localhost:3000 for detailed traces.")
    print("="*55 + "\n")


if __name__ == "__main__":
    run_eval_suite()