#!/usr/bin/env python3
"""
Quick test script for the NanoBanana diagram generator.

Usage:
  python scripts/test_nanobanana.py                    # pipeline template (default)
  python scripts/test_nanobanana.py --security         # security/governance template
  python scripts/test_nanobanana.py --model flash      # force flash model

Saves PNG to renderer/output/nanobanana_test.png and opens it.
"""
import os
import sys
import argparse
import pathlib

# ── env ───────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

# ── path setup ────────────────────────────────────────────────────────────────
_scripts = str(pathlib.Path(__file__).parent)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from nanobanana_diagram import build_nb_diagram, _IMAGE_MODELS

# ── test fixtures ─────────────────────────────────────────────────────────────

PIPELINE_USE_CASE = {
    "title":                "Real-Time Personalization Engine for E-Commerce",
    "category":             "ai_infra",
    "industry":             "E-Commerce / Retail",
    "problem":              "Recommendation latency >200ms hurts conversion; batch ML is stale within hours.",
    "solution_summary":     "Streaming feature pipeline + online feature store + sub-50ms Bedrock inference.",
    "architecture_insight": "Keeping the feature store write path separate from the inference path cuts p99 latency by 4×.",
    "business_impact":      "7% lift in add-to-cart rate, <50ms inference on 100k RPS.",
    "aws_services":         [
        "Amazon Kinesis Data Streams", "AWS Lambda", "Amazon DynamoDB",
        "Amazon ElastiCache", "Amazon Bedrock", "Amazon SageMaker",
        "AWS Step Functions", "Amazon S3", "AWS KMS", "Amazon CloudWatch",
    ],
    "agent_types":          ["Retrieval Agent", "Ranking Agent"],
    "solution_type":        "real-time ML",
    "source_company_or_idea": "Instacart-style grocery platform",
}

SECURITY_USE_CASE = {
    "title":                "Enterprise LLM Agent Governance Framework",
    "category":             "llm_security",
    "industry":             "Financial Services",
    "problem":              "Unguarded LLM agents exfiltrate PII and hallucinate compliance answers.",
    "solution_summary":     "Multi-layer guardrail pipeline with NeMo, WAF, RBAC, and immutable audit logs.",
    "architecture_insight": "Placing PII masking before the LLM gateway prevents prompt injection carrying sensitive data.",
    "business_impact":      "SOC 2 Type II + ISO 42001 compliance with <5ms guardrail overhead.",
    "aws_services":         [
        "AWS WAF", "Amazon Cognito", "AWS IAM", "AWS KMS",
        "Amazon Bedrock", "LangChain Agent Orchestration",
        "Pinecone Vector DB", "Amazon CloudTrail", "Amazon CloudWatch",
        "NeMo Guardrails", "AWS Secrets Manager",
    ],
    "agent_types":          ["Orchestration Agent", "Tool-Use Agent"],
    "solution_type":        "llm security governance",
    "source_company_or_idea": "Enterprise AI platform",
}


def main():
    parser = argparse.ArgumentParser(description="Test NanoBanana diagram generation")
    parser.add_argument("--security", action="store_true", help="Use security/governance template")
    parser.add_argument("--model",    choices=["pro", "flash"], default="pro",
                        help="Force model tier: pro (gemini-3-pro-image-preview) or flash (gemini-2.5-flash-image)")
    args = parser.parse_args()

    # Temporarily override model order if --model flash requested
    if args.model == "flash":
        import nanobanana_diagram as _nb
        _nb._IMAGE_MODELS = ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]
        print("  [test] Forcing flash model first.")

    use_case = SECURITY_USE_CASE if args.security else PIPELINE_USE_CASE
    template = "security" if args.security else "pipeline"
    print(f"\n[test_nanobanana] Template: {template}")
    print(f"[test_nanobanana] Use case: {use_case['title']}\n")

    out_path = str(pathlib.Path(__file__).parent.parent / "renderer" / "output" / "nanobanana_test.png")
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    png = build_nb_diagram(use_case, out_path)

    if png:
        print(f"\n✓ Diagram saved: {png}")
        # Try to open it for preview (macOS / Linux)
        try:
            import subprocess
            if sys.platform == "darwin":
                subprocess.run(["open", png], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", png], check=False)
        except Exception:
            pass
    else:
        print("\n✗ Diagram generation failed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
