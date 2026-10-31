"""
Ideation module — Step 1 of the business use case pipeline.

Gemini invents a fresh business use case or startup idea each run.
Returns a structured dict that feeds both the post prompt and the
diagram prompt downstream.

No topic list needed — every run produces something new.
"""

import json
import re

# ── prompt ────────────────────────────────────────────────────────────────────

_IDEATION_SYSTEM = """
You are a senior solutions architect and startup advisor with deep knowledge
of AWS, modern AI/ML stacks, and real business problems across industries.
You generate concrete, technically grounded business use cases and startup ideas
that a senior engineer would find interesting to build.
""".strip()

_IDEATION_PROMPT = """
Invent ONE compelling business use case or startup idea for today's LinkedIn post.

Requirements:
- It must be a REAL problem that exists in a specific industry (not generic "AI for X")
- It must have a production-grade technical solution involving AWS services
- It must be interesting to senior engineers and tech-savvy business readers
- Vary across industries: fintech, healthtech, retail, logistics, SaaS, legal, real estate,
  manufacturing, media, cybersecurity, agritech, edtech, proptech, etc.
- Vary across solution types: agentic AI, real-time ML, document intelligence, fraud/anomaly
  detection, recommendation systems, RAG/search, predictive analytics, IoT + ML, LLMOps, etc.

Output ONLY valid JSON matching this schema. No markdown fences, no explanation:
{{
  "title": "5-10 word catchy title for the use case",
  "industry": "single industry name",
  "company_type": "startup | enterprise | both",
  "problem": "2-3 sentences describing the specific business pain — concrete, not generic",
  "solution_summary": "2-3 sentences describing the AI/ML solution approach",
  "business_impact": "1 sentence with a plausible metric (cost, time, accuracy — be specific)",
  "solution_type": "one of: agentic_ai | real_time_ml | document_intelligence | fraud_detection | recommendation | rag_search | predictive_analytics | iot_ml | llmops | data_platform",
  "aws_services": ["list", "of", "5-8", "specific", "AWS service names"],
  "complexity": "medium | high",
  "linkedin_hook": "one vivid sentence that would stop a senior engineer scrolling — specific, no hype"
}}

Today's date context: vary the industry and solution type — avoid repeating the same domain
every day. The schema must be exactly as above. aws_services must be specific service names
(e.g. \"Amazon Bedrock\", \"AWS Step Functions\", \"Amazon SageMaker\", not generic like \"AI service\").
""".strip()


def generate_use_case(generate_text_fn: callable) -> dict:
    """
    Ask Gemini to invent a business use case.
    Returns the parsed dict or raises RuntimeError after 3 failed attempts.
    """
    for attempt in range(3):
        try:
            raw = generate_text_fn(_IDEATION_PROMPT, _IDEATION_SYSTEM).strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            raw = raw.strip()
            data = json.loads(raw)

            required = {"title", "industry", "problem", "solution_summary",
                        "aws_services", "solution_type", "linkedin_hook"}
            missing = required - data.keys()
            if missing:
                print(f"  [ideation] attempt {attempt+1}: missing fields {missing}, retrying...")
                continue

            if not isinstance(data.get("aws_services"), list) or len(data["aws_services"]) < 3:
                print(f"  [ideation] attempt {attempt+1}: aws_services too short, retrying...")
                continue

            print(f"  [ideation] Use case: {data['title']} ({data['industry']})")
            return data

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [ideation] attempt {attempt+1}: parse error ({e}), retrying...")

    raise RuntimeError("Failed to generate a valid use case after 3 attempts.")
