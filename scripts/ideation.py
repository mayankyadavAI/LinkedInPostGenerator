"""
Ideation module — Step 1 of the business use case pipeline.

Gemini invents a fresh business use case or startup idea each run.
Covers: multi-agent systems, YC-style startups, enterprise AI transformations,
and emerging AI-native product patterns.

No fixed topic list — every run produces something different.
"""

import json
import re

# ── prompts ───────────────────────────────────────────────────────────────────

_IDEATION_SYSTEM = """
You are a principal solutions architect and venture-backed startup advisor.
You have deep hands-on experience with AWS, multi-agent AI systems, LLMOps,
and production ML across industries. You think like a YC partner evaluating
whether a startup idea is technically feasible and commercially interesting.

You generate concrete, technically grounded business use cases and startup ideas
that a senior engineer building in 2025-2026 would find genuinely worth reading.
""".strip()

_IDEATION_PROMPT = """
Invent ONE compelling business use case or startup idea for a LinkedIn post targeting
senior engineers and technical leaders.

{category_instruction}

Requirements:
- Real, specific problem — not "AI for X" in generic terms
- Production-grade AWS architecture is clearly implied
- Interesting to people building, not just talking about AI
- Include at least one non-obvious architectural insight
- Vary the specific industry and use case within the chosen category

Output ONLY valid JSON. No markdown fences, no explanation:
{{
  "title": "5-10 word title that reads like a product name or LinkedIn hook",
  "category": "one of: multi_agent | yc_startup | enterprise_ai | ai_infra",
  "industry": "specific industry vertical (e.g. legal-tech, supply-chain, healthcare-ops)",
  "company_type": "startup | enterprise | both",
  "problem": "2-3 sentences — the specific, concrete pain. Name the workflow, the cost, the breakage.",
  "solution_summary": "2-3 sentences — the AI/ML architecture. Name the agent types, models, and data flows.",
  "architecture_insight": "1 sentence — the non-obvious design decision that makes this production-grade vs toy",
  "business_impact": "1 sentence — plausible metric. Be specific: time saved, error rate reduced, cost cut.",
  "solution_type": "one of: multi_agent_orchestration | agentic_rag | real_time_ml | document_intelligence | fraud_detection | llm_evaluation | llmops_platform | iot_ml | predictive_analytics | data_platform",
  "aws_services": [
    "8-12 specific AWS service names — include networking, compute, ML, storage, security, and observability layers"
  ],
  "agent_types": ["list agent roles if multi-agent, else empty list"],
  "complexity": "high",
  "linkedin_hook": "one sentence hook — specific observation or number that would stop a senior engineer scrolling"
}}

aws_services must use official AWS service names (e.g. \"Amazon Bedrock\",
\"AWS Step Functions\", \"Amazon SageMaker\", \"Amazon Bedrock Agents\",
\"AWS Lambda\", \"Amazon Aurora\", \"Amazon OpenSearch Service\",
\"AWS WAF\", \"Amazon CloudWatch\", \"AWS X-Ray\").
Include at least one security service and one observability service.
""".strip()

# Category instructions — injected into the prompt based on IDEA_CATEGORY env var
_CATEGORY_INSTRUCTIONS = {
    "multi_agent": """\
Category: Multi-agent AI system — multiple specialized LLM agents collaborating.
Pick a SPECIFIC industry and workflow. Examples of patterns to draw from:
  - Research + reasoning + action agents with tool-use pipelines
  - Orchestrator → specialist agents with critic/review loops
  - Hierarchical agent systems (planner → executor → validator)
  - Multi-agent debate or adversarial verification patterns
  - Agentic RAG with memory, re-ranking, and grounding agents
Vary the industry each run: legal, finance, healthcare, logistics, manufacturing,
cybersecurity, real estate, procurement, insurance, aerospace, media, etc.
Include agent_types with specific role names for each agent in the system.""",

    "yc_startup": """\
Category: YC-style AI-native startup — a specific niche problem a small team could own.
Think: AI for a workflow that currently requires expensive human specialists.
Examples: AI for medical coding, construction estimation, legal contract review,
customs compliance, clinical trial matching, grant writing, tax preparation,
insurance underwriting, architectural code compliance, patent drafting.
Must feel like something that could raise a seed round today.""",

    "enterprise_ai": """\
Category: Enterprise AI transformation — a Fortune 500 workflow rebuilt with LLMs and ML.
Focus on a specific department and workflow, not a generic "AI strategy".
Examples: supply chain disruption prediction, regulatory change monitoring,
accounts payable automation, warranty claims triage, field service dispatch,
customer churn intervention, contract lifecycle management, audit trail generation.""",

    "ai_infra": """\
Category: AI infrastructure and developer tooling — foundational systems other builders need.
Examples: LLM evaluation and regression testing frameworks, prompt versioning and
A/B testing platforms, multi-model routing and cost optimization, agent memory
and context management, guardrail and safety layers, LLMOps CI/CD pipelines,
vector store lifecycle management, model drift monitoring.""",
}

_DEFAULT_CATEGORY = "multi_agent"  # overridden by IDEA_CATEGORY env var


def generate_use_case(generate_text_fn: callable) -> dict:
    """
    Ask Gemini to invent a business use case / startup idea.
    Category is controlled by the IDEA_CATEGORY env var:
      - Not set / "multi_agent" → multi-agent AI systems (default)
      - "random"               → Gemini picks freely across all categories
      - "yc_startup"           → YC-style AI-native startup
      - "enterprise_ai"        → Enterprise AI transformation
      - "ai_infra"             → AI infrastructure / developer tooling
    Returns the parsed dict or raises RuntimeError after 3 failed attempts.
    """
    import os
    raw_category = os.environ.get("IDEA_CATEGORY", _DEFAULT_CATEGORY).strip().lower()

    if raw_category == "random":
        import random
        chosen = random.choice(list(_CATEGORY_INSTRUCTIONS.keys()))
        print(f"  [ideation] Category: random → picked '{chosen}'")
    elif raw_category in _CATEGORY_INSTRUCTIONS:
        chosen = raw_category
        print(f"  [ideation] Category: {chosen}")
    else:
        chosen = _DEFAULT_CATEGORY
        print(f"  [ideation] Unknown IDEA_CATEGORY '{raw_category}' — defaulting to '{chosen}'")

    category_instruction = _CATEGORY_INSTRUCTIONS[chosen]
    prompt = _IDEATION_PROMPT.format(category_instruction=category_instruction)

    for attempt in range(3):
        try:
            raw = generate_text_fn(prompt, _IDEATION_SYSTEM).strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            raw = raw.strip()
            data = json.loads(raw)

            required = {
                "title", "category", "industry", "problem", "solution_summary",
                "architecture_insight", "aws_services", "solution_type", "linkedin_hook"
            }
            missing = required - data.keys()
            if missing:
                print(f"  [ideation] attempt {attempt+1}: missing fields {missing}, retrying...")
                continue

            if not isinstance(data.get("aws_services"), list) or len(data["aws_services"]) < 6:
                print(f"  [ideation] attempt {attempt+1}: aws_services too short (<6), retrying...")
                continue

            print(f"  [ideation] Use case : {data['title']}")
            print(f"  [ideation] Industry : {data['industry']}")
            print(f"  [ideation] Services : {', '.join(data['aws_services'][:5])}...")
            return data

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [ideation] attempt {attempt+1}: parse error ({e}), retrying...")

    raise RuntimeError("Failed to generate a valid use case after 3 attempts.")
