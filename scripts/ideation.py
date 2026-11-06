"""
Ideation module — Step 1 of the business use case pipeline.

Three sourcing modes, controlled by the IDEA_SOURCE env var:

  exa (default)   Real-world signal: Exa neural search fetches live YC launches,
                  AI SaaS funding news, and Product Hunt launches. Gemini then
                  produces a grounded SA analysis of that real idea.
                  Falls back to "seed" if Exa fails or EXA_API_KEY is unset.

  seed            Rotates through a curated list of 40+ YC/AI SaaS ideas in
                  ideas_seed.json (day-of-year index, no network required).
                  Gemini produces a grounded SA analysis of the chosen idea.

  invented        Original behaviour — Gemini freely invents a use case.
                  Controlled by the IDEA_CATEGORY env var (same as before).

In all modes the output is the same use-case dict consumed by the rest of
the pipeline (generate_biz_post, aws_diagram, etc.).
"""

import json
import os
import re
import random
import time
import pathlib
from datetime import datetime, timezone

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).parent

# ==============================================================================
# PROMPT — grounded mode (exa / seed)
# ==============================================================================

_GROUNDED_SYSTEM = """
You are a principal AWS solutions architect reviewing a REAL AI SaaS startup idea,
YC company, or trending AI product. Your job: produce a production-grade AWS
implementation analysis for a LinkedIn post read by senior engineers and
technical leaders.

DOMAIN CONSTRAINT — only cover software and SaaS products. Specifically:
  ALLOWED: LLM applications, AI agents, developer tooling, data platforms, MLOps,
  enterprise software automation, SaaS workflow tools, fintech, legal-tech,
  healthcare software, e-commerce software, sales/marketing software,
  productivity software, search and retrieval, content generation platforms.
  EXCLUDED: IoT hardware, embedded systems, robotics, vehicle telematics,
  drone/aviation, physical sensors, industrial control systems, smart home devices.
If the input idea touches an excluded domain, pivot to the SOFTWARE layer of that
idea — e.g. the cloud data pipeline and AI model serving behind a fleet system,
not the vehicle hardware or sensor network itself.

You think like the engineer who would actually build this — naming the specific
AWS services, the non-obvious architectural tradeoff, and what breaks at scale.
Use Amazon Bedrock for LLM inference and agent orchestration unless a more specific
AWS ML service (SageMaker, Comprehend, Textract, Rekognition) is clearly a better fit
for the use case. Never include IoT Core, IoT Greengrass, or IoT Analytics.
You never hype. You never invent client results.
""".strip()

_GROUNDED_PROMPT = """
Analyze the REAL company/idea below and produce a production-grade AWS solution
architecture that a senior engineer would be proud to put in front of a VP of Eng.

━━━ REAL IDEA / COMPANY ━━━
{idea_source}

━━━ RESEARCH CONTEXT ━━━
{research_context}

DOMAIN RULES (enforce strictly):
- Focus ONLY on the software, data, and AI layer — never design for hardware,
  physical sensors, IoT devices, vehicle telematics, robotics, or embedded systems.
- If the company has a hardware component, analyze only the cloud/SaaS backend:
  the data ingestion API, the ML inference service, the storage layer, the analytics
  pipeline, and the developer-facing platform — NOT the device firmware or sensor network.
- Excluded AWS services: AWS IoT Core, AWS IoT Greengrass, AWS IoT Analytics,
  AWS Greengrass, Amazon Kinesis Video Streams for device streaming.
- For LLM inference and agent orchestration: prefer Amazon Bedrock + Amazon Bedrock Agents.
  Use Amazon SageMaker only when there is a clear model training or custom deployment need.

Requirements:
- Ground every architectural choice in the specific software problem this company solves
- Name the exact AWS services that map to their actual technical needs
- Include the non-obvious SA decision that separates production from prototype
- If it is a YC early-stage startup, keep the architecture cost-optimised and
  serverless-first — avoid premature infra complexity
- Business impact must reference the specific industry (plausible estimates, not invented ROI)
- source_company_or_idea: the actual company/idea name (use "category-level" if unknown)

Output ONLY valid JSON. No markdown fences, no explanation:
{{
  "title": "5-10 word title — reads like a solution name, not a company name",
  "category": "one of: multi_agent | yc_startup | enterprise_ai | ai_infra",
  "industry": "specific software industry vertical (e.g. legal-tech, fintech, developer-tooling, enterprise-saas, healthcare-software, e-commerce, sales-automation)",
  "company_type": "startup | enterprise | both",
  "source_company_or_idea": "name or short description of the real idea this is based on",
  "problem": "2-3 sentences — the specific concrete software/data pain. Name the workflow, the cost, the breakage.",
  "solution_summary": "2-3 sentences — the AWS architecture. Name agent types, models, data flows.",
  "architecture_insight": "1 sentence — the non-obvious SA decision that makes this production-grade vs prototype",
  "business_impact": "1 sentence — plausible metric tied to this specific industry",
  "solution_type": "one of: multi_agent_orchestration | agentic_rag | real_time_ml | document_intelligence | fraud_detection | llm_evaluation | llmops_platform | predictive_analytics | data_platform",
  "aws_services": [
    "8-12 specific AWS service names — include API Gateway or ALB, Lambda or ECS, Bedrock, S3, RDS or DynamoDB, OpenSearch Service or Kendra, CloudWatch, and at least one security service"
  ],
  "agent_types": ["list agent roles if multi-agent, else empty list"],
  "complexity": "high",
  "linkedin_hook": "one sentence — specific architectural observation or metric that stops a senior software engineer scrolling. Must reference the real use case."
}}

IMPORTANT: aws_services must use official AWS service names and must NOT include
IoT Core, IoT Greengrass, IoT Analytics, or any IoT/device service.
Include at least one security service (WAF, Cognito, KMS, or GuardDuty) and
one observability service (CloudWatch, X-Ray, or CloudTrail).
""".strip()


# ==============================================================================
# PROMPT — invented mode (original behaviour)
# ==============================================================================

_IDEATION_SYSTEM = """
You are a principal solutions architect and venture-backed startup advisor.
You have deep hands-on experience with AWS, multi-agent AI systems, LLMOps,
and production ML across industries. You think like a YC partner evaluating
whether a startup idea is technically feasible and commercially interesting.

DOMAIN CONSTRAINT — only cover software and SaaS products:
  ALLOWED: LLM applications, AI agents, developer tooling, MLOps, data platforms,
  fintech, legal-tech, healthcare software, e-commerce software, sales/marketing
  software, productivity tools, enterprise workflow automation, search and retrieval,
  content generation, document intelligence.
  EXCLUDED: IoT hardware, embedded systems, robotics, vehicle telematics,
  drone/aviation, industrial control systems, physical sensors, smart home devices.

You generate concrete, technically grounded business use cases and startup ideas
that a senior software engineer building in 2025-2026 would find genuinely worth reading.
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
  "industry": "specific software industry vertical (e.g. legal-tech, fintech, developer-tooling, enterprise-saas, healthcare-software, e-commerce, sales-automation)",
  "company_type": "startup | enterprise | both",
  "source_company_or_idea": "",
  "problem": "2-3 sentences — the specific, concrete software pain. Name the workflow, the cost, the breakage.",
  "solution_summary": "2-3 sentences — the AI/ML architecture. Name the agent types, models, and data flows.",
  "architecture_insight": "1 sentence — the non-obvious design decision that makes this production-grade vs toy",
  "business_impact": "1 sentence — plausible metric. Be specific: time saved, error rate reduced, cost cut.",
  "solution_type": "one of: multi_agent_orchestration | agentic_rag | real_time_ml | document_intelligence | fraud_detection | llm_evaluation | llmops_platform | predictive_analytics | data_platform",
  "aws_services": [
    "8-12 specific AWS service names — include API Gateway or ALB, Lambda or ECS, Bedrock, S3, RDS or DynamoDB, OpenSearch Service or Kendra, CloudWatch, and at least one security service"
  ],
  "agent_types": ["list agent roles if multi-agent, else empty list"],
  "complexity": "high",
  "linkedin_hook": "one sentence hook — specific observation or number that would stop a senior software engineer scrolling"
}}

aws_services must use official AWS service names. Do NOT include IoT Core, IoT Greengrass,
IoT Analytics, or any IoT/device service. Prefer Amazon Bedrock for LLM inference.
Include at least one security service (WAF, Cognito, or KMS) and one observability
service (CloudWatch, X-Ray, or CloudTrail).
""".strip()

_CATEGORY_INSTRUCTIONS = {
    "multi_agent": """\
Category: Multi-agent AI system — multiple specialized LLM agents collaborating.
Pick a SPECIFIC software industry and workflow. Examples of patterns:
  - Research + reasoning + action agents with tool-use pipelines
  - Orchestrator → specialist agents with critic/review loops
  - Hierarchical agent systems (planner → executor → validator)
  - Multi-agent debate or adversarial verification patterns
  - Agentic RAG with memory, re-ranking, and grounding agents
Industries to vary: legal, fintech, healthcare software, sales, marketing, HR,
procurement, insurance, media, e-commerce, developer tooling, cybersecurity.
Do NOT use: physical robotics, vehicle control, IoT device management.
Include agent_types with specific role names for each agent in the system.""",

    "yc_startup": """\
Category: YC-style AI-native SaaS startup — a niche software problem a small team could own.
Think: AI for a workflow that currently requires expensive human specialists.
Examples: AI for medical coding, legal contract review, customs compliance,
clinical trial matching, grant writing, tax preparation, insurance underwriting,
audit evidence collection, sales outreach personalization, code review automation.
Must feel like something that could raise a seed round today.
Do NOT use: hardware products, drone logistics, IoT sensor networks.""",

    "enterprise_ai": """\
Category: Enterprise AI transformation — a Fortune 500 software workflow rebuilt with LLMs and ML.
Focus on a specific department and SaaS/data workflow, not generic "AI strategy".
Examples: regulatory change monitoring, accounts payable automation, warranty claims
triage, customer churn prediction, contract lifecycle management, audit trail
generation, HR onboarding automation, spend analytics, vendor risk scoring.
Do NOT use: factory floor sensors, supply chain physical routing, vehicle dispatch.""",

    "ai_infra": """\
Category: AI infrastructure and developer tooling — foundational software other builders need.
Examples: LLM evaluation and regression testing frameworks, prompt versioning and
A/B testing, multi-model routing and cost optimization, agent memory and context
management, guardrail and safety layers, LLMOps CI/CD pipelines, vector store
lifecycle management, model drift monitoring, synthetic data generation for NLP.
Do NOT use: edge device management, IoT firmware OTA updates.""",
}

_DEFAULT_CATEGORY = "multi_agent"


# ==============================================================================
# EXA — real-world idea discovery
# ==============================================================================

# Rotate queries so each daily run fetches a different angle.
# All queries are scoped to software/SaaS — never IoT, robotics, or hardware.
_EXA_QUERIES = [
    "YC W25 S25 AI SaaS startup software product launch 2025",
    "AI SaaS startup raised seed funding enterprise software workflow automation 2025 2026",
    "new vertical AI SaaS tool for lawyers doctors accountants engineers launched 2025",
    "AI agent platform automate knowledge worker white collar workflow software 2025",
    "YC backed AI startup replace human specialist software SaaS seed round 2026",
    "AI developer tooling LLMOps evaluation guardrails agent infrastructure SaaS 2025",
    "enterprise AI automation Fortune 500 software LLM deployment production 2025",
    "generative AI fintech legal-tech healthcare software startup product launch 2025",
]

_EXA_TIMEOUT = 20  # seconds


def _pick_exa_query() -> str:
    """Rotate through Exa queries by day-of-year so consecutive runs differ."""
    day = datetime.now(timezone.utc).timetuple().tm_yday
    return _EXA_QUERIES[day % len(_EXA_QUERIES)]


def discover_real_idea(exa_api_key: str) -> tuple[str, str]:
    """
    Use Exa neural search to find a real AI SaaS / YC startup idea.
    Returns (idea_source, research_context) strings for prompt injection.
    Raises RuntimeError if Exa is unavailable or returns no results.
    """
    try:
        from exa_py import Exa
    except ImportError:
        raise RuntimeError("exa-py not installed. Run: pip install exa-py")

    query = _pick_exa_query()
    print(f"  [exa] Query: {query!r}")

    exa = Exa(api_key=exa_api_key)

    results = exa.search(
        query=query,
        type="neural",
        num_results=5,
        start_published_date="2024-01-01",
        contents={
            "text": {"max_characters": 1200},
            "highlights": {"max_characters": 400, "num_sentences": 3},
        },
    )

    if not results.results:
        raise RuntimeError("Exa returned no results.")

    # Build idea_source from the best result (highest relevance = first)
    best = results.results[0]
    idea_source = f"{best.title or 'Unknown'}\n{best.url or ''}"
    if getattr(best, "highlights", None):
        idea_source += f"\n\nKey excerpt: {best.highlights[0]}"

    # Build research_context from all results
    lines = []
    for i, r in enumerate(results.results, 1):
        lines.append(f"Source {i}: {r.title or 'Untitled'}")
        lines.append(f"URL: {r.url}")
        if getattr(r, "highlights", None):
            lines.append(f"Highlight: {r.highlights[0]}")
        if getattr(r, "text", None):
            lines.append(f"Context: {(r.text or '')[:400]}...")
        lines.append("")

    research_context = "\n".join(lines)
    print(f"  [exa] Found {len(results.results)} sources. Best: {best.title!r}")
    return idea_source, research_context


# ==============================================================================
# SEED LIST — offline fallback
# ==============================================================================

def _load_seed_idea() -> tuple[str, str]:
    """
    Rotate through the curated ideas_seed.json list by day-of-year.
    Returns (idea_source, research_context) — research_context is minimal
    since seeds are already well-described.
    """
    seed_path = _HERE / "ideas_seed.json"
    if not seed_path.exists():
        raise RuntimeError(f"Seed file not found: {seed_path}")

    with open(seed_path, "r") as f:
        seeds = json.load(f)

    day  = datetime.now(timezone.utc).timetuple().tm_yday
    seed = seeds[day % len(seeds)]

    idea_source = (
        f"{seed['name']}\n"
        f"Category: {seed['category']}\n"
        f"Industry: {seed['industry']}\n"
        f"Description: {seed['description']}"
    )
    research_context = seed.get("context", "No additional context available.")

    print(f"  [seed] Idea: {seed['name']} ({seed['industry']})")
    return idea_source, research_context


# ==============================================================================
# CORE — generate use case (public API)
# ==============================================================================

def _parse_and_validate(raw: str, attempt: int) -> dict | None:
    """Parse JSON response and validate required fields. Returns dict or None."""
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [ideation] attempt {attempt}: JSON parse error ({e}), retrying...")
        return None

    required = {
        "title", "category", "industry", "problem", "solution_summary",
        "architecture_insight", "aws_services", "solution_type", "linkedin_hook",
    }
    missing = required - data.keys()
    if missing:
        print(f"  [ideation] attempt {attempt}: missing fields {missing}, retrying...")
        return None

    if not isinstance(data.get("aws_services"), list) or len(data["aws_services"]) < 6:
        print(f"  [ideation] attempt {attempt}: aws_services too short (<6), retrying...")
        return None

    # Ensure source field always exists
    if "source_company_or_idea" not in data:
        data["source_company_or_idea"] = ""

    return data


def generate_use_case(generate_text_fn: callable) -> dict:
    """
    Generate a business use case dict consumed by the rest of the pipeline.

    Sourcing mode is controlled by IDEA_SOURCE env var:
      exa       (default) → Exa neural search → grounded prompt
      seed                → ideas_seed.json rotation → grounded prompt
      invented            → original Gemini-invents behaviour

    Falls back through: exa → seed → invented on failure.

    Returns parsed dict with at minimum:
      title, category, industry, problem, solution_summary, architecture_insight,
      aws_services, solution_type, linkedin_hook, source_company_or_idea
    """
    exa_api_key = os.environ.get("EXA_API_KEY", "").strip()
    raw_source  = os.environ.get("IDEA_SOURCE", "exa").strip().lower()

    # ── Resolve sourcing mode ──────────────────────────────────────────────────
    if raw_source not in ("exa", "seed", "invented"):
        print(f"  [ideation] Unknown IDEA_SOURCE '{raw_source}' — defaulting to 'exa'")
        raw_source = "exa"

    if raw_source == "exa" and not exa_api_key:
        print("  [ideation] IDEA_SOURCE=exa but EXA_API_KEY not set — falling back to 'seed'")
        raw_source = "seed"

    # ── Attempt grounded sourcing ──────────────────────────────────────────────
    idea_source      = ""
    research_context = ""
    actual_source    = raw_source  # tracks what we actually ended up using

    if raw_source == "exa":
        try:
            idea_source, research_context = discover_real_idea(exa_api_key)
            print("  [ideation] Mode: grounded (Exa)")
        except Exception as exc:
            print(f"  [ideation] Exa discovery failed ({exc}) — falling back to seed list")
            raw_source   = "seed"
            actual_source = "seed"

    if raw_source == "seed":
        try:
            idea_source, research_context = _load_seed_idea()
            print("  [ideation] Mode: grounded (seed list)")
        except Exception as exc:
            print(f"  [ideation] Seed load failed ({exc}) — falling back to invented mode")
            raw_source   = "invented"
            actual_source = "invented"

    # ── Build prompt ───────────────────────────────────────────────────────────
    if actual_source in ("exa", "seed"):
        prompt = _GROUNDED_PROMPT.format(
            idea_source=idea_source[:1500],
            research_context=research_context[:2000],
        )
        system = _GROUNDED_SYSTEM
    else:
        # Invented mode — original category-based ideation
        raw_cat = os.environ.get("IDEA_CATEGORY", _DEFAULT_CATEGORY).strip().lower()
        if raw_cat == "random":
            chosen = random.choice(list(_CATEGORY_INSTRUCTIONS.keys()))
            print(f"  [ideation] Category: random → picked '{chosen}'")
        elif raw_cat in _CATEGORY_INSTRUCTIONS:
            chosen = raw_cat
            print(f"  [ideation] Category: {chosen}")
        else:
            chosen = _DEFAULT_CATEGORY
            print(f"  [ideation] Unknown IDEA_CATEGORY '{raw_cat}' — defaulting to '{chosen}'")

        prompt = _IDEATION_PROMPT.format(
            category_instruction=_CATEGORY_INSTRUCTIONS[chosen]
        )
        system = _IDEATION_SYSTEM
        print("  [ideation] Mode: invented (Gemini free-form)")

    # ── Generate + parse with retries ──────────────────────────────────────────
    for attempt in range(1, 4):
        try:
            raw = generate_text_fn(prompt, system).strip()
            data = _parse_and_validate(raw, attempt)
            if data is None:
                continue

            # Log summary
            print(f"  [ideation] Source   : {actual_source}")
            print(f"  [ideation] Use case : {data['title']}")
            print(f"  [ideation] Industry : {data['industry']}")
            if data.get("source_company_or_idea"):
                print(f"  [ideation] Based on : {data['source_company_or_idea']}")
            print(f"  [ideation] Services : {', '.join(data['aws_services'][:5])}...")
            return data

        except Exception as e:
            print(f"  [ideation] attempt {attempt}: generation error ({e}), retrying...")
            if attempt < 3:
                time.sleep(5)

    raise RuntimeError("Failed to generate a valid use case after 3 attempts.")
