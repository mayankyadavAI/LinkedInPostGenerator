"""
NanoBanana Infographic Generator — premium dark-mode architecture diagrams
=========================================================================

Replaces the diagrams/Graphviz renderer with Gemini's native image generation
model. Produces high-contrast, neon-on-navy architecture visuals that match
the aesthetic described in the prompt framework.

Pipeline:
  build_image_prompt()  → assembles structured prompt from use_case dict
  generate_nb_diagram() → calls Gemini image-gen model → saves PNG
  render_nb_diagram()   → thin wrapper used by generate_and_schedule.py

Gemini model used:
  Primary  : gemini-3-pro-image-preview  (4K, best quality)
  Fallback : gemini-2.5-flash-image      (2K, fast)

GitHub Actions requirement:
  GEMINI_API_KEY_3 and/or GEMINI_API_KEY_4 must be set (same as aws_diagram).
  The model endpoints are accessed via the google-genai SDK (pip install google-genai).
"""

import os
import re
import time
import pathlib

# ---------------------------------------------------------------------------
# Style constants  (shared across both prompt templates)
# ---------------------------------------------------------------------------

_STYLE_PREAMBLE = (
    "Create a professional, high-resolution, dark-mode technical infrastructure "
    "architecture diagram for a LinkedIn post. "
    "The aspect ratio must be 16:9 widescreen. "
    "The style must be modern, sleek, and enterprise-grade enterprise SaaS presentation. "
    "Use a very dark navy/black background (#0b0f19). "
    "Use a vibrant neon color palette: electric cyan, deep purple, hot pink, and bright "
    "emerald green for borders, arrows, and glowing accents. "
    "Apply glowing drop shadows behind key nodes to create a subtle 3D tech effect. "
    "Use thin, crisp, glowing neon connecting lines with clean direction arrows. "
    "Do NOT use photorealistic humans or 3D character avatars. Use abstract, minimalist "
    "geometric technical icons (e.g., servers, databases, security shields, pipeline arrows). "
    "All text must be perfectly legible, spelled correctly, and use a modern "
    "sans-serif font like Montserrat or Roboto."
)

_VISUAL_RULES = (
    "Visual Rules:\n"
    "- Apply glowing drop shadows behind key nodes to create a subtle 3D tech effect.\n"
    "- Use thin, crisp, glowing neon connecting lines with clean direction arrows.\n"
    "- Strictly no photorealistic elements or human figures; use flat, geometric tech "
    "silhouettes (servers, databases, pipeline arrows, security shields, cloud icons).\n"
    "- Ensure all technical terms are spelled perfectly using a clean sans-serif typeface.\n"
    "- Every node label must appear in white or bright neon text on its dark background.\n"
    "- Group related nodes inside subtly glowing rectangular bounding boxes with a "
    "thin colored border matching the column's accent color."
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_pipeline_prompt(use_case: dict) -> str:
    """
    Builds the 'multi-column pipeline' prompt variant.
    Best for: data engineering, agent orchestration, real-time streaming,
    multi-cloud, LLM inference pipelines.
    """
    title     = use_case.get("title", "Enterprise AI Architecture").upper()
    services  = use_case.get("aws_services", [])
    industry  = use_case.get("industry", "enterprise")
    solution  = use_case.get("solution_summary", "")
    problem   = use_case.get("problem", "")
    insight   = use_case.get("architecture_insight", "")
    agents    = use_case.get("agent_types", [])
    category  = use_case.get("category", "")

    # ── Derive logical columns from the use-case context ──────────────────
    # We always want exactly 4 columns; their names and contents are inferred
    # from the use-case data so diagrams aren't generic.

    ingestion_nodes  = _pick_services(services, ["kafka", "kinesis", "iot", "sqs", "api", "event", "stream", "ingest", "source", "erp", "saas"])
    processing_nodes = _pick_services(services, ["spark", "databricks", "lambda", "fargate", "ecs", "glue", "processing", "transform", "feature"])
    storage_nodes    = _pick_services(services, ["s3", "snowflake", "redshift", "dynamo", "aurora", "rds", "opensearch", "pinecone", "vector", "storage", "db"])
    output_nodes     = _pick_services(services, ["bedrock", "sagemaker", "llm", "agent", "bi", "dashboard", "governance", "app", "lex", "comprehend", "rekognition"])

    # Fallback: distribute remaining services evenly
    used = set(ingestion_nodes + processing_nodes + storage_nodes + output_nodes)
    remaining = [s for s in services if s not in used]
    buckets = [ingestion_nodes, processing_nodes, storage_nodes, output_nodes]
    for i, svc in enumerate(remaining):
        buckets[i % 4].append(svc)

    # Add agent types to processing or output column
    if agents:
        for a in agents[:2]:
            output_nodes.append(a)

    def _node_list(nodes: list[str], fallback: str) -> str:
        if not nodes:
            return f'  - "{fallback}"'
        return "\n".join(f'  - "{n}"' for n in nodes[:5])

    cross_cutting = _infer_cross_cutting(services, category)

    prompt = f"""{_STYLE_PREAMBLE}

Layout Structure:
Title: At the very top, create a bold, clean neon cyan title that reads exactly: "{title}".

Four Main Columns: Create four vertical bounding boxes arranged side-by-side from left to right:

1. "Data Sources & Ingestion":
{_node_list(ingestion_nodes, 'Event Sources')}
   Connect them with thin lines to a central aggregator node at the column's right edge.

2. "Processing & Enrichment":
   A thick glowing arrow enters this column from the left.
{_node_list(processing_nodes, 'Core Processing Engine')}
   Show branching sub-nodes where multiple transforms occur in parallel.

3. "Storage & Intelligence Layer":
   A glowing purple arrow routes data into this column.
{_node_list(storage_nodes, 'Persistent Store')}
   Include a vector/embedding node if relevant to: {solution[:120]}.

4. "Downstream Consumption":
   Arrows converge from the storage tier into final target nodes:
{_node_list(output_nodes, 'AI Applications')}

Bottom Row:
Create a full-width continuous horizontal banner labeled:
"Operational Layer: {cross_cutting}"

Architecture insight (embed as a subtle footnote label inside the diagram):
"{insight[:140]}"

{_VISUAL_RULES}
"""
    return prompt.strip()


def _build_security_prompt(use_case: dict) -> str:
    """
    Builds the 'security / governance flowchart' prompt variant.
    Best for: LLM security, OWASP for LLMs, access control, compliance,
    audit, guardrails, identity.
    """
    title    = use_case.get("title", "Enterprise AI Governance Framework").upper()
    services = use_case.get("aws_services", [])
    solution = use_case.get("solution_summary", "")
    insight  = use_case.get("architecture_insight", "")
    category = use_case.get("category", "")

    # Infer security layers from services
    waf_nodes    = _pick_services(services, ["waf", "shield", "firewall", "ingress", "apigw", "api gateway", "cognito", "auth"])
    guard_nodes  = _pick_services(services, ["guardrail", "nemo", "pii", "mask", "filter", "prompt", "validation", "inspector"])
    gateway_nodes= _pick_services(services, ["bedrock", "llm", "gateway", "langchain", "orchestrat", "agent", "lex"])
    eval_nodes   = _pick_services(services, ["hallucination", "eval", "output", "check", "monitor", "cloudwatch", "log"])
    iam_nodes    = _pick_services(services, ["kms", "iam", "rbac", "audit", "cloudtrail", "vault", "secret", "key", "identity"])

    blocked_risks = _infer_risks(services, category, solution)
    compliance    = _infer_compliance(category, solution)

    prompt = f"""{_STYLE_PREAMBLE.replace('pipeline', 'security governance flowchart')}

Layout Structure:
Title: At the very top, create a bold, glowing neon cyan title that reads exactly: "{title}".

Top Section: A wide horizontal block across the top with a subtle pink border, labeled:
"Untrusted Client Input Zone (Web, Mobile, Enterprise Chat Apps)"

Middle Section (Multi-Stage Inspection Pipeline):
Create a sequential left-to-right flowing architecture using distinct connected nodes:

- Node 1 (Entry): "User Prompt Ingress"
  Arrow flows right →

- Node 2 (Pink glowing border): "WAF & Prompt Injection Defenses"
  Sub-nodes: {", ".join(waf_nodes[:3]) or "API Gateway, WAF Rules"}
  Arrow flows right →

- Node 3 (Teal glowing border): "Guardrails & PII Masking"
  Sub-nodes: {", ".join(guard_nodes[:3]) or "Input Validation, PII Filter"}
  Arrow flows right →

- Node 4 (Central circle cluster, purple glow): "AI Gateway / LLM Orchestration"
  Internal branching nodes:
{chr(10).join(f"    * {n}" for n in (gateway_nodes[:4] or ["Agent Orchestration", "Vector DB Search", "Knowledge Base"]))}
  Arrow flows right →

- Node 5 (Purple border): "Output Evaluation & Hallucination Check"
  Sub-nodes: {", ".join(eval_nodes[:3]) or "Response Validation, Monitoring"}

Right-Hand Vertical Block:
A security pillar box running the full height of the middle section on the right side.
Label: "Identity & Access Control"
Contents (stacked vertically):
{chr(10).join(f"  - {n}" for n in (iam_nodes[:4] or ["RBAC Policies", "KMS Key Management", "Immutable Audit Logs"]))}

Bottom Row — split banner:
- Left Box (Pink outline): "Blocked Risks"
  Bullets: {blocked_risks}

- Right Box (Cyan outline): "Compliance & Standards"
  Bullets: {compliance}

Architecture insight (embed as a footnote label at the bottom):
"{insight[:140]}"

{_VISUAL_RULES.replace('pipeline arrows', 'security locks, keys, shield icons, pipeline arrows')}
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

def _pick_services(services: list[str], keywords: list[str]) -> list[str]:
    """Return services whose names contain any of the given keywords (case-insensitive)."""
    result = []
    for svc in services:
        svc_lower = svc.lower()
        if any(kw in svc_lower for kw in keywords):
            result.append(svc)
    return result


def _infer_cross_cutting(services: list[str], category: str) -> str:
    pieces = []
    for kw, label in [
        ("vault|secrets|kms|key",      "Secrets Management (KMS / HashiCorp Vault)"),
        ("cloudwatch|prometheus|grafana|monitor", "Observability (CloudWatch / Prometheus / Grafana)"),
        ("cloudtrail|audit|lineage|provenance",   "Audit & Data Lineage (CloudTrail / OpenLineage)"),
        ("iam|cognito|rbac|auth",      "Identity & Access (IAM / Cognito)"),
    ]:
        svc_str = " ".join(services).lower() + " " + category.lower()
        if re.search(kw, svc_str):
            pieces.append(label)
    if not pieces:
        pieces = ["Security & Secrets", "Observability & Monitoring", "Audit & Compliance"]
    return " | ".join(pieces[:3])


def _infer_risks(services: list[str], category: str, solution: str) -> str:
    ctx = (category + " " + solution).lower()
    risks = []
    if "inject" in ctx or "prompt" in ctx:
        risks.append("Prompt Injection")
    if "pii" in ctx or "data" in ctx or "exfil" in ctx:
        risks.append("Data Exfiltration")
    if "hallucin" in ctx or "llm" in ctx or "agent" in ctx:
        risks.append("Hallucinations")
    if "bypass" in ctx or "jailbreak" in ctx:
        risks.append("Jailbreak / Bypass")
    if not risks:
        risks = ["Prompt Injection", "Data Exfiltration", "Hallucinations"]
    return " | ".join(risks[:3])


def _infer_compliance(category: str, solution: str) -> str:
    ctx = (category + " " + solution).lower()
    standards = []
    if "health" in ctx or "hipaa" in ctx or "phi" in ctx:
        standards.append("HIPAA")
    if "gdpr" in ctx or "europe" in ctx or "privacy" in ctx:
        standards.append("GDPR")
    if "soc" in ctx or "audit" in ctx:
        standards.append("SOC 2 Type II")
    if "ai" in ctx or "llm" in ctx or "iso" in ctx:
        standards.append("ISO 42001 (AI)")
    if "finance" in ctx or "pci" in ctx:
        standards.append("PCI DSS")
    if not standards:
        standards = ["SOC 2 Type II", "ISO 42001 (AI)", "GDPR"]
    return " | ".join(standards[:3])


def _choose_template(use_case: dict) -> str:
    """
    Decide which prompt template to use based on use-case metadata.
    Returns "security" or "pipeline".
    """
    ctx = " ".join([
        use_case.get("category", ""),
        use_case.get("solution_summary", ""),
        use_case.get("architecture_insight", ""),
        " ".join(use_case.get("aws_services", [])),
        " ".join(use_case.get("agent_types", [])),
    ]).lower()

    security_signals = [
        "security", "governance", "compliance", "guardrail", "audit",
        "pii", "waf", "iam", "owasp", "llm security", "trust", "access control",
        "identity", "zero trust", "nemo", "firewall",
    ]
    return "security" if any(s in ctx for s in security_signals) else "pipeline"


# ---------------------------------------------------------------------------
# Gemini image generation
# ---------------------------------------------------------------------------

_IMAGE_MODELS = [
    "gemini-3-pro-image-preview",    # 4K — best quality
    "gemini-2.5-flash-image",        # 2K — fast fallback
]

_MAX_RETRIES     = 3
_RETRY_BASE_SECS = 8


def _get_image_keys() -> list[str]:
    """Return diagram-generation keys (keys 3+4, then fall back to 1+2)."""
    keys = []
    for var in ("GEMINI_API_KEY_3", "GEMINI_API_KEY_4",
                "GEMINI_API_KEY_2", "GEMINI_API_KEY"):
        val = os.environ.get(var, "").strip()
        if val:
            keys.append(val)
    return keys


def _call_gemini_image(prompt: str, api_key: str, model: str) -> bytes:
    """
    Call the Gemini image-generation endpoint via the google-genai SDK
    and return raw PNG bytes.
    Raises RuntimeError on failure or missing image data.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # Extract inline image bytes from the response parts
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data  # already bytes

    raise RuntimeError("Gemini returned no image data in response.")


def generate_nb_diagram(prompt: str, out_path: str) -> str:
    """
    Generate a diagram image using Gemini image models and save to out_path.
    Tries all available API keys × model variants.
    Returns the path to the saved PNG.
    """
    keys = _get_image_keys()
    if not keys:
        raise RuntimeError(
            "No Gemini API keys found for image generation. "
            "Set GEMINI_API_KEY_3 and/or GEMINI_API_KEY_4."
        )

    out_path = str(pathlib.Path(out_path).with_suffix(".png"))
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None

    for key_idx, api_key in enumerate(keys):
        key_label = f"key#{key_idx + 1} (...{api_key[-6:]})"
        for model in _IMAGE_MODELS:
            print(f"  [nanobanana] Trying {model} with {key_label}...")
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    png_bytes = _call_gemini_image(prompt, api_key, model)
                    with open(out_path, "wb") as f:
                        f.write(png_bytes)
                    size_kb = len(png_bytes) // 1024
                    print(
                        f"  [nanobanana] ✓ {model} → {out_path} ({size_kb} KB)"
                    )
                    return out_path

                except Exception as e:
                    last_error = e
                    msg = str(e)

                    # Model not found or permission denied — skip this model/key combo
                    if any(sig in msg for sig in ("404", "NOT_FOUND", "not found",
                                                   "INVALID_ARGUMENT", "not supported",
                                                   "PERMISSION_DENIED", "403")):
                        print(f"  [nanobanana] {model} on {key_label} skipped ({msg[:100]}). Trying next.")
                        break

                    # Daily quota exhausted — skip to next key entirely
                    if "PerDay" in msg or ("RESOURCE_EXHAUSTED" in msg and "PerDay" in msg):
                        print(f"  [nanobanana] Daily quota on {key_label}. Switching key.")
                        break

                    # Per-minute rate limit — back off and retry same model
                    if any(sig in msg for sig in ("429", "503", "RESOURCE_EXHAUSTED",
                                                   "quota", "overloaded")):
                        wait = _RETRY_BASE_SECS * attempt
                        print(
                            f"  [nanobanana] Rate limit on {model} ({key_label}), "
                            f"attempt {attempt}/{_MAX_RETRIES}. Retrying in {wait}s..."
                        )
                        if attempt < _MAX_RETRIES:
                            time.sleep(wait)
                        else:
                            print(f"  [nanobanana] Retries exhausted for {model}. Trying next.")
                        continue

                    # Any other error — skip this model, try next
                    print(f"  [nanobanana] Error on {model} ({key_label}): {msg[:120]}. Trying next.")
                    break

    raise RuntimeError(
        f"All NanoBanana diagram attempts failed. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Public API — called by generate_and_schedule.py
# ---------------------------------------------------------------------------

def build_nb_diagram(use_case: dict, out_path: str) -> str | None:
    """
    Entry point for the main pipeline.

    1. Selects the right prompt template (pipeline vs security/governance).
    2. Builds the structured Gemini image-generation prompt.
    3. Calls the Gemini image API and saves the PNG.

    Returns the PNG path, or None on failure (pipeline continues text-only).
    """
    template = _choose_template(use_case)
    print(f"  [nanobanana] Template selected: {template}")

    if template == "security":
        prompt = _build_security_prompt(use_case)
    else:
        prompt = _build_pipeline_prompt(use_case)

    try:
        png = generate_nb_diagram(prompt, out_path)
        print(f"  [nanobanana] Diagram saved: {png}\n")
        return png
    except Exception as exc:
        print(f"  [nanobanana] WARNING: failed ({exc}) — posting text-only.\n")
        return None
