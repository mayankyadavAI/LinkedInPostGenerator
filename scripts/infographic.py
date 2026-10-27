"""
Infographic generator for LinkedIn posts.

Pipeline:
  generate_diagram_content()  → Gemini → diagram JSON (type-aware)
  render_diagram()            → Matplotlib → PNG  (no browser needed)
  upload_to_linkedin()        → LinkedIn media API → asset URN

Gemini picks the diagram type that best fits the post content:
  flow     — process / how something works (boxes + arrows)
  split    — comparison / before vs after (two columns)
  layers   — hierarchy / stack (pyramid or stacked bars)
  cycle    — loop / repeating process (circular)
  scale    — tradeoff / spectrum (axis with markers)
  stats    — numbers / data callouts (large numbers + labels)
"""

import os
import re
import json
import pathlib
import requests

# ── prompt ────────────────────────────────────────────────────────────────────

_DIAGRAM_PROMPT = """
You are designing a diagram for a LinkedIn post. Your job is to pick the diagram
type that best matches the post's core idea, then fill in the content.

Post topic : {topic}

Post text (use ONLY facts from here, never invent):
{post_text}

── DIAGRAM TYPES ──────────────────────────────────────────────────────────────

Choose exactly ONE from: flow | split | layers | cycle | scale | stats

flow   — best when the post explains a process, a pipeline, or "how X works"
         step by step. Example: how a token gets predicted, how RAG retrieves.

split  — best when the post compares two things, shows before/after, or
         contrasts a wrong assumption with the reality.

layers — best when the post describes a stack, hierarchy, or levels of
         abstraction (e.g. hardware → OS → framework → model).

cycle  — best when the post describes a loop, feedback cycle, or repeating
         process (e.g. RLHF feedback loop, agent planning loop).

scale  — best when the post explains a spectrum, tradeoff, or dial
         (e.g. temperature low→high, precision vs recall).

stats  — best when the post has 3-5 strong numbers or facts that hit harder
         as large visual callouts than as prose.

── OUTPUT FORMAT ───────────────────────────────────────────────────────────────

Return ONLY valid JSON matching the schema for the type you chose.
No markdown fences, no explanation, just the JSON object.

─── flow schema ───
{{
  "diagram_type": "flow",
  "title": "5-8 word title capturing the core insight",
  "subtitle": "one line setup, under 12 words",
  "steps": [
    {{"label": "2-4 words", "detail": "one crisp line, plain English"}},
    {{"label": "2-4 words", "detail": "one crisp line, plain English"}},
    {{"label": "2-4 words", "detail": "one crisp line, plain English"}},
    {{"label": "2-4 words", "detail": "one crisp line, plain English"}}
  ],
  "takeaway": "one sentence bottom insight, plain English, under 20 words"
}}

─── split schema ───
{{
  "diagram_type": "split",
  "title": "5-8 word title",
  "subtitle": "one line framing the contrast, under 12 words",
  "left_label": "2-3 words (the wrong side / before / common belief)",
  "right_label": "2-3 words (the right side / after / reality)",
  "rows": [
    {{"left": "short phrase", "right": "short phrase"}},
    {{"left": "short phrase", "right": "short phrase"}},
    {{"left": "short phrase", "right": "short phrase"}},
    {{"left": "short phrase", "right": "short phrase"}}
  ],
  "takeaway": "one sentence bottom insight, under 20 words"
}}

─── layers schema ───
{{
  "diagram_type": "layers",
  "title": "5-8 word title",
  "subtitle": "one line, under 12 words",
  "layers": [
    {{"label": "2-4 words", "detail": "short phrase — top of stack = highest abstraction"}},
    {{"label": "2-4 words", "detail": "short phrase"}},
    {{"label": "2-4 words", "detail": "short phrase"}},
    {{"label": "2-4 words", "detail": "short phrase — bottom = foundation"}}
  ],
  "takeaway": "one sentence bottom insight, under 20 words"
}}

─── cycle schema ───
{{
  "diagram_type": "cycle",
  "title": "5-8 word title",
  "subtitle": "one line, under 12 words",
  "steps": [
    {{"label": "2-4 words", "detail": "short phrase"}},
    {{"label": "2-4 words", "detail": "short phrase"}},
    {{"label": "2-4 words", "detail": "short phrase"}},
    {{"label": "2-4 words", "detail": "short phrase"}}
  ],
  "takeaway": "one sentence bottom insight, under 20 words"
}}

─── scale schema ───
{{
  "diagram_type": "scale",
  "title": "5-8 word title",
  "subtitle": "one line, under 12 words",
  "axis_label": "what the axis measures, 3-5 words",
  "left_pole": "2-3 words (low end of spectrum)",
  "right_pole": "2-3 words (high end of spectrum)",
  "markers": [
    {{"position": 0.1, "label": "2-3 words", "detail": "short phrase"}},
    {{"position": 0.4, "label": "2-3 words", "detail": "short phrase"}},
    {{"position": 0.75, "label": "2-3 words", "detail": "short phrase"}}
  ],
  "takeaway": "one sentence bottom insight, under 20 words"
}}

─── stats schema ───
{{
  "diagram_type": "stats",
  "title": "5-8 word title",
  "subtitle": "one line, under 12 words",
  "stats": [
    {{"value": "big number or short fact", "label": "2-4 words", "detail": "short phrase"}},
    {{"value": "big number or short fact", "label": "2-4 words", "detail": "short phrase"}},
    {{"value": "big number or short fact", "label": "2-4 words", "detail": "short phrase"}}
  ],
  "takeaway": "one sentence bottom insight, under 20 words"
}}
""".strip()

_SYSTEM = (
    "You pick the right diagram type for a LinkedIn post and output the matching JSON. "
    "Return only valid JSON, no extra text."
)


# ── content generation ────────────────────────────────────────────────────────

def generate_diagram_content(topic: str, post_text: str, generate_text_fn) -> dict:
    """Ask Gemini to pick a diagram type and return structured content."""
    prompt = _DIAGRAM_PROMPT.format(topic=topic, post_text=post_text[:2500])

    for attempt in range(2):
        raw = generate_text_fn(prompt, _SYSTEM).strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()
        try:
            data = json.loads(raw)
            dtype = data.get("diagram_type", "")
            if dtype in ("flow", "split", "layers", "cycle", "scale", "stats"):
                return data
            print(f"  [infographic] attempt {attempt+1}: unknown diagram_type '{dtype}', retrying...")
        except json.JSONDecodeError as e:
            print(f"  [infographic] attempt {attempt+1}: JSON parse failed ({e}), retrying...")

    raise RuntimeError("Failed to generate valid diagram JSON after 2 attempts.")


# ── render dispatch ───────────────────────────────────────────────────────────

def render_diagram(content: dict, out_path: str) -> str:
    """Render diagram content → PNG using Matplotlib. Returns out_path."""
    import sys
    root = pathlib.Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from renderer.render_matplotlib import render
    return render(content, out_path)


# ── LinkedIn upload (unchanged) ───────────────────────────────────────────────

def upload_to_linkedin(png_path: str, access_token: str, person_id: str) -> str:
    """Upload PNG to LinkedIn media API. Returns asset URN."""
    headers_json = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type":  "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }

    reg = requests.post(
        "https://api.linkedin.com/v2/assets?action=registerUpload",
        headers=headers_json,
        json={
            "registerUploadRequest": {
                "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                "owner":   f"urn:li:person:{person_id}",
                "serviceRelationships": [{
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent",
                }],
            }
        },
        timeout=20,
    )
    reg.raise_for_status()
    val = reg.json()["value"]
    upload_url = val["uploadMechanism"][
        "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
    ]["uploadUrl"]
    asset_urn = val["asset"]

    with open(png_path, "rb") as f:
        img_bytes = f.read()

    put = requests.put(
        upload_url,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "image/png"},
        data=img_bytes,
        timeout=60,
    )
    put.raise_for_status()

    print(f"  [infographic] Uploaded → {asset_urn}")
    return asset_urn
