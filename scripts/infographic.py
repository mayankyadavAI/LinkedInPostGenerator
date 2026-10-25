"""
Infographic generator for LinkedIn posts.

Pipeline:
  generate_process_content()  → Gemini → "how it works" content dict
  render_infographic()        → Playwright → PNG
  upload_to_linkedin()        → LinkedIn media API → asset URN
"""

import os
import re
import json
import pathlib
import requests

_PROCESS_CONTENT_PROMPT = """
You are designing content for a professional, visually polished LinkedIn infographic aimed at AI/ML
engineers and technical practitioners. The infographic must look like it belongs in a high-quality
engineering publication — precise language, clean structure, zero fluff.

Topic: {topic}

The LinkedIn post this infographic will accompany (the infographic MUST illustrate the SAME narrative —
same claims, same specifics, no invented facts):
{post_text}

Read the post and identify its natural 3-part arc. Use the framing that fits the post exactly:
- Technical mechanism: INPUT → PROCESS → OUTPUT
- Problem/solution: THE PROBLEM → THE FIX → THE RESULT
- Comparison/contrast: BEFORE → THE SHIFT → AFTER
- News/discovery: CONTEXT → THE FINDING → THE IMPLICATION

This infographic has THREE structured sections, all walking through the same arc at increasing depth:

1. "stages" — exactly 3 items representing the arc at the highest level. Each has:
   - "label": 2-3 words, Title Case — sharp and specific, not generic ("Token Sampling" not "The Process")
   - "snippet": a realistic code/terminal/data fragment (max 22 chars) that visually represents the stage.
     Use actual syntax where possible: 'logits[vocab]', 'p = softmax(z)', 'token = argmax(p)',
     'loss = NaN', 'ctx_len = 128k', 'status: FAILED'. Even for non-code topics, use a log-line or
     metric-style fragment — it's a visual device that signals technical credibility.

2. "steps" — exactly 4 numbered cards that walk through the arc in more detail (step 1 = entry point,
   step 4 = payoff). Each has:
   - "label": 2-4 words, Title Case — specific and action-oriented ("Compute Attention Scores",
     not "Step One")
   - "points": exactly 3 crisp technical phrases, max 4 words each. These should read like bullet
     points from an engineering design doc — concrete, no adjective padding, no "enables X" or
     "leverages Y" filler.

3. Two flow summaries of the same arc at different granularities:
   - "flow_a_items": 5-6 items (1-3 words each) — the detailed technical pipeline in order
   - "flow_b_items": 4 items (1-2 words each) — the simplified big-picture arc

Also produce these five metadata fields:
- title_line1: 3-5 words, Title Case. A bold, specific technical claim or contrast — NOT a topic label.
  ("Closed APIs Cap Control" is better than "Open vs Closed Models"). This is what stops the scroll.
- title_line2: 3-5 words, Title Case. The payoff that completes line1. Should hit harder than line1,
  not just continue it ("Open Weights Don't." completes a contrast opener cleanly).
- tagline: under 10 words. A precise setup line for the breakdown — specific to this topic, not generic.
  ("From raw logits to a sampled token" beats "How it works step by step").
- hook: max 20 words. The single most quotable insight from the post — phrased as a standalone
  declarative sentence that reads well when screenshotted out of context.
- section_label: 2-4 words, ALL CAPS. Names what this breakdown is specifically about — not a
  generic placeholder. ("LOGIT CONTROL CHAIN", "THE FALLBACK SEQUENCE", "ATTENTION UNDER THE HOOD").

Standards for all fields:
- Every label, point, and phrase must be specific to THIS topic — no generic filler that could apply
  to any infographic ("Key Insight", "Important Step", "The Solution").
- Technical terms should be used correctly and precisely — do not simplify to the point of inaccuracy.
- Write for an audience of working engineers, not executives or beginners.
- Zero marketing language: no "seamless", "robust", "powerful", "unlock", "revolutionize", "leverage".

Return ONLY valid JSON — no markdown fences, no explanation:
{{
  "title_line1": "...", "title_line2": "...", "tagline": "...", "hook": "...", "section_label": "...",
  "stages": [{{"label": "...", "snippet": "..."}}, {{"label": "...", "snippet": "..."}}, {{"label": "...", "snippet": "..."}}],
  "steps": [
    {{"label": "...", "points": ["...", "...", "..."]}},
    {{"label": "...", "points": ["...", "...", "..."]}},
    {{"label": "...", "points": ["...", "...", "..."]}},
    {{"label": "...", "points": ["...", "...", "..."]}}
  ],
  "flow_a_items": ["...", "...", "...", "...", "..."],
  "flow_b_items": ["...", "...", "...", "..."]
}}
""".strip()

_SYSTEM = "You generate structured JSON content for professional technical infographics. Return only valid JSON, no extra text."


def _clean_text(s: str) -> str:
    """Strip stray markdown/comment markers the model sometimes leaks into 'plain text' fields."""
    s = s.strip()
    s = re.sub(r'^(#+|//+)\s*', '', s)
    s = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', s)
    s = re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', s)
    s = s.replace('`', '')
    return s.strip()


def _clean_process_content(data: dict) -> dict:
    data["title_line1"] = _clean_text(data["title_line1"])
    data["title_line2"] = _clean_text(data["title_line2"])
    data["tagline"] = _clean_text(data["tagline"])
    data["hook"] = _clean_text(data["hook"])
    data["section_label"] = _clean_text(data.get("section_label", "")).upper() or "HOW IT ACTUALLY WORKS"
    for stage in data["stages"]:
        stage["label"] = _clean_text(stage["label"])
        stage["snippet"] = _clean_text(stage["snippet"])
    for step in data["steps"]:
        step["label"] = _clean_text(step["label"])
        step["points"] = [_clean_text(p) for p in step["points"]]
    data["flow_a_items"] = [_clean_text(i) for i in data["flow_a_items"]]
    data["flow_b_items"] = [_clean_text(i) for i in data["flow_b_items"]]
    return data


def generate_process_content(topic: str, post_text: str, generate_text_fn) -> dict:
    """Call the LLM to produce the 'how it works' process-infographic content dict."""
    prompt = _PROCESS_CONTENT_PROMPT.format(topic=topic, post_text=post_text[:2500])

    for attempt in range(2):
        raw = generate_text_fn(prompt, _SYSTEM)
        raw = raw.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()
        try:
            data = json.loads(raw)
            required = ["title_line1", "title_line2", "tagline", "hook",
                        "stages", "steps", "flow_a_items", "flow_b_items"]
            if all(k in data for k in required) and len(data["stages"]) == 3 and len(data["steps"]) == 4:
                return _clean_process_content(data)
        except (json.JSONDecodeError, KeyError):
            pass

    raise RuntimeError("Failed to generate valid process-infographic JSON after 2 attempts.")


def render_infographic(content: dict, out_path: str, template: str = "process_infographic.html.j2") -> str:
    """Render the content dict to a PNG using Playwright. Returns PNG path."""
    import sys
    root = pathlib.Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from renderer.render import render
    return render(content, out_path, template=template)


def upload_to_linkedin(png_path: str, access_token: str, person_id: str) -> str:
    """
    Upload PNG to LinkedIn via the media upload API.
    Returns the asset URN to embed in the ugcPost.
    """
    headers_json = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type":  "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }

    # Step 1 — register upload
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

    # Step 2 — upload image bytes
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
