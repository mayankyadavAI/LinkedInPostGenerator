#!/usr/bin/env python3
"""
LinkedIn Post Agent
===================

Two independent pipelines, selected by CONTENT_SLOT env var:

  business_usecase (DEFAULT):
    Step 1  Gemini invents a business use case / startup idea (ideation.py)
    Step 2  Gemini writes a LinkedIn post about it
    Step 2.5 Humanize pass
    Step 3  Gemini generates AWS diagram spec → diagrams library renders PNG
    Step 4  Upload PNG + post to LinkedIn  (no comment)

  news | educational | personal | advanced  (LEGACY — kept intact):
    Step 1  Exa research
    Step 2  Gemini writes LinkedIn post (STYLE_1 or STYLE_2)
    Step 2.5 Humanize pass
    Step 2.6 Manim infographic (if available)
    Step 3  Post to LinkedIn
    Step 4  Drop first comment  (DISABLED for now)

Run locally:
  python scripts/generate_and_schedule.py              # business_usecase
  python scripts/generate_and_schedule.py --preview    # preview only, no post
  CONTENT_SLOT=educational python scripts/generate_and_schedule.py --preview
"""

import os
import sys
import json
import re
import random
import time
import requests
from datetime import datetime, timezone
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY        = os.environ.get("GEMINI_API_KEY")
GEMINI_API_KEY_2      = os.environ.get("GEMINI_API_KEY_2")
# Keys 3 & 4 are reserved exclusively for diagram generation to avoid
# hitting quota on the post-generation keys during the same run.
GEMINI_API_KEY_3      = os.environ.get("GEMINI_API_KEY_3")
GEMINI_API_KEY_4      = os.environ.get("GEMINI_API_KEY_4")
DEEPSEEK_API_KEY      = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL        = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
EXA_API_KEY           = os.environ.get("EXA_API_KEY")
LINKEDIN_ACCESS_TOKEN = os.environ.get("LINKEDIN_ACCESS_TOKEN")
LINKEDIN_PERSON_ID    = os.environ.get("LINKEDIN_PERSON_ID")

GEMINI_MODEL           = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-latest"]
MAX_RETRIES            = 2
RETRY_BASE_SECONDS     = 8

# ── Platform character limits ─────────────────────────────────────────────────
PLATFORM_CHAR_LIMITS = {
    "linkedin": 3000,
    "twitter":  280,
    "x":        280,
}
PLATFORM = "linkedin"

# ── Content slot ──────────────────────────────────────────────────────────────
CONTENT_SLOT = os.environ.get("CONTENT_SLOT", "business_usecase")
_VALID_SLOTS = ("news", "educational", "personal", "advanced", "business_usecase")
if CONTENT_SLOT not in _VALID_SLOTS:
    CONTENT_SLOT = "business_usecase"

# ── Load topics config ────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_script_dir, "topics.json"), "r") as f:
    _config = json.load(f)

NICHE        = _config["niche"]
PERSONA      = _config["persona"]
BIZ_PERSONA  = _config.get("business_usecase_persona", PERSONA)

_IS_BIZ = (CONTENT_SLOT == "business_usecase")

if _IS_BIZ:
    _biz_slot  = _config["content_slots"]["business_usecase"]
    SLOT_LABEL = _biz_slot["label"]
    TOPICS     = []
    TONES      = _biz_slot["tones"]
else:
    # Legacy slots — import hook_matrix + exa only when needed
    from hook_matrix import select_hook_formula, POWER_PHRASES
    try:
        from exa_py import Exa as _Exa
    except ImportError:
        _Exa = None
    _slot      = _config["content_slots"][CONTENT_SLOT]
    SLOT_LABEL = _slot["label"]
    TOPICS     = _slot["topics"]
    TONES      = _slot["tones"]

INCLUDE_INFOGRAPHIC = os.environ.get("INCLUDE_INFOGRAPHIC", "1") == "1"
_PNG_PATH = os.path.join(_script_dir, "..", "renderer", "output", "infographic.png")


# ==============================================================================
# GEMINI RETRY + FALLBACK CHAIN  (key1 -> key2 -> DeepSeek)
# ==============================================================================

def _parse_retry_seconds(error: Exception) -> int:
    match = re.search(r"retryDelay['\"]:\s*['\"](\d+)s", str(error))
    return min(int(match.group(1)), 60) if match else RETRY_BASE_SECONDS


def _is_quota_error(e: Exception) -> bool:
    return "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower()


def _is_retryable_server_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("503" in msg or "unavailable" in msg or "high demand" in msg
            or "server disconnected" in msg or "connection reset" in msg
            or "connection error" in msg or "remotedisconnected" in msg)


def _is_daily_quota_exhausted(e: Exception) -> bool:
    s = str(e)
    return "PerDay" in s or "GenerateRequestsPerDay" in s or ("limit: 0" in s and "429" in s)


def _is_model_not_found(e: Exception) -> bool:
    s = str(e)
    return "404" in s and ("NOT_FOUND" in s or "not found" in s.lower())


def _call_deepseek(prompt: str, system_instruction: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set.")
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user",   "content": prompt},
    ]
    for attempt in range(1, 4):
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": DEEPSEEK_MODEL, "messages": messages},
            timeout=90,
        )
        if resp.status_code == 402:
            raise RuntimeError("DeepSeek 402 — no credits.")
        if resp.status_code == 429:
            time.sleep(20 * attempt)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("DeepSeek failed after 3 attempts.")


def _gemini_call(prompt: str, system_instruction: str, api_keys: list[str]) -> str:
    """
    Core Gemini call with retry across the given list of API keys.
    Raises RuntimeError if all keys/models are exhausted.
    """
    models_to_try = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    last_error = None

    for key_index, api_key in enumerate(api_keys):
        client = genai.Client(api_key=api_key)
        key_label = f"key#{key_index + 1} (...{api_key[-6:]})"
        daily_exhausted = False
        print(f"  [Gemini] Trying {key_label}")

        for model_id in models_to_try:
            if daily_exhausted:
                break
            config = types.GenerateContentConfig(system_instruction=system_instruction)
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = client.models.generate_content(
                        model=model_id, contents=prompt, config=config
                    )
                    print(f"  [Gemini] Success with {model_id} on {key_label}")
                    return response.text.strip()
                except Exception as e:
                    if _is_model_not_found(e):
                        last_error = e
                        print(f"  [Gemini] {model_id} not found. Trying next model.")
                        break
                    if _is_quota_error(e) or _is_retryable_server_error(e):
                        last_error = e
                        if _is_daily_quota_exhausted(e):
                            print(f"  [Gemini] Daily quota on {key_label}. Switching key.")
                            daily_exhausted = True
                            break
                        is_503 = _is_retryable_server_error(e)
                        wait = min(RETRY_BASE_SECONDS * attempt, 60) if is_503 else _parse_retry_seconds(e)
                        kind = "overloaded (503)" if is_503 else "quota (429)"
                        print(f"  [Gemini] {kind} on {model_id} ({key_label}), attempt {attempt}/{MAX_RETRIES}. Retrying in {wait}s...")
                        if attempt < MAX_RETRIES:
                            time.sleep(wait)
                        else:
                            print(f"  [Gemini] Retries exhausted for {model_id}.")
                            break
                    else:
                        raise

    raise last_error or RuntimeError("All Gemini keys exhausted.")


def generate_text(prompt: str, system_instruction: str) -> str:
    """
    Post/content generation — uses GEMINI_API_KEY + GEMINI_API_KEY_2 only.
    Keys 3 & 4 are kept free for diagram generation in the same run.
    """
    keys = [k for k in [GEMINI_API_KEY, GEMINI_API_KEY_2] if k]
    if not keys:
        raise RuntimeError("No Gemini keys set (GEMINI_API_KEY / GEMINI_API_KEY_2).")
    return _gemini_call(prompt, system_instruction, keys)


def generate_text_diagram(prompt: str, system_instruction: str) -> str:
    """
    Diagram spec generation — uses GEMINI_API_KEY_3 + GEMINI_API_KEY_4 first,
    then falls back to keys 1 & 2 if the dedicated diagram keys are not set.
    This prevents diagram calls from burning the post-generation quota.
    """
    diagram_keys = [k for k in [GEMINI_API_KEY_3, GEMINI_API_KEY_4] if k]
    fallback_keys = [k for k in [GEMINI_API_KEY, GEMINI_API_KEY_2] if k]
    keys = diagram_keys if diagram_keys else fallback_keys
    if not keys:
        raise RuntimeError("No Gemini keys set for diagram generation.")
    if not diagram_keys:
        print("  [Gemini] GEMINI_API_KEY_3/4 not set — using post keys for diagram (shared quota).")
    return _gemini_call(prompt, system_instruction, keys)


# ==============================================================================
# CHARACTER LIMIT HELPERS
# ==============================================================================

def validate_post_length(content: str, platform: str = PLATFORM) -> bool:
    limit = PLATFORM_CHAR_LIMITS.get(platform.lower(), 3000)
    if len(content) > limit:
        raise ValueError(
            f"{platform.capitalize()} posts cannot exceed {limit} chars. "
            f"Current: {len(content)} chars."
        )
    return True


def truncate_for_platform(content: str, platform: str = PLATFORM) -> str:
    limit = PLATFORM_CHAR_LIMITS.get(platform.lower(), 3000)
    if len(content) <= limit:
        return content
    truncated = content[:limit - 3]
    last_period = truncated.rfind(".")
    if last_period > limit // 2:
        truncated = truncated[:last_period + 1]
    else:
        truncated = truncated.rstrip() + "..."
    print(f"  [truncate] Hard-truncated to {len(truncated)} chars.")
    return truncated


def _clean_post(text: str) -> str:
    """Strip markdown formatting that LinkedIn renders as literal characters."""
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', text)
    text = re.sub(r'^\s*\*\s+', '-> ', text, flags=re.MULTILINE)
    return text.strip()


def _shorten_if_needed(post: str, system_prompt: str) -> str:
    """Ask Gemini to shorten the post if it exceeds LinkedIn's limit."""
    limit = PLATFORM_CHAR_LIMITS[PLATFORM]
    for attempt in range(2):
        if len(post) <= limit:
            break
        print(f"  Post is {len(post)} chars — shortening (attempt {attempt + 1}/2)...")
        shorten_prompt = (
            f"This LinkedIn post is {len(post)} chars, over the {limit}-char limit.\n\n"
            f"Shorten to under {limit - 50} chars. Keep the hook, insights, and hashtags. "
            f"Cut filler, not ideas. Plain text only — no markdown, no JSON wrapper.\n\n"
            f"Original:\n{post}\n\nOutput ONLY the shortened post."
        )
        try:
            shortened = generate_text(shorten_prompt, system_prompt)
            post = _clean_post(shortened)
        except Exception as err:
            print(f"  [warn] Shorten attempt {attempt + 1} failed ({err}). Using hard truncation.")
            break
    if len(post) > limit:
        post = truncate_for_platform(post, PLATFORM)
    return post


# ==============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# BUSINESS USECASE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
# ==============================================================================

BIZ_POST_SYSTEM = """
You are ghostwriting LinkedIn posts for a senior software engineer and solutions architect
who has built production AI systems and writes publicly about technical architecture.

Voice: first-person practitioner. Ground every claim in system design specifics — service names,
data flow, tradeoffs, failure modes. Never claim real client results or measured ROI; use plausible
estimates and mark them as such. No hype, no generic motivational lines.

LENGTH: Keep the post under 1600 characters (roughly 250-280 words). Short, crisp, and dense.
Every sentence must earn its place. Cut anything that restates what was already said.

Platform rules: plain text only. Use "→ " for bullets (never "*" or "-"). No markdown. No emoji
by default; at most one if it genuinely helps readability.
""".strip()

BIZ_POST_PROMPT = """
Write a LinkedIn post about this business use case / startup idea.

Use case: {title}
Industry: {industry}
Problem : {problem}
Solution: {solution_summary}
Business impact (estimate): {business_impact}
AWS services involved: {aws_services}
Hook to use as inspiration: {linkedin_hook}
Tone: {tone}

Structure:
1. Opening hook (< 210 chars before LinkedIn's "see more"). Use the provided hook as inspiration —
   rewrite it into a crisp, specific observation or number. Never open with a question, command,
   or "Here's how/why".
2. The business problem — 2 sentences max, specific to this industry. What breaks today without AI?
3. The technical solution — name the key AWS services and their roles in one short paragraph.
   Be specific: why this service over the alternative?
4. Two or three bullet points (prefixed "→ ") covering the most important of: data flow highlight,
   scalability concern, or security/compliance consideration.
5. One closing line — what this architecture pattern means for engineers building similar systems.
6. 3-5 hashtags on their own line. Specific, searchable, camelCase.

Writing rules:
- HARD LIMIT: under 1600 characters total (roughly 250 words). Count carefully. Be concise.
- No hype ("game-changing", "revolutionary", "transformative").
- No company name claims or real client results — use "a system like this" framing.
- Specific numbers are fine if clearly framed as estimates or design targets.
- No closing question — end with the closing line and hashtags only.

Output ONLY valid JSON:
{{
  "post_text": "full LinkedIn post including hashtags at the end",
  "hook_score": <1-10>,
  "viral_score": <1-10>
}}
""".strip()


def generate_biz_post(use_case: dict, tone: str) -> str:
    """Generate a LinkedIn post for the given use case. Returns post_text string."""
    prompt = BIZ_POST_PROMPT.format(
        title=use_case.get("title", ""),
        industry=use_case.get("industry", ""),
        problem=use_case.get("problem", ""),
        solution_summary=use_case.get("solution_summary", ""),
        business_impact=use_case.get("business_impact", ""),
        aws_services=", ".join(use_case.get("aws_services", [])),
        linkedin_hook=use_case.get("linkedin_hook", ""),
        tone=tone,
    )

    print("[ Step 2 ] Generating business use case post...")
    raw = generate_text(prompt, BIZ_POST_SYSTEM).strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    hook_score = viral_score = "?"
    try:
        parsed = json.loads(raw)
        post = parsed["post_text"].strip()
        hook_score  = parsed.get("hook_score", "?")
        viral_score = parsed.get("viral_score", "?")
    except (json.JSONDecodeError, KeyError):
        # Try regex recovery
        m = re.search(r'"post_text"\s*:\s*"(.*?)"\s*[,}]', raw, re.DOTALL)
        if m:
            post = m.group(1).replace('\\n', '\n').replace('\\"', '"')
        else:
            post = raw

    post = _clean_post(post)
    print(f"  Hook score: {hook_score}/10  |  Viral score: {viral_score}/10")
    print(f"  Characters: {len(post)}/{PLATFORM_CHAR_LIMITS[PLATFORM]}")
    return post


# ==============================================================================
# HUMANIZE (shared by both pipelines)
# ==============================================================================

HUMANIZE_SYSTEM = """
You are an expert human editor. Rewrite the text so it sounds like it was written by a real,
thoughtful human — NOT by an AI.

REMOVE:
- "In today's fast-paced world...", "In the ever-evolving landscape...", "Let's dive in..."
- Reveal bridges: "The result?", "Plot twist:", "Here's what/how/why" — just state it
- 2026 AI vocabulary overuse: significant, crucial, notably, comprehensive, robust, leverage,
  foster, nuanced, holistic, streamline, elevate, empower, utilize, facilitate, harness, unlock,
  navigate, seamless, ecosystem — replace with the plain word or a real number
- Generic closing lines: "What do you think?", "Thoughts?", "Agree or disagree?"
- Excessive emoji, decorative bullets, emoji before every heading
- Fake enthusiasm, corporate language, motivational filler
- Negative parallelism overuse ("it's not X, it's Y") — max one per post, rewrite as plain sentences
- Present-participle openers ("Leveraging X, we...") — put the actor first

KEEP: natural rhythm, the author's actual ideas, specific numbers, named services, all hashtags.

PLATFORM RULES (do not break):
- Plain text only. No markdown (no *, no _, no # headings).
- Bullet lines use "→ " prefix, never "*" or "-".
- Do not add hashtags. Do not spell out URLs.
- Match input length or shorter — never pad with new content.

Return ONLY the rewritten post. No explanation.
""".strip()


def humanize_post(post_text: str) -> str:
    """Run the post through a human-editor rewrite pass."""
    print("[ Step 2.5 ] Humanizing post...")
    rewritten = generate_text(post_text, HUMANIZE_SYSTEM).strip()
    rewritten = re.sub(r'^```(?:\w+)?\s*', '', rewritten)
    rewritten = re.sub(r'\s*```$', '', rewritten)
    rewritten = _clean_post(rewritten)
    rewritten = _shorten_if_needed(rewritten, HUMANIZE_SYSTEM)

    print(f"\n  Humanized post:\n  {'─'*50}")
    for line in rewritten.split("\n"):
        print(f"  {line}")
    print(f"  {'─'*50}")
    print(f"  Characters: {len(rewritten)}/{PLATFORM_CHAR_LIMITS[PLATFORM]}\n")

    validate_post_length(rewritten, PLATFORM)
    return rewritten


# ==============================================================================
# AWS DIAGRAM (business_usecase pipeline only)
# ==============================================================================

def build_aws_diagram(use_case: dict) -> str | None:
    """
    Generate diagram spec via Gemini, render to PNG via diagrams library.
    Returns PNG path or None on failure (pipeline continues text-only).
    """
    if not INCLUDE_INFOGRAPHIC:
        return None

    try:
        import sys as _sys, pathlib as _pl
        _root = str(_pl.Path(__file__).parent.parent)
        _scripts = str(_pl.Path(__file__).parent)
        for _p in (_root, _scripts):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from aws_diagram import generate_diagram_spec, render_diagram
    except ImportError as ie:
        print(f"  [aws_diagram] skipped — import error: {ie}")
        return None

    print("[ Step 3 ] Generating AWS architecture diagram...")
    try:
        spec = generate_diagram_spec(use_case, generate_text_diagram)
        png  = render_diagram(spec, _PNG_PATH)
        print(f"  AWS diagram saved: {png}\n")
        return png
    except Exception as exc:
        print(f"  [aws_diagram] WARNING: failed ({exc}) — posting text-only.\n")
        return None


# ==============================================================================
# LEGACY PIPELINE HELPERS  (news / educational / personal / advanced)
# ==============================================================================

# (Commented-out prompts preserved for reference — not called in business_usecase mode)

_LEGACY_SYSTEM = """
You are ghostwriting LinkedIn posts for ONE individual engineer who studies AI/ML deeply every
day and shares what they learn in public. This person is NOT a founder, NOT a CEO, NOT a manager.
Teach people something real about the technology. Make them stop scrolling in the first 2 lines.
VOICE: hands-on engineer, first-person (I ran, I tested, I debugged). No "we", no company voice.
No hype. No bold/italic markdown. Bullets use "→ " prefix.
""".strip()

STYLE_1_PROMPT = """
━━━ INPUT ━━━
Persona: {persona}  |  Slot: {slot_label}  |  Topic: {topic}  |  Tone: {tone}

Research:
{research}

━━━ FORMAT: PROBLEM → SOLUTION ━━━
1. Opening hook (< 210 chars). Formula: {hook_name}. {hook_skeleton}
2. Why the problem is worse than it looks — 1 line.
3. The specific technical solution — name it directly.
4. 3 concrete capabilities that prove it solves the problem — 1 line each.
5. Closing reframe — 1 line.
6. Follow CTA — 1 short line.

Rules: first-person engineer voice; no hype; no markdown; "→ " for bullets; up to 3000 chars;
3-5 hashtags at the end.

Optional power phrases (use at most one if it fits): {power_phrases}

Output ONLY valid JSON:
{{
  "post_text": "...",
  "hook_score": <1-10>,
  "viral_score": <1-10>,
  "image_recommended": <bool>,
  "image_type": "<infographic|meme|carousel|chart|none>",
  "image_prompt": "...",
  "manim_code": "...",
  "first_comment": "..."
}}
""".strip()

STYLE_2_PROMPT = """
━━━ INPUT ━━━
Persona: {persona}  |  Slot: {slot_label}  |  Topic: {topic}  |  Tone: {tone}

Research:
{research}

━━━ FORMAT: SCENARIO → RISK → SOLUTION ━━━
1. Vivid scenario hook or strong stat — < 210 chars.
2. Why it matters at scale — 1 line.
3. The specific technical risk — 1 line.
4. Solution — name the technique or architecture.
5. Exactly 3 bullets (prefix "→ ") — specific, technical.
6. Closing question naming the specific thing this post covers.

Rules: first-person engineer; no hype; no markdown; "→ " bullets; up to 3000 chars;
3-5 hashtags at the end.

Optional power phrases (use at most one): {power_phrases}

Output ONLY valid JSON:
{{
  "post_text": "...",
  "hook_score": <1-10>,
  "viral_score": <1-10>,
  "image_recommended": <bool>,
  "image_type": "<infographic|meme|carousel|chart|none>",
  "image_prompt": "...",
  "manim_code": "...",
  "first_comment": "..."
}}
""".strip()

_SLOT_ORDER = ["news", "educational", "personal", "advanced"]


def research_topic(topic: str, niche: str) -> str:
    """Find 5 recent articles on the topic via Exa."""
    print("\n[ Step 1 ] Researching topic with Exa...")
    if _Exa is None:
        print("  [exa] exa-py not installed — skipping research.")
        return ""
    exa = _Exa(api_key=EXA_API_KEY)
    results = exa.search(
        query=f"{topic} {niche} insights trends 2025",
        type="auto",
        num_results=5,
        start_published_date="2025-01-01",
        contents={"text": {"max_characters": 800}, "highlights": {"max_characters": 300}},
    )
    lines = []
    for i, result in enumerate(results.results, 1):
        lines.append(f"Source {i}: {result.title or 'Untitled'}")
        lines.append(f"URL: {result.url}")
        if result.highlights:
            lines.append(f"Key insight: {result.highlights[0]}")
        if result.text:
            lines.append(f"Context: {(result.text or '')[:300]}...")
        lines.append("")
    print(f"  Found {len(results.results)} sources.\n")
    return "\n".join(lines)


def generate_legacy_post(topic: str, tone: str, niche: str, persona: str, research: str) -> tuple[str, str, str]:
    """
    Legacy post generation for non-business_usecase slots.
    Returns (post_text, first_comment, manim_code).
    """
    slot_index  = _SLOT_ORDER.index(CONTENT_SLOT) if CONTENT_SLOT in _SLOT_ORDER else 0
    day_of_year = datetime.now(timezone.utc).timetuple().tm_yday
    use_style_2 = (day_of_year + slot_index) % 2 == 1
    template    = STYLE_2_PROMPT if use_style_2 else STYLE_1_PROMPT
    hook        = select_hook_formula(CONTENT_SLOT, day_of_year, slot_index)
    power_phrases = "; ".join(POWER_PHRASES)

    style_name = "2 (scenario/risk/solution)" if use_style_2 else "1 (problem/solution)"
    print(f"[ Step 2 ] Generating post (style: {style_name})...")
    if not use_style_2:
        print(f"  Hook: {hook['id']} — {hook['name']}")

    prompt = template.format(
        persona=persona,
        slot_label=SLOT_LABEL,
        topic=topic,
        tone=tone,
        research=research[:2000],
        hook_name=hook["name"],
        hook_skeleton=hook["skeleton"],
        power_phrases=power_phrases,
    )

    raw = generate_text(prompt, _LEGACY_SYSTEM).strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    manim_code = ""
    first_comment = f"Curious about this — what's your take on {topic.lower()}?"
    try:
        parsed = json.loads(raw)
        post = parsed["post_text"].strip()
        manim_code    = parsed.get("manim_code", "")
        first_comment = parsed.get("first_comment", "").strip() or first_comment
        print(f"  Hook score: {parsed.get('hook_score','?')}/10  |  Viral: {parsed.get('viral_score','?')}/10")
    except (json.JSONDecodeError, KeyError):
        m = re.search(r'"post_text"\s*:\s*"(.*)"\s*,\s*"hook_score"', raw, re.DOTALL)
        if m:
            post = m.group(1).replace('\\n', '\n').replace('\\"', '"')
        else:
            post = raw

    post = _clean_post(post)
    post = _shorten_if_needed(post, _LEGACY_SYSTEM)

    print(f"\n  Generated post:\n  {'─'*50}")
    for line in post.split("\n"):
        print(f"  {line}")
    print(f"  {'─'*50}")
    print(f"  Characters: {len(post)}/{PLATFORM_CHAR_LIMITS[PLATFORM]}\n")

    validate_post_length(post, PLATFORM)
    return post, first_comment, manim_code


def build_manim_infographic(manim_code: str) -> str | None:
    """Render the Manim scene code returned by Gemini (legacy pipeline)."""
    if not INCLUDE_INFOGRAPHIC or not manim_code or not manim_code.strip():
        return None
    try:
        import sys as _sys, pathlib as _pl
        _root    = str(_pl.Path(__file__).parent.parent)
        _scripts = str(_pl.Path(__file__).parent)
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        if _scripts not in _sys.path:
            _sys.path.insert(0, _scripts)
        import manim_infographic as manim_renderer
        print("[ Step 2.6 ] Rendering Manim scene...")
        png = manim_renderer.render_manim_source(manim_code, _PNG_PATH)
        print(f"  Manim infographic: {png}\n")
        return png
    except Exception as exc:
        print(f"  [manim] WARNING: failed ({exc}) — text-only.\n")
        return None


# ==============================================================================
# LINKEDIN API  (shared)
# ==============================================================================

def post_to_linkedin(post_text: str, image_urn: str = None) -> str:
    """Publish the post to LinkedIn. Returns post ID."""
    print("[ Step 3/4 ] Posting to LinkedIn...")

    if not LINKEDIN_ACCESS_TOKEN:
        raise RuntimeError("LINKEDIN_ACCESS_TOKEN not set.")
    if not LINKEDIN_PERSON_ID:
        raise RuntimeError("LINKEDIN_PERSON_ID not set.")

    author_urn = f"urn:li:person:{LINKEDIN_PERSON_ID}"

    if image_urn:
        share_content = {
            "shareCommentary":    {"text": post_text},
            "shareMediaCategory": "IMAGE",
            "media": [{"status": "READY", "media": image_urn}],
        }
    else:
        share_content = {
            "shareCommentary":    {"text": post_text},
            "shareMediaCategory": "NONE",
        }

    payload = {
        "author":         author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": share_content,
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.post(
            "https://api.linkedin.com/v2/ugcPosts",
            headers={
                "Authorization":             f"Bearer {LINKEDIN_ACCESS_TOKEN}",
                "Content-Type":              "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
            },
            json=payload,
            timeout=15,
        )

        if response.status_code == 201:
            post_id = response.headers.get("x-restli-id", "unknown")
            print(f"  Published! LinkedIn Post ID: {post_id}\n")
            return post_id

        if response.status_code == 429:
            wait = RETRY_BASE_SECONDS * attempt
            print(f"  LinkedIn 429 rate limit — waiting {wait}s (attempt {attempt}/{MAX_RETRIES})...")
            if attempt == MAX_RETRIES:
                raise RuntimeError("LinkedIn rate limit — too many requests.")
            time.sleep(wait)
            continue

        if response.status_code == 401:
            raise RuntimeError("LinkedIn token invalid or expired. Re-run get_linkedin_token.py.")

        try:
            err = response.json()
        except ValueError:
            err = response.text
        raise RuntimeError(f"LinkedIn API error {response.status_code}: {err}")

    raise RuntimeError("LinkedIn API: exhausted retry attempts.")


def upload_image_to_linkedin(png_path: str) -> str | None:
    """Upload a PNG to LinkedIn media API. Returns asset URN or None on failure."""
    if not png_path:
        return None
    try:
        import sys as _sys, pathlib as _pl
        _root = str(_pl.Path(__file__).parent.parent)
        _scripts = str(_pl.Path(__file__).parent)
        for _p in (_root, _scripts):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from infographic import upload_to_linkedin
        urn = upload_to_linkedin(png_path, LINKEDIN_ACCESS_TOKEN, LINKEDIN_PERSON_ID)
        return urn
    except Exception as exc:
        print(f"  [upload] WARNING: image upload failed ({exc}) — posting text-only.\n")
        return None


# First-comment posting is DISABLED.  Function kept for easy re-enable.
def _post_engagement_comment_disabled(post_id: str, comment_text: str) -> None:
    """DISABLED — uncomment the call in main() to re-enable first comments."""
    pass


# ==============================================================================
# MAIN
# ==============================================================================

def main(preview: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"  LinkedIn Post Agent — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    if preview:
        print("  MODE: PREVIEW (no LinkedIn posting)")
    print(f"{'='*60}")
    print(f"  Slot  : [{CONTENT_SLOT.upper()}] {SLOT_LABEL}")
    print(f"{'='*60}\n")

    try:
        # ── BUSINESS USECASE PIPELINE ──────────────────────────────────────────
        if _IS_BIZ:
            # Add scripts/ dir to path so sibling modules import cleanly
            if _script_dir not in sys.path:
                sys.path.insert(0, _script_dir)
            from ideation import generate_use_case

            # Step 1: Invent the use case
            print("[ Step 1 ] Inventing business use case with Gemini...")
            use_case = generate_use_case(generate_text)
            print(f"  Title    : {use_case['title']}")
            print(f"  Industry : {use_case['industry']}")
            print(f"  Type     : {use_case['solution_type']}")
            print(f"  Services : {', '.join(use_case['aws_services'])}\n")

            # Step 2: Generate the post
            tone  = random.choice(TONES)
            draft = generate_biz_post(use_case, tone)

            # Step 2.5: Humanize
            post = humanize_post(draft)

            # Step 3: AWS diagram
            png_path  = build_aws_diagram(use_case)
            image_urn = upload_image_to_linkedin(png_path) if not preview else None

            if preview:
                print(f"{'='*60}")
                print("  PREVIEW ONLY — post NOT published.")
                if png_path:
                    print(f"  Diagram preview: {png_path}")
                print(f"{'='*60}\n")
                return

            validate_post_length(post, PLATFORM)
            post_id = post_to_linkedin(post, image_urn=image_urn)

            # First comment DISABLED
            # _post_engagement_comment_disabled(post_id, "")

            print(f"{'='*60}")
            print("  Done! Business use case post published.")
            print(f"  Image attached : {'yes' if image_urn else 'no (text-only)'}")
            print(f"  LinkedIn ID    : {post_id}")
            print(f"{'='*60}\n")

        # ── LEGACY PIPELINE ────────────────────────────────────────────────────
        else:
            topic_override = os.environ.get("TOPIC_OVERRIDE", "").strip()
            topic = topic_override if topic_override else random.choice(TOPICS)
            tone  = random.choice(TONES)
            print(f"  Topic : {topic}" + (" (forced)" if topic_override else ""))
            print(f"  Tone  : {tone}\n")

            research = research_topic(topic, NICHE)
            draft, first_comment, manim_code = generate_legacy_post(
                topic, tone, NICHE, PERSONA, research
            )
            post     = humanize_post(draft)
            png_path = build_manim_infographic(manim_code)

            if preview:
                print(f"  Would comment: {first_comment}\n")
                print(f"{'='*60}")
                print("  PREVIEW ONLY — post NOT published.")
                if png_path:
                    print(f"  Infographic: {png_path}")
                print(f"{'='*60}\n")
                return

            validate_post_length(post, PLATFORM)
            image_urn = upload_image_to_linkedin(png_path)
            post_id   = post_to_linkedin(post, image_urn=image_urn)

            # First comment DISABLED
            # _post_engagement_comment_disabled(post_id, first_comment)

            print(f"{'='*60}")
            print("  Done! Post published to LinkedIn.")
            print(f"  Image attached : {'yes' if image_urn else 'no (text-only)'}")
            print(f"  LinkedIn ID    : {post_id}")
            print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n  ERROR: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main(preview="--preview" in sys.argv)
