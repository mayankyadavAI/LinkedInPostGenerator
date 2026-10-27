#!/usr/bin/env python3
"""
Local diagram preview — Gemini picks diagram type from post text,
Matplotlib renders PNG, opens it.  Does NOT post to LinkedIn.

Usage:
  python test_infographic.py                          # uses sample post
  python test_infographic.py flow                     # force a diagram type
  python test_infographic.py "RAG systems explained"  # custom topic
"""
import sys, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from dotenv import load_dotenv
load_dotenv()

# ── args ───────────────────────────────────────────────────────────────────────
arg = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
FORCE_TYPE = arg if arg in ("flow", "split", "layers", "cycle", "scale", "stats") else None
TOPIC = arg if arg and not FORCE_TYPE else "Why ChatGPT sounds confident when it's wrong"

SAMPLE_POST = """
LLMs don't look things up — they predict the most likely next token based on
patterns learned during training. There's no fact database being queried. When
the model says something confidently wrong, it's because confident-sounding text
appeared frequently in training data about similar topics.

Temperature controls how 'risky' the next-token selection is. At 0 you get the
same deterministic answer every time. At 1.5 you get creative chaos. Most
production systems sit around 0.7.

The KV cache stores attention keys and values for previous tokens so the model
doesn't recompute them every step. Without it, generating 1000 tokens would be
roughly 1000x more expensive than generating 1.
"""

OUT = str(Path(__file__).parent / "renderer" / "output" / "preview.png")


def main():
    from scripts.generate_and_schedule import generate_text
    import scripts.infographic as ig

    print(f"\nTopic  : {TOPIC}")
    if FORCE_TYPE:
        print(f"Forcing diagram type: {FORCE_TYPE}")

    print("Asking Gemini to pick diagram type and generate content...\n")

    if FORCE_TYPE:
        # bypass Gemini, use the built-in sample for that type
        from renderer.render_matplotlib import _SAMPLES, render
        content = _SAMPLES[FORCE_TYPE]
        print(f"  Using built-in sample for '{FORCE_TYPE}'")
    else:
        content = ig.generate_diagram_content(TOPIC, SAMPLE_POST, generate_text)

    print(f"  diagram_type : {content.get('diagram_type')}")
    print(f"  title        : {content.get('title')}")
    print(f"  subtitle     : {content.get('subtitle')}")
    print(f"  takeaway     : {content.get('takeaway')}\n")

    ig.render_diagram(content, OUT)
    print(f"Saved → {OUT}\n")
    subprocess.run(["open", OUT])


if __name__ == "__main__":
    main()
