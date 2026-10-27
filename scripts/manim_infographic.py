"""Render Gemini-generated Manim Community Edition scene code to a PNG."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from pathlib import Path


def validate_manim_source(source: str) -> None:
    """Apply a conservative syntax-level allowlist before running model output."""
    if len(source) > 30_000:
        raise ValueError("Generated Manim source exceeds the 30,000-character limit.")
    tree = ast.parse(source, filename="generated_manim_scene.py")
    if len(tree.body) != 2 or not isinstance(tree.body[0], ast.ImportFrom) or not isinstance(tree.body[1], ast.ClassDef):
        raise ValueError("Source may contain only the Manim import and PostInfographic class.")
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    if len(imports) != 1 or not isinstance(imports[0], ast.ImportFrom):
        raise ValueError("Manim source must contain only `from manim import *`.")
    imp = imports[0]
    if imp.module != "manim" or len(imp.names) != 1 or imp.names[0].name != "*":
        raise ValueError("Manim source must contain only `from manim import *`.")

    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1 or classes[0].name != "PostInfographic":
        raise ValueError("Source must define exactly one class named PostInfographic.")
    if not any(isinstance(base, ast.Name) and base.id == "Scene" for base in classes[0].bases):
        raise ValueError("PostInfographic must subclass Scene.")

    forbidden_names = {
        "open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
        "globals", "locals", "getattr", "setattr", "delattr", "vars",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            raise ValueError(f"Disallowed Python operation: {node.id}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder name access is not allowed in generated scene code.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed in generated scene code.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
            raise ValueError(f"Disallowed function call: {node.func.id}")


def render_manim_source(source: str, output_path: str, timeout_seconds: int = 180) -> str:
    """Write the generated script, render its final frame with Manim, and return PNG path."""
    validate_manim_source(source)
    manim = shutil.which("manim")
    if not manim:
        raise RuntimeError("Manim is not installed. Install Manim Community Edition to render infographics.")

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    script_path = output.parent / "generated_manim_scene.py"
    script_path.write_text(source, encoding="utf-8")
    media_dir = output.parent / "manim_media"
    command = [
        manim, "-s", "-ql", "--format=png", "--media_dir", str(media_dir),
        "-o", output.stem, str(script_path), "PostInfographic",
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        command, cwd=str(output.parent), env=env, capture_output=True, text=True,
        timeout=timeout_seconds, check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout)[-4000:]
        raise RuntimeError(f"Manim rendering failed (exit {result.returncode}):\n{detail}")

    candidates = list(media_dir.rglob(f"{output.stem}.png"))
    if not candidates:
        candidates = list(media_dir.rglob("PostInfographic.png"))
    if not candidates:
        raise RuntimeError("Manim completed but did not produce a PNG output.")
    rendered = max(candidates, key=lambda path: path.stat().st_mtime)
    if rendered.resolve() != output:
        shutil.copy2(rendered, output)
    return str(output)
