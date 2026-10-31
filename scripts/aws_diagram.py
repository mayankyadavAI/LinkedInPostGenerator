"""
AWS Architecture Diagram Generator — Step 3 of the business use case pipeline.

Pipeline:
  generate_diagram_spec()   → Gemini → structured JSON (nodes, clusters, edges)
  render_diagram()          → `diagrams` Python library + Graphviz → PNG

The `diagrams` library (pip install diagrams) ships with real AWS service icons
and uses Graphviz for layout — same icon set AWS uses in their official docs.
Produces ByteByteGo-style architecture PNGs with labeled service boxes and arrows.

GitHub Actions requirements:
  apt-get install -y graphviz
  pip install diagrams

Local requirements:
  brew install graphviz  (macOS)
  pip install diagrams
"""

import os
import re
import json
import pathlib
import tempfile

# ── Gemini prompt ─────────────────────────────────────────────────────────────

_DIAGRAM_SYSTEM = """
You are a senior AWS solutions architect. You design clear, production-grade
AWS architecture diagrams described as structured JSON. Your diagrams are
accurate, use real AWS service names, and show meaningful data flow.
""".strip()

_DIAGRAM_PROMPT = """
Design a production-grade AWS architecture diagram for this business use case.

Use case title : {title}
Industry       : {industry}
Problem        : {problem}
Solution       : {solution_summary}
AWS services   : {aws_services}
Solution type  : {solution_type}

Output ONLY valid JSON matching this exact schema. No markdown fences, no explanation:
{{
  "diagram_title": "short title for the diagram (5-8 words)",
  "diagram_direction": "LR",
  "clusters": [
    {{
      "name": "Zone/Layer name (e.g. Client Layer, Ingestion, Processing, Storage, Monitoring)",
      "color": "one of: blue | green | orange | red | purple | teal | gray",
      "nodes": [
        {{
          "id": "unique_snake_case_id",
          "aws_class": "exact Python class path in diagrams library, e.g. diagrams.aws.compute.Lambda",
          "label": "service label shown in diagram (2-4 words, can include newlines as \\n)"
        }}
      ]
    }}
  ],
  "edges": [
    {{
      "from": "node_id",
      "to": "node_id",
      "label": "short connection label (1-4 words)",
      "color": "one of: black | blue | green | orange | red | gray",
      "style": "one of: solid | dashed"
    }}
  ],
  "footnote": "one sentence describing what makes this architecture production-grade"
}}

Rules:
- Use 3-5 clusters representing logical layers (e.g. Client, Ingestion, Processing, Storage, Monitoring)
- Each cluster should have 1-4 nodes
- Total nodes: 6-14 (enough to show the full solution, not overwhelming)
- All node IDs must be unique snake_case strings
- Every edge `from` and `to` must reference a valid node `id`
- Use EXACT diagrams library class paths. Common ones:
    diagrams.aws.compute.Lambda
    diagrams.aws.compute.ECS
    diagrams.aws.compute.Fargate
    diagrams.aws.compute.EC2
    diagrams.aws.database.RDS
    diagrams.aws.database.Dynamodb
    diagrams.aws.database.ElastiCache
    diagrams.aws.database.Aurora
    diagrams.aws.analytics.Kinesis
    diagrams.aws.analytics.KinesisDataStreams
    diagrams.aws.analytics.Glue
    diagrams.aws.analytics.Athena
    diagrams.aws.analytics.Redshift
    diagrams.aws.network.APIGateway
    diagrams.aws.network.CloudFront
    diagrams.aws.network.ELB
    diagrams.aws.network.Route53
    diagrams.aws.ml.Sagemaker
    diagrams.aws.ml.SagemakerModel
    diagrams.aws.ml.Rekognition
    diagrams.aws.ml.Comprehend
    diagrams.aws.ml.Textract
    diagrams.aws.ml.Translate
    diagrams.aws.ml.Lex
    diagrams.aws.integration.SQS
    diagrams.aws.integration.SNS
    diagrams.aws.integration.Eventbridge
    diagrams.aws.integration.StepFunctions
    diagrams.aws.storage.S3
    diagrams.aws.storage.EFS
    diagrams.aws.security.Cognito
    diagrams.aws.security.IAM
    diagrams.aws.security.WAF
    diagrams.aws.security.KMS
    diagrams.aws.management.Cloudwatch
    diagrams.aws.management.Cloudformation
    diagrams.aws.general.General
    diagrams.aws.mobile.APIGateway
    diagrams.programming.flowchart.Database
- For Amazon Bedrock, use: diagrams.aws.ml.Sagemaker (closest available)
- For unknown services, use: diagrams.aws.general.General
""".strip()


# ── Gemini diagram spec generation ───────────────────────────────────────────

def generate_diagram_spec(use_case: dict, generate_text_fn: callable) -> dict:
    """
    Ask Gemini to produce a structured diagram spec from the use case dict.
    Returns parsed dict with clusters + edges, or raises RuntimeError.
    """
    prompt = _DIAGRAM_PROMPT.format(
        title=use_case.get("title", ""),
        industry=use_case.get("industry", ""),
        problem=use_case.get("problem", ""),
        solution_summary=use_case.get("solution_summary", ""),
        aws_services=", ".join(use_case.get("aws_services", [])),
        solution_type=use_case.get("solution_type", ""),
    )

    for attempt in range(3):
        try:
            raw = generate_text_fn(prompt, _DIAGRAM_SYSTEM).strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            raw = raw.strip()
            spec = json.loads(raw)

            # Validate structure
            if not spec.get("clusters") or not spec.get("edges"):
                print(f"  [aws_diagram] attempt {attempt+1}: missing clusters/edges, retrying...")
                continue

            # Collect all node IDs
            all_ids = {
                n["id"]
                for c in spec["clusters"]
                for n in c.get("nodes", [])
            }

            # Validate edges reference real IDs
            bad_edges = [
                e for e in spec["edges"]
                if e.get("from") not in all_ids or e.get("to") not in all_ids
            ]
            if bad_edges:
                print(f"  [aws_diagram] attempt {attempt+1}: {len(bad_edges)} bad edge(s), retrying...")
                continue

            print(f"  [aws_diagram] Spec ready: {len(all_ids)} nodes, {len(spec['edges'])} edges")
            return spec

        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  [aws_diagram] attempt {attempt+1}: parse error ({exc}), retrying...")

    raise RuntimeError("Failed to generate a valid diagram spec after 3 attempts.")


# ── diagrams library renderer ─────────────────────────────────────────────────

# Colour mapping for cluster graph_attr borders
_CLUSTER_COLORS = {
    "blue":   ("#1a3a5c", "#4f8dff"),
    "green":  ("#1a3d2b", "#2ec4b6"),
    "orange": ("#3d2a0a", "#f5a623"),
    "red":    ("#3d0a0a", "#f06292"),
    "purple": ("#2a1a4a", "#9d6fff"),
    "teal":   ("#0a2d2d", "#2ec4b6"),
    "gray":   ("#1e2533", "#7d8ba3"),
}

# Edge color mapping
_EDGE_COLORS = {
    "black": "#c0cad8",
    "blue":  "#4f8dff",
    "green": "#2ec4b6",
    "orange": "#f5a623",
    "red":   "#f06292",
    "gray":  "#7d8ba3",
}


def _import_class(class_path: str):
    """Dynamically import a diagrams node class from its dotted path."""
    import importlib
    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        from diagrams.aws.general import General
        return General
    module_path, class_name = parts
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        # Graceful fallback to General icon
        from diagrams.aws.general import General
        return General


def render_diagram(spec: dict, out_path: str) -> str:
    """
    Render a diagram spec dict to PNG using the `diagrams` library.
    Returns the final PNG path.
    """
    try:
        from diagrams import Diagram, Cluster, Edge
    except ImportError:
        raise RuntimeError(
            "The `diagrams` package is not installed. Run: pip install diagrams\n"
            "Also requires Graphviz: brew install graphviz (macOS) or "
            "apt-get install graphviz (Linux)"
        )

    out_path = str(pathlib.Path(out_path).with_suffix(""))  # diagrams adds .png itself
    out_dir = str(pathlib.Path(out_path).parent)
    out_filename = pathlib.Path(out_path).name

    direction = spec.get("diagram_direction", "LR")
    title = spec.get("diagram_title", "AWS Architecture")

    graph_attrs = {
        "fontsize":        "18",
        "bgcolor":         "#060b14",
        "fontcolor":       "#edf0f7",
        "fontname":        "Inter, Helvetica, Arial, sans-serif",
        "pad":             "0.5",
        "nodesep":         "0.6",
        "ranksep":         "0.9",
        "splines":         "ortho",
    }
    node_attrs = {
        "fontsize":  "13",
        "fontcolor": "#edf0f7",
        "fontname":  "Inter, Helvetica, Arial, sans-serif",
        "labelloc":  "b",
        "height":    "1.4",
        "width":     "1.4",
    }
    edge_attrs = {
        "fontsize":  "11",
        "fontcolor": "#7d8ba3",
        "fontname":  "Inter, Helvetica, Arial, sans-serif",
        "color":     "#4f8dff",
        "arrowsize": "0.8",
        "penwidth":  "1.6",
    }

    # Build node registry while rendering
    node_registry: dict = {}

    with Diagram(
        title,
        filename=str(pathlib.Path(out_dir) / out_filename),
        outformat="png",
        show=False,
        direction=direction,
        graph_attr=graph_attrs,
        node_attr=node_attrs,
        edge_attr=edge_attrs,
    ):
        # Render clusters and their nodes
        for cluster_spec in spec.get("clusters", []):
            cname = cluster_spec.get("name", "Layer")
            ccolor = cluster_spec.get("color", "blue")
            bg, border = _CLUSTER_COLORS.get(ccolor, _CLUSTER_COLORS["blue"])

            with Cluster(
                cname,
                graph_attr={
                    "bgcolor":    bg,
                    "style":      "rounded,filled",
                    "color":      border,
                    "penwidth":   "2",
                    "fontsize":   "14",
                    "fontcolor":  border,
                    "fontname":   "Inter, Helvetica, Arial, sans-serif",
                    "margin":     "20",
                },
            ):
                for node_spec in cluster_spec.get("nodes", []):
                    node_id = node_spec["id"]
                    aws_class = node_spec.get("aws_class", "diagrams.aws.general.General")
                    label = node_spec.get("label", node_id)
                    NodeClass = _import_class(aws_class)
                    node_registry[node_id] = NodeClass(label)

        # Render edges
        for edge_spec in spec.get("edges", []):
            from_id = edge_spec.get("from")
            to_id = edge_spec.get("to")
            if from_id not in node_registry or to_id not in node_registry:
                continue

            label = edge_spec.get("label", "")
            color = _EDGE_COLORS.get(edge_spec.get("color", "blue"), "#4f8dff")
            style = "dashed" if edge_spec.get("style") == "dashed" else "solid"

            edge_kwargs = {
                "label": label,
                "color": color,
                "style": style,
                "fontcolor": "#7d8ba3",
                "fontsize": "11",
            }
            node_registry[from_id] >> Edge(**edge_kwargs) >> node_registry[to_id]

    final_path = str(pathlib.Path(out_dir) / (out_filename + ".png"))
    if not pathlib.Path(final_path).exists():
        # Some versions of diagrams write <name>.png directly
        alt = str(pathlib.Path(out_dir) / (out_filename + ".png"))
        if pathlib.Path(alt).exists():
            return alt
        raise RuntimeError(f"diagrams library did not produce PNG at expected path: {final_path}")

    print(f"  [aws_diagram] PNG rendered: {final_path}")
    return final_path
