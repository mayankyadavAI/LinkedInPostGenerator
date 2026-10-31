"""
AWS Architecture Diagram Generator — Step 3 of the business use case pipeline.

Produces solution-architect-level architecture diagrams:
  - 5-8 logical layers (network boundary, security, ingestion, agents, processing, storage, observability)
  - 12-20 nodes covering the full prod-grade stack
  - Labeled directional edges showing data flow, API calls, triggers, and async events
  - Security and observability services included as first-class layers

Pipeline:
  generate_diagram_spec()  → Gemini (diagram keys 3+4) → detailed JSON spec
  render_diagram()         → `diagrams` library + Graphviz → PNG

GitHub Actions: apt-get install -y graphviz && pip install diagrams
Local:          brew install graphviz && pip install diagrams
"""

import re
import json
import pathlib

# ==============================================================================
# Gemini prompt — solution-architect level
# ==============================================================================

_DIAGRAM_SYSTEM = """
You are a principal AWS solutions architect designing production diagrams for
LinkedIn posts read by senior engineers. Your diagrams show the FULL system:
network boundary, security controls, data ingestion, AI/agent layer, processing,
persistence, and observability. Every diagram must be immediately useful as a
reference architecture — not a toy example.
""".strip()

_DIAGRAM_PROMPT = """
Design a DETAILED, solution-architect-level AWS architecture diagram for this use case.

Use case    : {title}
Category    : {category}
Industry    : {industry}
Problem     : {problem}
Solution    : {solution_summary}
Key insight : {architecture_insight}
AWS services: {aws_services}
Agent types : {agent_types}
Solution type: {solution_type}

DIAGRAM REQUIREMENTS — read carefully:
1. Include 5-7 clusters covering ALL of these layers (use only what applies):
   - "Client / Entry" — users, external APIs, SDKs, web/mobile clients
   - "Security & Network" — WAF, Cognito, IAM, VPC, API Gateway / ALB
   - "Orchestration" — Step Functions, EventBridge, SQS/SNS, Lambda triggers
   - "Agent Layer" — Bedrock Agents, LLM calls, tool-use, memory, RAG retrieval
   - "Processing" — SageMaker, Lambda compute, ECS/Fargate workloads, Glue
   - "Data & Storage" — S3, Aurora/RDS, DynamoDB, OpenSearch, ElastiCache
   - "Observability" — CloudWatch, X-Ray, CloudTrail, Bedrock model metrics

2. Target 12-18 nodes total. Every important AWS service in the use case must appear.
   Do NOT collapse the architecture — show each service as its own node.

3. Edges must show meaningful flow:
   - Use "solid" for synchronous calls, real-time data, API requests
   - Use "dashed" for async events, triggers, batch jobs, feedback loops
   - Label every edge with the data or operation flowing (e.g. "invoke tool",
     "stream tokens", "query embeddings", "publish event", "read audit log")
   - Show bidirectional flows where relevant (e.g. agent → tool → agent)

4. Edge colors by semantic type:
   - blue  = API call / invocation
   - green = data read/write / persistence
   - orange = event / trigger / async
   - purple = ML inference / embedding / model call
   - gray  = logging / metrics / audit
   - red   = security check / auth / guard

Output ONLY valid JSON. No markdown fences, no explanation:
{{
  "diagram_title": "6-10 word title",
  "diagram_direction": "LR",
  "clusters": [
    {{
      "name": "cluster name",
      "color": "blue | green | orange | red | purple | teal | gray",
      "nodes": [
        {{
          "id": "unique_snake_case_id",
          "aws_class": "EXACT diagrams library class path (see list below)",
          "label": "2-4 word label (use \\n for line break if needed)"
        }}
      ]
    }}
  ],
  "edges": [
    {{
      "from": "node_id",
      "to": "node_id",
      "label": "1-4 word label describing the flow",
      "color": "blue | green | orange | purple | gray | red",
      "style": "solid | dashed"
    }}
  ],
  "footnote": "one sentence on the key production-grade decision in this architecture"
}}

EXACT diagrams library class paths — use ONLY these:
  diagrams.aws.compute.Lambda
  diagrams.aws.compute.ECS
  diagrams.aws.compute.Fargate
  diagrams.aws.compute.EC2
  diagrams.aws.compute.EKS
  diagrams.aws.compute.Batch
  diagrams.aws.compute.ECR
  diagrams.aws.database.RDS
  diagrams.aws.database.Aurora
  diagrams.aws.database.Dynamodb
  diagrams.aws.database.ElastiCache
  diagrams.aws.database.Redshift
  diagrams.aws.analytics.Kinesis
  diagrams.aws.analytics.KinesisDataStreams
  diagrams.aws.analytics.Glue
  diagrams.aws.analytics.Athena
  diagrams.aws.analytics.Redshift
  diagrams.aws.analytics.OpenSearchService
  diagrams.aws.network.APIGateway
  diagrams.aws.network.CloudFront
  diagrams.aws.network.ELB
  diagrams.aws.network.ALB
  diagrams.aws.network.Route53
  diagrams.aws.network.VPC
  diagrams.aws.network.PrivateSubnet
  diagrams.aws.network.NATGateway
  diagrams.aws.ml.Bedrock
  diagrams.aws.ml.Sagemaker
  diagrams.aws.ml.SagemakerModel
  diagrams.aws.ml.SagemakerTrainingJob
  diagrams.aws.ml.Rekognition
  diagrams.aws.ml.Comprehend
  diagrams.aws.ml.Textract
  diagrams.aws.ml.Kendra
  diagrams.aws.ml.Lex
  diagrams.aws.ml.Forecast
  diagrams.aws.ml.FraudDetector
  diagrams.aws.ml.Personalize
  diagrams.aws.integration.SQS
  diagrams.aws.integration.SNS
  diagrams.aws.integration.Eventbridge
  diagrams.aws.integration.StepFunctions
  diagrams.aws.storage.S3
  diagrams.aws.storage.EFS
  diagrams.aws.security.Cognito
  diagrams.aws.security.IAM
  diagrams.aws.security.IAMRole
  diagrams.aws.security.WAF
  diagrams.aws.security.KMS
  diagrams.aws.security.Guardduty
  diagrams.aws.security.Inspector
  diagrams.aws.management.Cloudwatch
  diagrams.aws.management.CloudwatchAlarm
  diagrams.aws.management.CloudwatchLogs
  diagrams.aws.management.Cloudtrail
  diagrams.aws.management.Config
  diagrams.aws.iot.IotCore
  diagrams.aws.iot.IotAnalytics
  diagrams.aws.iot.Greengrass
  diagrams.aws.general.General

For Amazon Bedrock Agents: use diagrams.aws.ml.Bedrock
For OpenSearch / vector store: use diagrams.aws.analytics.OpenSearchService
For unknown services: use diagrams.aws.general.General
""".strip()


# ==============================================================================
# Gemini spec generation
# ==============================================================================

def generate_diagram_spec(use_case: dict, generate_text_fn: callable) -> dict:
    """
    Ask Gemini to produce a detailed SA-level diagram spec from the use case dict.
    Returns parsed dict with clusters + edges, or raises RuntimeError.
    """
    prompt = _DIAGRAM_PROMPT.format(
        title=use_case.get("title", ""),
        category=use_case.get("category", ""),
        industry=use_case.get("industry", ""),
        problem=use_case.get("problem", ""),
        solution_summary=use_case.get("solution_summary", ""),
        architecture_insight=use_case.get("architecture_insight", ""),
        aws_services=", ".join(use_case.get("aws_services", [])),
        agent_types=", ".join(use_case.get("agent_types", [])) or "N/A",
        solution_type=use_case.get("solution_type", ""),
    )

    for attempt in range(3):
        try:
            raw = generate_text_fn(prompt, _DIAGRAM_SYSTEM).strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            raw = raw.strip()
            spec = json.loads(raw)

            if not spec.get("clusters") or not spec.get("edges"):
                print(f"  [aws_diagram] attempt {attempt+1}: missing clusters/edges, retrying...")
                continue

            all_ids = {
                n["id"]
                for c in spec["clusters"]
                for n in c.get("nodes", [])
            }

            node_count = len(all_ids)
            if node_count < 8:
                print(f"  [aws_diagram] attempt {attempt+1}: only {node_count} nodes (need ≥8), retrying...")
                continue

            bad_edges = [
                e for e in spec["edges"]
                if e.get("from") not in all_ids or e.get("to") not in all_ids
            ]
            if bad_edges:
                # Drop bad edges instead of retrying — partial diagrams are still useful
                print(f"  [aws_diagram] attempt {attempt+1}: dropping {len(bad_edges)} bad edge(s).")
                spec["edges"] = [
                    e for e in spec["edges"]
                    if e.get("from") in all_ids and e.get("to") in all_ids
                ]

            print(f"  [aws_diagram] Spec: {node_count} nodes, {len(spec['edges'])} edges, "
                  f"{len(spec['clusters'])} clusters")
            return spec

        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  [aws_diagram] attempt {attempt+1}: parse error ({exc}), retrying...")

    raise RuntimeError("Failed to generate a valid diagram spec after 3 attempts.")


# ==============================================================================
# Colour palettes — kept in sync with current theme in render_diagram()
# ==============================================================================

# Cluster: (fill_bg, border_color) — caller may override for light/dark
_CLUSTER_COLORS_LIGHT = {
    "blue":   ("#dbeafe", "#2563eb"),
    "green":  ("#dcfce7", "#16a34a"),
    "orange": ("#ffedd5", "#ea580c"),
    "red":    ("#fee2e2", "#dc2626"),
    "purple": ("#ede9fe", "#7c3aed"),
    "teal":   ("#ccfbf1", "#0d9488"),
    "gray":   ("#f1f5f9", "#475569"),
}

_CLUSTER_COLORS_DARK = {
    "blue":   ("#1a3a5c", "#4f8dff"),
    "green":  ("#1a3d2b", "#2ec4b6"),
    "orange": ("#3d2a0a", "#f5a623"),
    "red":    ("#3d0a0a", "#f06292"),
    "purple": ("#2a1a4a", "#9d6fff"),
    "teal":   ("#0a2d2d", "#2ec4b6"),
    "gray":   ("#1e2533", "#7d8ba3"),
}

_EDGE_COLORS_LIGHT = {
    "black":  "#1e293b",
    "blue":   "#2563eb",
    "green":  "#16a34a",
    "orange": "#ea580c",
    "red":    "#dc2626",
    "purple": "#7c3aed",
    "gray":   "#64748b",
}

_EDGE_COLORS_DARK = {
    "black":  "#c0cad8",
    "blue":   "#4f8dff",
    "green":  "#2ec4b6",
    "orange": "#f5a623",
    "red":    "#f06292",
    "purple": "#9d6fff",
    "gray":   "#7d8ba3",
}


# ==============================================================================
# diagrams library renderer
# ==============================================================================

def _import_class(class_path: str):
    """Dynamically import a diagrams node class. Falls back to General on failure."""
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
        from diagrams.aws.general import General
        return General


def render_diagram(spec: dict, out_path: str, dark: bool = True) -> str:
    """
    Render a diagram spec dict to PNG using the `diagrams` library.
    dark=True  → dark background (default, matches LinkedIn dark mode)
    dark=False → white background
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

    cluster_colors = _CLUSTER_COLORS_DARK if dark else _CLUSTER_COLORS_LIGHT
    edge_colors    = _EDGE_COLORS_DARK    if dark else _EDGE_COLORS_LIGHT

    bg_color   = "#0d1520"  if dark else "white"
    font_color = "#edf0f7"  if dark else "#1e293b"
    edge_label_color = "#7d8ba3" if dark else "#475569"
    default_edge_color = "#4f8dff" if dark else "#2563eb"

    out_path     = str(pathlib.Path(out_path).with_suffix(""))
    out_dir      = str(pathlib.Path(out_path).parent)
    out_filename = pathlib.Path(out_path).name

    direction = spec.get("diagram_direction", "LR")
    title     = spec.get("diagram_title", "AWS Architecture")

    graph_attrs = {
        "fontsize":  "20",
        "bgcolor":   bg_color,
        "fontcolor": font_color,
        "fontname":  "Helvetica Neue, Helvetica, Arial, sans-serif",
        "pad":       "0.7",
        "nodesep":   "0.5",
        "ranksep":   "0.8",
        "splines":   "ortho",
    }
    node_attrs = {
        "fontsize":  "12",
        "fontcolor": font_color,
        "fontname":  "Helvetica Neue, Helvetica, Arial, sans-serif",
        "labelloc":  "b",
        "height":    "1.3",
        "width":     "1.3",
    }
    edge_attrs = {
        "fontsize":  "10",
        "fontcolor": edge_label_color,
        "fontname":  "Helvetica Neue, Helvetica, Arial, sans-serif",
        "color":     default_edge_color,
        "arrowsize": "0.75",
        "penwidth":  "1.8",
    }

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
        for cluster_spec in spec.get("clusters", []):
            cname  = cluster_spec.get("name", "Layer")
            ccolor = cluster_spec.get("color", "blue")
            bg, border = cluster_colors.get(ccolor, cluster_colors["blue"])

            with Cluster(
                cname,
                graph_attr={
                    "bgcolor":   bg,
                    "style":     "rounded,filled",
                    "color":     border,
                    "penwidth":  "2",
                    "fontsize":  "13",
                    "fontcolor": border,
                    "fontname":  "Helvetica Neue, Helvetica, Arial, sans-serif",
                    "margin":    "18",
                },
            ):
                for node_spec in cluster_spec.get("nodes", []):
                    nid       = node_spec["id"]
                    aws_class = node_spec.get("aws_class", "diagrams.aws.general.General")
                    label     = node_spec.get("label", nid)
                    NodeClass = _import_class(aws_class)
                    node_registry[nid] = NodeClass(label)

        for edge_spec in spec.get("edges", []):
            from_id = edge_spec.get("from")
            to_id   = edge_spec.get("to")
            if from_id not in node_registry or to_id not in node_registry:
                continue

            label = edge_spec.get("label", "")
            color = edge_colors.get(edge_spec.get("color", "blue"), default_edge_color)
            style = "dashed" if edge_spec.get("style") == "dashed" else "solid"

            node_registry[from_id] >> Edge(
                label=label,
                color=color,
                style=style,
                fontcolor=edge_label_color,
                fontsize="10",
            ) >> node_registry[to_id]

    final_path = str(pathlib.Path(out_dir) / (out_filename + ".png"))
    if pathlib.Path(final_path).exists():
        print(f"  [aws_diagram] PNG rendered: {final_path}")
        return final_path

    raise RuntimeError(f"diagrams did not produce PNG at: {final_path}")
