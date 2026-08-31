"""
visualize_dag.py
Visualizes a DAG from a JSON config file in the style of DARTS/PDARTS papers.

Usage:
    python visualize_dag.py graph_params_dag4.json
    python visualize_dag.py graph_params_dag4.json --output my_dag --format png
"""

import json
import argparse
import math
from pathlib import Path
import graphviz


# Node color scheme
NODE_COLORS = {
    "start": "darkseagreen2",
    "end":   "palegoldenrod",
    "intermediate": "lightblue",
}


def short_name(node_id: str) -> str:
    """Strip the @dagN suffix for display, e.g. '1@dag4' -> '1'."""
    return node_id.split("@")[0]


def node_role(node_id: str, edges: list) -> str:
    """Classify a node as start, end, or intermediate."""
    name = short_name(node_id)
    if name == "start":
        return "start"
    if name == "end":
        return "end"
    return "intermediate"


def edge_label(attrs: dict) -> str:
    """Build a compact edge label from edge attributes."""
    parts = []
    t = attrs.get("type", "")
    t = "conv" if t=="convolution" else t
    ks = attrs.get("kernel_size")
    if ks:
        t += f" {ks[0]}x{ks[1]}"
    bias = attrs.get("use_bias")
    if bias is not None and bias:
        t += " + bias"
    if t:
        parts.append(t)
    return "\n".join(parts)


def node_label(node_id: str, attrs: dict) -> str:
    """Build a multi-line node label from node attributes."""
    name = short_name(node_id)
    lines = [name]
    if "size" in attrs:
        lines.append(f"size={attrs['size']}")
    if "activation" in attrs and attrs['activation'] != "id":
        lines.append(f"{attrs['activation']}")
    if "use_layer_norm" in attrs and attrs['use_layer_norm']:
        lines.append(f"LN")
    return "\n".join(lines)


def build_graph(config: dict, rankdir: str = "LR") -> graphviz.Digraph:
    edges = config["edges"]
    node_attrs = config.get("node_attributes", {})
    edge_attrs = config.get("edge_attributes", {})

    # Compute size range for proportional box width
    sizes = [attrs["size"] for attrs in node_attrs.values() if "size" in attrs]
    min_size = min(sizes, default=1)
    max_size = max(sizes, default=1)
    min_width, max_width = 0.8, 1.6

    def channel_width(size: int) -> str:
        if max_size == min_size:
            return f"{(min_width + max_width) / 2:.2f}"
        t = (math.sqrt(size) - math.sqrt(min_size)) / (math.sqrt(max_size) - math.sqrt(min_size))
        w = min_width + t * (max_width - min_width)
        return f"{w:.2f}"

    g = graphviz.Digraph(
        format="pdf",
        graph_attr=dict(rankdir=rankdir, nodesep="0.7", ranksep="1.0", fontname="times"),
        edge_attr=dict(fontsize="15", fontname="times", arrowsize="0.8"),
        node_attr=dict(
            style="filled",
            shape="rect",
            align="center",
            fontsize="13",
            fontname="times",
            penwidth="2",
        ),
    )

    # Add nodes
    for node_id, attrs in node_attrs.items():
        role = node_role(node_id, edges)
        color = NODE_COLORS[role]
        label = node_label(node_id, attrs)
        size = channel_width(attrs["size"]) if "size" in attrs else "1.5"
        g.node(node_id, label=label, fillcolor=color, width=size, height=size, fixedsize="true")

    # Add edges
    for src, dst in edges:
        # Edge attributes dict uses string repr of tuple as key
        key = str((src, dst))
        attrs = edge_attrs.get(key, {})
        label = edge_label(attrs)
        g.edge(src, dst, label=label)

    return g


def main():
    parser = argparse.ArgumentParser(description="Visualize a DAG from a JSON config.")
    parser.add_argument("config", help="Path to the JSON config file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output filename without extension (default: <config_stem>)")
    parser.add_argument("--format", "-f", default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="Output format (default: pdf)")
    parser.add_argument("--rankdir", default="LR",
                        choices=["LR", "TB"],
                        help="Graph direction: LR (left→right) or TB (top→bottom)")
    parser.add_argument("--view", action="store_true",
                        help="Open the rendered file after generation")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)

    output = args.output or config_path.stem
    output = Path("temp", output)

    g = build_graph(config, rankdir=args.rankdir)
    g.format = args.format
    rendered = g.render(output, view=args.view, cleanup=True)
    print(f"Saved: {rendered}")


if __name__ == "__main__":
    main()
