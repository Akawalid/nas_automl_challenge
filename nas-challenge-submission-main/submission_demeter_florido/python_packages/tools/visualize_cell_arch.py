"""
visualize_architecture.py

Generates a ResNet/VGG-style top-to-bottom architecture diagram from a list
of DAG JSON config files, with average pooling between each DAG and two
linear layers at the end.

Usage:
    python visualize_architecture.py dag1.json dag2.json dag3.json dag4.json [OPTIONS]

Options:
    --output      Output filename without extension (default: architecture)
    --format      pdf | png | svg (default: pdf)
    --linear      Comma-separated output sizes of the two linear layers (default: D1,D2)
    --pool-label  Label to use for pooling nodes (default: avg pool)

Example:
    python visualize_architecture.py dag1.json dag2.json dag3.json dag4.json \
        --output my_arch --format png --linear 512,10
"""

import json
import argparse
from pathlib import Path
import graphviz


def load_dag(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def dag_output_size(config: dict) -> int:
    """Infer the output size of a DAG as the size of its 'end' node."""
    for node_id, attrs in config.get("node_attributes", {}).items():
        if node_id.split("@")[0] == "end":
            return attrs.get("size", 0)
    return 0


def dag_label(index: int, config: dict) -> str:
    """Human-readable DAG name, inferred from the node IDs."""
    for node_id in config.get("node_attributes", {}):
        if "@" in node_id:
            return f"DAG{node_id.split('@dag')[1]}"   # e.g. "dag4"
    return f"DAG{index + 1}"


def build_architecture_graph(
    dag_configs: list[dict],
    dag_paths: list[str],
    linear_sizes: list[str],
    pool_label: str,
    rankdir: str,
    output_size: str | None = None,
    dpi: int = 300,
) -> graphviz.Digraph:

    # Layout-specific sizing: LR needs compact widths and consistent heights
    if rankdir == "LR":
        _ranksep       = "0.15"
        dag_h          = "0.85"
        lin_w, lin_h   = "1.15", "0.75"
        _fontsize_edge  = "8"
    else:
        _ranksep       = "0.4"
        dag_h          = "1.0"
        lin_w, lin_h   = "1.4", "0.55"
        _fontsize_edge  = "11"

    # Compact pool labels: "/2" for intermediate, "G<initials>" for global (e.g. "GAP")
    _pool_edge_label  = "/2"
    _global_pool_label = "G" + "".join(w[0].upper() for w in pool_label.split())

    _graph_attr = dict(
        rankdir=rankdir,
        nodesep="0.3",
        ranksep=_ranksep,
        fontname="Helvetica",
        dpi=str(dpi),
    )
    if output_size is not None:
        _graph_attr["size"] = output_size

    g = graphviz.Digraph(
        name="architecture",
        graph_attr=_graph_attr,
        node_attr=dict(
            fontname="Helvetica",
            fontsize="13",
            style="rounded,filled",
            penwidth="1.5",
            margin="0.15,0.10",
        ),
        edge_attr=dict(
            arrowsize="0.7",
            penwidth="1.2",
        ),
    )

    # Colour scheme (matches the visual: purple DAGs, gray pooling, teal linears)
    COLORS = {
        "dag":    ("#EEEDFE", "#534AB7", "#3C3489"),
        "linear": ("#E1F5EE", "#0F6E56", "#085041"),
    }

    dag_widths = ["1.2"]*4 if rankdir == "LR" else ["1.8"]*4

    # Phantom start node — renders as a bare incoming arrow with no source box
    g.node("_start", label="", shape="none", width="0", height="0", margin="0")
    prev = "_start"

    for i, (cfg, path) in enumerate(zip(dag_configs, dag_paths)):
        dag_name = dag_label(i, cfg)
        scale    = 2 ** i
        w        = dag_widths[min(i, len(dag_widths) - 1)]

        ch      = f"C{i + 1}"
        spatial = f"H/{scale} × W/{scale} × {ch}" if scale > 1 else f"H × W × {ch}"

        dag_id = f"dag_{i}"
        g.node(
            dag_id,
            label=f"{dag_name}\n{spatial}",
            fillcolor=COLORS["dag"][0],
            color=COLORS["dag"][1],
            fontcolor=COLORS["dag"][2],
            width=w,
            height=dag_h,
            shape="box",
            fontname="Helvetica",
            fontsize="13",
            style="rounded,filled",
            penwidth="1.5",
            margin="0.15,0.10",
        )

        # Pool annotated compactly on the edge; first DAG has no incoming pool.
        if i == 0:
            g.edge(prev, dag_id)
        else:
            g.edge(prev, dag_id,
                   label=_pool_edge_label,
                   fontname="Helvetica",
                   fontsize=_fontsize_edge,
                   fontcolor="#888780",
                   color="#888780",
            )
        prev = dag_id

    # Two linear layers
    n = len(dag_configs)
    last_ch = f"C{n}"
    lin_labels = [
        (f"linear 1", f"{last_ch} → {linear_sizes[0]}"),
        (f"linear 2", f"{linear_sizes[0]} → {linear_sizes[1]}"),
    ]

    for j, (lname, ldim) in enumerate(lin_labels):
        lin_id = f"linear_{j}"
        g.node(
            lin_id,
            label=f"{lname}\n{ldim}",
            fillcolor=COLORS["linear"][0],
            color=COLORS["linear"][1],
            fontcolor=COLORS["linear"][2],
            width=lin_w,
            height=lin_h,
            shape="box",
            fontname="Helvetica",
            fontsize="12",
            style="filled",
            penwidth="1.5",
            margin="0.12,0.10",
        )
        if j == 0:
            g.edge(prev, lin_id,
                   label=_global_pool_label,
                   fontname="Helvetica",
                   fontsize=_fontsize_edge,
                   fontcolor="#888780",
                   color="#888780",
            )
        else:
            g.edge(prev, lin_id)
        prev = lin_id

    return g


def main():
    parser = argparse.ArgumentParser(
        description="Generate a ResNet/VGG-style architecture diagram from DAG JSON configs."
    )
    parser.add_argument("configs", nargs="+", help="Paths to DAG JSON config files (in order)")
    parser.add_argument("--output", "-o", default="temp/architecture",
                        help="Output filename without extension (default: architecture)")
    parser.add_argument("--format", "-f", default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="Output format (default: pdf)")
    parser.add_argument("--rankdir", default="LR",
                        choices=["LR", "TB"],
                        help="Graph direction: LR (left→right) or TB (top→bottom)")
    parser.add_argument("--linear", default="D1,D2",
                        help="Comma-separated output sizes of the two linear layers (default: D1,D2)")
    parser.add_argument("--pool-label", default="avg pool",
                        help="Label for pooling nodes (default: avg pool)")
    parser.add_argument("--size", default=None,
                        help='Output bounding box as "W,H" in inches (e.g. "5.5,1.5"). '
                             'Defaults to "5.5,1.5" for LR (NeurIPS text width) and "3,5" for TB.')
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution for raster output (default: 300)")
    parser.add_argument("--view", action="store_true",
                        help="Open the rendered file after generation")
    args = parser.parse_args()

    linear_sizes = [s.strip() for s in args.linear.split(",")]
    if len(linear_sizes) != 2:
        parser.error("--linear must be exactly two comma-separated values, e.g. --linear 512,10")

    output_size = args.size
    if output_size is None:
        output_size = "5.5,1.5" if args.rankdir == "LR" else "3,5"

    dag_configs = []
    for p in args.configs:
        dag_configs.append(load_dag(p))

    g = build_architecture_graph(
        dag_configs,
        args.configs,
        linear_sizes,
        args.pool_label,
        rankdir=args.rankdir,
        output_size=output_size,
        dpi=args.dpi,
    )
    g.format = args.format
    rendered = g.render(args.output, view=args.view, cleanup=True)
    print(f"Saved: {rendered}")


if __name__ == "__main__":
    main()