from langchain.tools import Tool
from typing import Dict, List, Any, Optional


class GraphVisualizationTool:
    """Tool for creating Mermaid diagram visualizations of graph structures."""

    def __init__(self):
        self.node_styles = {
            'default': 'fill:#bbf,stroke:#333,stroke-width:1px',
            'router': 'fill:#f9f,stroke:#333,stroke-width:2px',
            'end': 'fill:#f96,stroke:#333,stroke-width:2px',
            'start': 'fill:#9f9,stroke:#333,stroke-width:2px'
        }

    def create_mermaid_diagram(
            self,
            nodes: List[str],
            edges: Dict[str, List[str]],
            node_types: Optional[Dict[str, str]] = None,
            title: str = "Graph Structure"
    ) -> str:
        """
        Create a Mermaid diagram representation of a graph.

        Args:
            nodes: List of node names
            edges: Dictionary mapping source nodes to lists of target nodes
            node_types: Optional dictionary mapping nodes to their types
            title: Title of the diagram

        Returns:
            Mermaid diagram string
        """
        mermaid_lines = ["graph TD"]

        # Add title as comment
        mermaid_lines.append(f"    %% {title}")
        mermaid_lines.append("")

        # Add nodes
        for node in nodes:
            node_label = node.replace(" ", "_")
            if node.upper() == "START":
                mermaid_lines.append(f"    {node_label}(({node}))")
            elif node.upper() == "END":
                mermaid_lines.append(f"    {node_label}(({node}))")
            elif "ROUTER" in node.upper():
                mermaid_lines.append(f"    {node_label}{{{node}}}")
            else:
                mermaid_lines.append(f"    {node_label}[{node}]")

        # Add edges
        for source, targets in edges.items():
            source_label = source.replace(" ", "_")
            for target in targets:
                target_label = target.replace(" ", "_")
                mermaid_lines.append(f"    {source_label} --> {target_label}")

        # Add styling
        mermaid_lines.append("")
        mermaid_lines.append("    %% Styling")
        for style_name, style_def in self.node_styles.items():
            mermaid_lines.append(f"    classDef {style_name} {style_def};")

        # Apply styles
        mermaid_lines.append("")
        if node_types:
            for node_type, nodes_of_type in node_types.items():
                if nodes_of_type:
                    nodes_str = ",".join(node.replace(" ", "_") for node in nodes_of_type)
                    mermaid_lines.append(f"    class {nodes_str} {node_type};")

        return "\n".join(mermaid_lines)

    def parse_graph_structure(self, graph_builder: 'StateGraph') -> Dict[str, Any]:
        """
        Parse a StateGraph object to extract nodes and edges.

        Args:
            graph_builder: StateGraph object to parse

        Returns:
            Dictionary containing nodes, edges, and node types
        """
        nodes = list(graph_builder.nodes.keys())
        if "START" not in nodes:
            nodes.insert(0, "START")
        if "END" not in nodes:
            nodes.append("END")

        edges = {}
        node_types = {
            'start': ['START'],
            'end': ['END'],
            'router': [n for n in nodes if 'ROUTER' in n.upper()],
            'default': [n for n in nodes if n not in ['START', 'END'] and 'ROUTER' not in n.upper()]
        }

        # Extract edges from the graph
        for node in graph_builder.nodes:
            edges[node] = []
            if hasattr(graph_builder, 'edges') and node in graph_builder.edges:
                for edge in graph_builder.edges[node]:
                    if isinstance(edge, dict):
                        edges[node].extend(edge.values())
                    else:
                        edges[node].append(edge)

        return {
            'nodes': nodes,
            'edges': edges,
            'node_types': node_types
        }


def create_graph_visualization_tool() -> Tool:
    """Create a Tool instance for graph visualization."""
    viz_tool = GraphVisualizationTool()

    def visualize_graph(graph_builder: 'StateGraph', title: str = "Graph Structure") -> str:
        """Tool function to create Mermaid diagram from graph structure."""
        try:
            structure = viz_tool.parse_graph_structure(graph_builder)
            return viz_tool.create_mermaid_diagram(
                nodes=structure['nodes'],
                edges=structure['edges'],
                node_types=structure['node_types'],
                title=title
            )
        except Exception as e:
            return f"Error creating graph visualization: {str(e)}"

    return Tool(
        name="visualize_graph",
        description="Creates a Mermaid diagram visualization of a graph structure. Input should be a StateGraph object.",
        func=visualize_graph
    )
