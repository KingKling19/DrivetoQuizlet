#!/usr/bin/env python3
"""
Lesson Relationship Visualizer

Generates visual maps of lesson relationships and content correlation networks for the cross-lesson context system.
Provides insights into curriculum structure and concept relationships.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pickle
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("ERROR: matplotlib and networkx are required. pip install matplotlib networkx", file=sys.stderr)
    raise

try:
    from sklearn.manifold import MDS
    from sklearn.decomposition import PCA
except ImportError:
    print("ERROR: scikit-learn is required. pip install scikit-learn", file=sys.stderr)
    raise


class LessonRelationshipVisualizer:
    """Visualizes lesson relationships and content correlation networks."""
    
    def __init__(self, config_dir: Path = Path("config"), output_dir: Path = Path("outputs")):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load cross-lesson data
        self.cross_lesson_data = self.load_cross_lesson_data()
        
        # Visualization configuration
        self.viz_config = {
            "figure_size": (16, 12),
            "dpi": 300,
            "node_size": 2000,
            "font_size": 10,
            "edge_width_range": (1, 5),
            "color_scheme": "viridis",
            "max_edges": 20,
            "min_similarity_threshold": 0.2
        }
    
    def load_cross_lesson_data(self) -> Dict[str, Any]:
        """Load cross-lesson analysis data."""
        data = {
            "content_index": {},
            "semantic_embeddings": {},
            "lesson_relationships": {},
            "cross_references": {}
        }
        
        try:
            # Load content index
            index_file = self.config_dir / "lesson_content_index.json"
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    data["content_index"] = json.load(f)
            
            # Load semantic embeddings
            embeddings_file = self.config_dir / "semantic_embeddings.pkl"
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    data["semantic_embeddings"] = pickle.load(f)
            
            # Load lesson relationships
            relationships_file = self.config_dir / "lesson_relationships_analysis.json"
            if relationships_file.exists():
                with open(relationships_file, 'r', encoding='utf-8') as f:
                    data["lesson_relationships"] = json.load(f)
            
            # Load cross-references
            cross_refs_file = self.config_dir / "cross_references.json"
            if cross_refs_file.exists():
                with open(cross_refs_file, 'r', encoding='utf-8') as f:
                    data["cross_references"] = json.load(f)
            
            print(f"✓ Loaded cross-lesson data: {len(data['content_index'])} lessons indexed")
            return data
        except Exception as e:
            print(f"⚠️  Could not load cross-lesson data: {e}")
            return data
    
    def create_relationship_network(self) -> nx.Graph:
        """Create a network graph from lesson relationships."""
        G = nx.Graph()
        
        # Add nodes (lessons)
        for lesson_id, lesson_data in self.cross_lesson_data["content_index"].items():
            lesson_name = lesson_data.get("lesson_name", lesson_id)
            G.add_node(lesson_id, name=lesson_name, data=lesson_data)
        
        # Add edges (relationships)
        relationships = self.cross_lesson_data["lesson_relationships"]
        for source_lesson, targets in relationships.items():
            for target_lesson, relationship_data in targets.items():
                if target_lesson in G.nodes():
                    similarity = relationship_data.get("similarity_score", 0)
                    if similarity >= self.viz_config["min_similarity_threshold"]:
                        G.add_edge(source_lesson, target_lesson, 
                                 weight=similarity,
                                 relationship_type=relationship_data.get("relationship_type", "related"))
        
        return G
    
    def create_concept_correlation_network(self) -> nx.Graph:
        """Create a network graph showing concept correlations across lessons."""
        G = nx.Graph()
        
        # Extract concepts and their relationships
        concept_lessons = {}
        concept_relationships = {}
        
        for lesson_id, lesson_data in self.cross_lesson_data["content_index"].items():
            concepts = lesson_data.get("key_concepts", [])
            for concept in concepts:
                if concept not in concept_lessons:
                    concept_lessons[concept] = []
                concept_lessons[concept].append(lesson_id)
        
        # Add concept nodes
        for concept, lessons in concept_lessons.items():
            if len(lessons) > 1:  # Only concepts that appear in multiple lessons
                G.add_node(concept, 
                          lessons=lessons, 
                          frequency=len(lessons),
                          type="concept")
        
        # Add lesson nodes
        for lesson_id, lesson_data in self.cross_lesson_data["content_index"].items():
            lesson_name = lesson_data.get("lesson_name", lesson_id)
            G.add_node(lesson_id, 
                      name=lesson_name, 
                      type="lesson",
                      concepts=lesson_data.get("key_concepts", []))
        
        # Add edges between concepts and lessons
        for concept, lessons in concept_lessons.items():
            if concept in G.nodes():
                for lesson_id in lessons:
                    if lesson_id in G.nodes():
                        G.add_edge(concept, lesson_id, type="concept_lesson")
        
        # Add edges between related concepts
        for concept1 in G.nodes():
            if G.nodes[concept1].get("type") == "concept":
                for concept2 in G.nodes():
                    if (concept2 != concept1 and 
                        G.nodes[concept2].get("type") == "concept"):
                        # Calculate concept similarity based on shared lessons
                        lessons1 = set(G.nodes[concept1]["lessons"])
                        lessons2 = set(G.nodes[concept2]["lessons"])
                        shared_lessons = lessons1.intersection(lessons2)
                        if shared_lessons:
                            similarity = len(shared_lessons) / max(len(lessons1), len(lessons2))
                            if similarity >= 0.3:  # Only show significant relationships
                                G.add_edge(concept1, concept2, 
                                         weight=similarity,
                                         type="concept_concept",
                                         shared_lessons=list(shared_lessons))
        
        return G
    
    def create_semantic_space_visualization(self) -> Tuple[np.ndarray, List[str]]:
        """Create 2D semantic space visualization using dimensionality reduction."""
        lesson_ids = list(self.cross_lesson_data["content_index"].keys())
        embeddings = []
        valid_lesson_ids = []
        
        # Collect embeddings
        for lesson_id in lesson_ids:
            embedding = self.cross_lesson_data["semantic_embeddings"].get(lesson_id)
            if embedding is not None and len(embedding) > 0:
                embeddings.append(embedding)
                valid_lesson_ids.append(lesson_id)
        
        if not embeddings:
            print("⚠️  No semantic embeddings found for visualization")
            return np.array([]), []
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Apply dimensionality reduction
        if embeddings_array.shape[1] > 2:
            # Use MDS for better preservation of distances
            mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
            coords_2d = mds.fit_transform(embeddings_array)
        else:
            coords_2d = embeddings_array
        
        return coords_2d, valid_lesson_ids
    
    def visualize_lesson_relationships(self, output_file: Optional[str] = None) -> str:
        """Create and save lesson relationship network visualization."""
        G = self.create_relationship_network()
        
        if len(G.nodes()) == 0:
            print("⚠️  No lesson relationships found for visualization")
            return ""
        
        # Create figure
        plt.figure(figsize=self.viz_config["figure_size"], dpi=self.viz_config["dpi"])
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Prepare edge weights and colors
        edges = G.edges(data=True)
        edge_weights = [data.get('weight', 0) for _, _, data in edges]
        edge_colors = [data.get('weight', 0) for _, _, data in edges]
        
        # Sort edges by weight and limit number
        edge_data = list(zip(edges, edge_weights))
        edge_data.sort(key=lambda x: x[1], reverse=True)
        edge_data = edge_data[:self.viz_config["max_edges"]]
        
        edges_filtered = [edge for edge, _ in edge_data]
        weights_filtered = [weight for _, weight in edge_data]
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                              edgelist=edges_filtered,
                              width=weights_filtered,
                              edge_color=weights_filtered,
                              edge_cmap=plt.cm.viridis,
                              alpha=0.7)
        
        # Draw nodes
        node_labels = {node: G.nodes[node].get('name', node) for node in G.nodes()}
        nx.draw_networkx_nodes(G, pos, 
                              node_size=self.viz_config["node_size"],
                              node_color='lightblue',
                              alpha=0.8)
        nx.draw_networkx_labels(G, pos, 
                               labels=node_labels,
                               font_size=self.viz_config["font_size"])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(vmin=min(weights_filtered), 
                                                    vmax=max(weights_filtered)))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Relationship Strength', rotation=270, labelpad=15)
        
        plt.title('Lesson Relationship Network', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Save or show
        if output_file:
            output_path = self.output_dir / output_file
            plt.savefig(output_path, bbox_inches='tight', dpi=self.viz_config["dpi"])
            plt.close()
            print(f"✓ Lesson relationship visualization saved to: {output_path}")
            return str(output_path)
        else:
            plt.show()
            return ""
    
    def visualize_concept_correlations(self, output_file: Optional[str] = None) -> str:
        """Create and save concept correlation network visualization."""
        G = self.create_concept_correlation_network()
        
        if len(G.nodes()) == 0:
            print("⚠️  No concept correlations found for visualization")
            return ""
        
        # Create figure
        plt.figure(figsize=self.viz_config["figure_size"], dpi=self.viz_config["dpi"])
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        
        # Separate nodes by type
        concept_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'concept']
        lesson_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'lesson']
        
        # Draw edges
        concept_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'concept_concept']
        lesson_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'concept_lesson']
        
        # Draw concept-concept edges
        if concept_edges:
            edge_weights = [G[u][v].get('weight', 0) for u, v in concept_edges]
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=concept_edges,
                                  width=edge_weights,
                                  edge_color=edge_weights,
                                  edge_cmap=plt.cm.Reds,
                                  alpha=0.6)
        
        # Draw concept-lesson edges
        if lesson_edges:
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=lesson_edges,
                                  width=1,
                                  edge_color='gray',
                                  alpha=0.3)
        
        # Draw concept nodes
        if concept_nodes:
            concept_sizes = [G.nodes[n].get('frequency', 1) * 500 for n in concept_nodes]
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=concept_nodes,
                                  node_size=concept_sizes,
                                  node_color='red',
                                  alpha=0.7)
        
        # Draw lesson nodes
        if lesson_nodes:
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=lesson_nodes,
                                  node_size=1000,
                                  node_color='blue',
                                  alpha=0.7)
        
        # Draw labels
        concept_labels = {n: n for n in concept_nodes}
        lesson_labels = {n: G.nodes[n].get('name', n) for n in lesson_nodes}
        all_labels = {**concept_labels, **lesson_labels}
        
        nx.draw_networkx_labels(G, pos, 
                               labels=all_labels,
                               font_size=8)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='red', alpha=0.7, label='Concepts'),
            mpatches.Patch(color='blue', alpha=0.7, label='Lessons')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.title('Concept Correlation Network', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Save or show
        if output_file:
            output_path = self.output_dir / output_file
            plt.savefig(output_path, bbox_inches='tight', dpi=self.viz_config["dpi"])
            plt.close()
            print(f"✓ Concept correlation visualization saved to: {output_path}")
            return str(output_path)
        else:
            plt.show()
            return ""
    
    def visualize_semantic_space(self, output_file: Optional[str] = None) -> str:
        """Create and save semantic space visualization."""
        coords_2d, lesson_ids = self.create_semantic_space_visualization()
        
        if len(coords_2d) == 0:
            print("⚠️  No semantic data found for visualization")
            return ""
        
        # Create figure
        plt.figure(figsize=(12, 10), dpi=self.viz_config["dpi"])
        
        # Plot points
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                   s=200, alpha=0.7, c='blue')
        
        # Add labels
        for i, lesson_id in enumerate(lesson_ids):
            lesson_name = self.cross_lesson_data["content_index"].get(lesson_id, {}).get("lesson_name", lesson_id)
            plt.annotate(lesson_name, 
                        (coords_2d[i, 0], coords_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.title('Semantic Space of Lessons', fontsize=16, fontweight='bold')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        
        # Save or show
        if output_file:
            output_path = self.output_dir / output_file
            plt.savefig(output_path, bbox_inches='tight', dpi=self.viz_config["dpi"])
            plt.close()
            print(f"✓ Semantic space visualization saved to: {output_path}")
            return str(output_path)
        else:
            plt.show()
            return ""
    
    def generate_curriculum_insights(self) -> Dict[str, Any]:
        """Generate insights about curriculum structure and relationships."""
        insights = {
            "total_lessons": len(self.cross_lesson_data["content_index"]),
            "total_concepts": 0,
            "concept_distribution": {},
            "relationship_types": {},
            "strongest_relationships": [],
            "isolated_lessons": [],
            "concept_clusters": {},
            "recommendations": []
        }
        
        # Analyze concept distribution
        concept_counts = {}
        for lesson_data in self.cross_lesson_data["content_index"].values():
            concepts = lesson_data.get("key_concepts", [])
            insights["total_concepts"] += len(concepts)
            for concept in concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        insights["concept_distribution"] = dict(sorted(concept_counts.items(), 
                                                      key=lambda x: x[1], reverse=True)[:10])
        
        # Analyze relationship types
        for source_lesson, targets in self.cross_lesson_data["lesson_relationships"].items():
            for target_lesson, relationship_data in targets.items():
                rel_type = relationship_data.get("relationship_type", "related")
                insights["relationship_types"][rel_type] = insights["relationship_types"].get(rel_type, 0) + 1
        
        # Find strongest relationships
        all_relationships = []
        for source_lesson, targets in self.cross_lesson_data["lesson_relationships"].items():
            for target_lesson, relationship_data in targets.items():
                similarity = relationship_data.get("similarity_score", 0)
                all_relationships.append({
                    "source": source_lesson,
                    "target": target_lesson,
                    "similarity": similarity,
                    "type": relationship_data.get("relationship_type", "related")
                })
        
        insights["strongest_relationships"] = sorted(all_relationships, 
                                                   key=lambda x: x["similarity"], 
                                                   reverse=True)[:5]
        
        # Find isolated lessons
        lesson_ids = set(self.cross_lesson_data["content_index"].keys())
        connected_lessons = set()
        for source_lesson, targets in self.cross_lesson_data["lesson_relationships"].items():
            connected_lessons.add(source_lesson)
            connected_lessons.update(targets.keys())
        
        insights["isolated_lessons"] = list(lesson_ids - connected_lessons)
        
        # Generate recommendations
        if insights["isolated_lessons"]:
            insights["recommendations"].append(
                f"Consider adding relationships for {len(insights['isolated_lessons'])} isolated lessons"
            )
        
        if len(insights["concept_distribution"]) > 0:
            most_common_concept = list(insights["concept_distribution"].keys())[0]
            insights["recommendations"].append(
                f"Concept '{most_common_concept}' appears frequently - consider creating a dedicated lesson"
            )
        
        return insights
    
    def create_comprehensive_visualization(self, output_prefix: str = "curriculum_analysis") -> Dict[str, str]:
        """Create all visualizations and generate comprehensive analysis."""
        results = {}
        
        print("🎨 Creating comprehensive curriculum visualization...")
        
        # Create individual visualizations
        results["lesson_relationships"] = self.visualize_lesson_relationships(
            f"{output_prefix}_relationships.png"
        )
        
        results["concept_correlations"] = self.visualize_concept_correlations(
            f"{output_prefix}_concepts.png"
        )
        
        results["semantic_space"] = self.visualize_semantic_space(
            f"{output_prefix}_semantic.png"
        )
        
        # Generate insights
        insights = self.generate_curriculum_insights()
        
        # Save insights
        insights_file = self.output_dir / f"{output_prefix}_insights.json"
        with open(insights_file, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        results["insights"] = str(insights_file)
        
        print(f"✓ Comprehensive visualization complete:")
        print(f"  - Lesson relationships: {results['lesson_relationships']}")
        print(f"  - Concept correlations: {results['concept_correlations']}")
        print(f"  - Semantic space: {results['semantic_space']}")
        print(f"  - Insights: {results['insights']}")
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Lesson Relationship Visualizer")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--output-prefix", default="curriculum_analysis", help="Output file prefix")
    parser.add_argument("--visualization-type", choices=["all", "relationships", "concepts", "semantic"], 
                       default="all", help="Type of visualization to create")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = LessonRelationshipVisualizer(Path(args.config_dir), Path(args.output_dir))
    
    if args.visualization_type == "all":
        results = visualizer.create_comprehensive_visualization(args.output_prefix)
        
        # Print insights summary
        insights_file = Path(results["insights"])
        if insights_file.exists():
            with open(insights_file, 'r', encoding='utf-8') as f:
                insights = json.load(f)
            
            print(f"\n📊 Curriculum Insights Summary:")
            print(f"  Total Lessons: {insights['total_lessons']}")
            print(f"  Total Concepts: {insights['total_concepts']}")
            print(f"  Relationship Types: {insights['relationship_types']}")
            print(f"  Isolated Lessons: {len(insights['isolated_lessons'])}")
            
            if insights['recommendations']:
                print(f"  Recommendations:")
                for rec in insights['recommendations']:
                    print(f"    - {rec}")
    
    elif args.visualization_type == "relationships":
        visualizer.visualize_lesson_relationships(f"{args.output_prefix}_relationships.png")
    
    elif args.visualization_type == "concepts":
        visualizer.visualize_concept_correlations(f"{args.output_prefix}_concepts.png")
    
    elif args.visualization_type == "semantic":
        visualizer.visualize_semantic_space(f"{args.output_prefix}_semantic.png")


if __name__ == "__main__":
    main()
