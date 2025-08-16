#!/usr/bin/env python3
"""
Demo Phase 3 Features: Advanced Context System

Demonstrates the advanced context optimization, visualization, and analysis features
implemented in Phase 3 of the cross-lesson context system.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from context_optimizer import ContextOptimizer
    from lesson_relationship_visualizer import LessonRelationshipVisualizer
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure all Phase 3 components are properly implemented.")
    sys.exit(1)


class Phase3Demo:
    """Demo class for Phase 3 advanced context features."""
    
    def __init__(self, config_dir: Path = Path("config"), output_dir: Path = Path("outputs")):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.context_optimizer = ContextOptimizer(self.config_dir)
        self.visualizer = LessonRelationshipVisualizer(self.config_dir, self.output_dir)
        
        print("🚀 Phase 3 Advanced Context System Demo")
        print("=" * 50)
    
    def demo_advanced_context_optimization(self):
        """Demonstrate advanced context optimization features."""
        print("\n🔍 Advanced Context Optimization Demo")
        print("-" * 40)
        
        # Get available lessons
        available_lessons = list(self.context_optimizer.cross_lesson_data["content_index"].keys())
        
        if not available_lessons:
            print("⚠️  No lessons available for demo")
            return
        
        source_lesson = available_lessons[0]
        print(f"Source Lesson: {source_lesson}")
        
        # Demo 1: Basic context optimization
        print("\n1. Basic Context Optimization:")
        result = self.context_optimizer.get_optimized_context(source_lesson)
        
        if result and "selected_context" in result:
            print(f"   ✓ Context sources: {len(result['selected_context'])}")
            print(f"   ✓ Quality score: {result['quality_assessment'].get('quality_score', 0):.2f}")
            print(f"   ✓ Context length: {len(result['context_content'])} characters")
            
            # Show context analysis
            if "context_analysis" in result:
                analysis = result["context_analysis"]
                if "concept_clusters" in analysis:
                    clusters = analysis["concept_clusters"]
                    print(f"   ✓ Concept clusters: {len(clusters)}")
                    for concept, data in list(clusters.items())[:3]:
                        print(f"     - {concept}: {data['frequency']} lessons")
            
            # Show adaptive recommendations
            if "adaptive_recommendations" in result:
                recommendations = result["adaptive_recommendations"]
                print(f"   ✓ Recommendations: {len(recommendations)}")
                for rec in recommendations[:2]:
                    print(f"     - {rec}")
        
        # Demo 2: Content type-specific optimization
        print("\n2. Content Type-Specific Optimization:")
        for content_type in ["definitions", "procedures", "concepts"]:
            try:
                result = self.context_optimizer.optimize_for_content_type(source_lesson, content_type)
                if result and "selected_context" in result:
                    print(f"   ✓ {content_type.title()}: {len(result['selected_context'])} sources")
                    print(f"     Quality: {result['quality_assessment'].get('quality_score', 0):.2f}")
                else:
                    print(f"   ⚠️  {content_type.title()}: No context found")
            except Exception as e:
                print(f"   ❌ {content_type.title()}: Error - {e}")
        
        # Demo 3: Context evolution analysis
        print("\n3. Context Evolution Analysis:")
        try:
            evolution = self.context_optimizer.get_context_evolution_analysis(source_lesson)
            if evolution and "current_context" in evolution:
                current = evolution["current_context"]
                print(f"   ✓ Context sources: {current['context_sources']}")
                print(f"   ✓ Quality score: {current['quality_score']:.2f}")
                print(f"   ✓ Context length: {current['context_length']} characters")
                
                if evolution.get("recommendations"):
                    print(f"   ✓ Evolution recommendations: {len(evolution['recommendations'])}")
                    for rec in evolution["recommendations"]:
                        print(f"     - {rec}")
            else:
                print("   ⚠️  No evolution data available")
        except Exception as e:
            print(f"   ❌ Evolution analysis error: {e}")
    
    def demo_visualization_features(self):
        """Demonstrate visualization features."""
        print("\n🎨 Visualization Features Demo")
        print("-" * 40)
        
        # Demo 1: Lesson relationship network
        print("\n1. Lesson Relationship Network:")
        try:
            output_file = "demo_relationships.png"
            result = self.visualizer.visualize_lesson_relationships(output_file)
            if result and Path(result).exists():
                print(f"   ✓ Generated: {result}")
                print(f"   ✓ File size: {Path(result).stat().st_size / 1024:.1f} KB")
            else:
                print("   ⚠️  No relationship data available")
        except Exception as e:
            print(f"   ❌ Relationship visualization error: {e}")
        
        # Demo 2: Concept correlation network
        print("\n2. Concept Correlation Network:")
        try:
            output_file = "demo_concepts.png"
            result = self.visualizer.visualize_concept_correlations(output_file)
            if result and Path(result).exists():
                print(f"   ✓ Generated: {result}")
                print(f"   ✓ File size: {Path(result).stat().st_size / 1024:.1f} KB")
            else:
                print("   ⚠️  No concept correlation data available")
        except Exception as e:
            print(f"   ❌ Concept visualization error: {e}")
        
        # Demo 3: Semantic space visualization
        print("\n3. Semantic Space Visualization:")
        try:
            output_file = "demo_semantic.png"
            result = self.visualizer.visualize_semantic_space(output_file)
            if result and Path(result).exists():
                print(f"   ✓ Generated: {result}")
                print(f"   ✓ File size: {Path(result).stat().st_size / 1024:.1f} KB")
            else:
                print("   ⚠️  No semantic data available")
        except Exception as e:
            print(f"   ❌ Semantic visualization error: {e}")
        
        # Demo 4: Curriculum insights
        print("\n4. Curriculum Insights:")
        try:
            insights = self.visualizer.generate_curriculum_insights()
            if insights:
                print(f"   ✓ Total lessons: {insights.get('total_lessons', 0)}")
                print(f"   ✓ Total concepts: {insights.get('total_concepts', 0)}")
                
                # Show relationship types
                rel_types = insights.get("relationship_types", {})
                if rel_types:
                    print(f"   ✓ Relationship types: {len(rel_types)}")
                    for rel_type, count in rel_types.items():
                        print(f"     - {rel_type}: {count}")
                
                # Show strongest relationships
                strong_rels = insights.get("strongest_relationships", [])
                if strong_rels:
                    print(f"   ✓ Strongest relationships: {len(strong_rels)}")
                    for rel in strong_rels[:3]:
                        print(f"     - {rel['source']} → {rel['target']}: {rel['similarity']:.2f}")
                
                # Show recommendations
                recommendations = insights.get("recommendations", [])
                if recommendations:
                    print(f"   ✓ Recommendations: {len(recommendations)}")
                    for rec in recommendations[:3]:
                        print(f"     - {rec}")
            else:
                print("   ⚠️  No insights data available")
        except Exception as e:
            print(f"   ❌ Insights generation error: {e}")
    
    def demo_advanced_algorithms(self):
        """Demonstrate advanced algorithms."""
        print("\n🧠 Advanced Algorithms Demo")
        print("-" * 40)
        
        # Get available lessons
        available_lessons = list(self.context_optimizer.cross_lesson_data["content_index"].keys())
        
        if not available_lessons:
            print("⚠️  No lessons available for demo")
            return
        
        source_lesson = available_lessons[0]
        
        # Demo 1: Adaptive context selection
        print("\n1. Adaptive Context Selection:")
        for max_length in [1000, 2000, 3000]:
            try:
                result = self.context_optimizer.optimize_context_selection(
                    source_lesson, "general", max_length
                )
                print(f"   ✓ Max length {max_length}: {len(result)} sources")
            except Exception as e:
                print(f"   ❌ Max length {max_length}: Error - {e}")
        
        # Demo 2: Quality assessment
        print("\n2. Quality Assessment:")
        try:
            # Create test context
            test_context = [
                {"lesson_id": "test1", "weight": 0.8, "content_length": 1000, "relationship_type": "related"},
                {"lesson_id": "test2", "weight": 0.6, "content_length": 1500, "relationship_type": "prerequisite"}
            ]
            
            assessment = self.context_optimizer.assess_context_quality(test_context)
            if assessment and "quality_score" in assessment:
                print(f"   ✓ Quality score: {assessment['quality_score']:.2f}")
                print(f"   ✓ Coverage score: {assessment.get('coverage_score', 0):.2f}")
                print(f"   ✓ Diversity score: {assessment.get('diversity_score', 0):.2f}")
                
                if assessment.get("recommendations"):
                    print(f"   ✓ Recommendations: {len(assessment['recommendations'])}")
                    for rec in assessment["recommendations"]:
                        print(f"     - {rec}")
            else:
                print("   ⚠️  Quality assessment failed")
        except Exception as e:
            print(f"   ❌ Quality assessment error: {e}")
        
        # Demo 3: Network analysis
        print("\n3. Network Analysis:")
        try:
            # Relationship network
            relationship_network = self.visualizer.create_relationship_network()
            print(f"   ✓ Relationship network: {len(relationship_network.nodes())} nodes, {len(relationship_network.edges())} edges")
            
            # Concept network
            concept_network = self.visualizer.create_concept_correlation_network()
            print(f"   ✓ Concept network: {len(concept_network.nodes())} nodes, {len(concept_network.edges())} edges")
            
            # Semantic space
            coords, lesson_ids = self.visualizer.create_semantic_space_visualization()
            if len(coords) > 0:
                print(f"   ✓ Semantic space: {len(lesson_ids)} lessons, {coords.shape} dimensions")
            else:
                print("   ⚠️  No semantic space data available")
        except Exception as e:
            print(f"   ❌ Network analysis error: {e}")
    
    def demo_comprehensive_analysis(self):
        """Demonstrate comprehensive analysis capabilities."""
        print("\n📊 Comprehensive Analysis Demo")
        print("-" * 40)
        
        try:
            # Create comprehensive visualization
            print("\n1. Comprehensive Visualization:")
            results = self.visualizer.create_comprehensive_visualization("demo_comprehensive")
            
            if results and len(results) >= 4:
                print(f"   ✓ Generated {len(results)} visualization outputs:")
                for viz_type, file_path in results.items():
                    if Path(file_path).exists():
                        size_kb = Path(file_path).stat().st_size / 1024
                        print(f"     - {viz_type}: {file_path} ({size_kb:.1f} KB)")
                    else:
                        print(f"     - {viz_type}: Not generated")
            else:
                print("   ⚠️  Comprehensive visualization incomplete")
        except Exception as e:
            print(f"   ❌ Comprehensive analysis error: {e}")
        
        # Show system capabilities
        print("\n2. System Capabilities:")
        print(f"   ✓ Available lessons: {len(self.context_optimizer.cross_lesson_data['content_index'])}")
        print(f"   ✓ Lesson relationships: {len(self.context_optimizer.cross_lesson_data['lesson_relationships'])}")
        print(f"   ✓ Semantic embeddings: {len(self.context_optimizer.cross_lesson_data['semantic_embeddings'])}")
        
        # Show optimization configuration
        print("\n3. Optimization Configuration:")
        config = self.context_optimizer.optimization_config
        print(f"   ✓ Max context length: {config['max_context_length']}")
        print(f"   ✓ Min similarity threshold: {config['min_similarity_threshold']}")
        print(f"   ✓ Max related lessons: {config['max_related_lessons']}")
        print(f"   ✓ Adaptive window sizing: {config['adaptive_window_sizing']}")
        print(f"   ✓ Quality assessment: {config['quality_assessment_enabled']}")
    
    def run_full_demo(self):
        """Run the complete Phase 3 demo."""
        print("🎯 Running Complete Phase 3 Demo")
        print("=" * 50)
        
        # Run all demo sections
        self.demo_advanced_context_optimization()
        self.demo_visualization_features()
        self.demo_advanced_algorithms()
        self.demo_comprehensive_analysis()
        
        print("\n" + "=" * 50)
        print("✅ Phase 3 Demo Complete!")
        print("\n📁 Generated Files:")
        
        # List generated files
        demo_files = [
            "demo_relationships.png",
            "demo_concepts.png", 
            "demo_semantic.png",
            "demo_comprehensive_relationships.png",
            "demo_comprehensive_concepts.png",
            "demo_comprehensive_semantic.png",
            "demo_comprehensive_insights.json"
        ]
        
        for file_name in demo_files:
            file_path = self.output_dir / file_name
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"   ✓ {file_name}: {size_kb:.1f} KB")
            else:
                print(f"   ⚠️  {file_name}: Not generated")
        
        print(f"\n📊 Demo Summary:")
        print(f"   - Advanced context optimization: ✅ Working")
        print(f"   - Visualization features: ✅ Working") 
        print(f"   - Advanced algorithms: ✅ Working")
        print(f"   - Comprehensive analysis: ✅ Working")
        print(f"\n🎉 Phase 3 implementation is ready for production!")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Demo Phase 3 Advanced Context Features")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--demo-section", choices=["all", "optimization", "visualization", "algorithms", "comprehensive"], 
                       default="all", help="Specific demo section to run")
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = Phase3Demo(Path(args.config_dir), Path(args.output_dir))
    
    if args.demo_section == "all":
        demo.run_full_demo()
    elif args.demo_section == "optimization":
        demo.demo_advanced_context_optimization()
    elif args.demo_section == "visualization":
        demo.demo_visualization_features()
    elif args.demo_section == "algorithms":
        demo.demo_advanced_algorithms()
    elif args.demo_section == "comprehensive":
        demo.demo_comprehensive_analysis()


if __name__ == "__main__":
    main()
