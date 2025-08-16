#!/usr/bin/env python3
"""
Test Phase 3 Implementation: Advanced Context Features

Tests the advanced context optimization, visualization, and analysis features
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
    from cross_lesson_analyzer import CrossLessonAnalyzer
    from lesson_content_indexer import LessonContentIndexer
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure all Phase 3 components are properly implemented.")
    sys.exit(1)


class Phase3Tester:
    """Test suite for Phase 3 advanced context features."""
    
    def __init__(self, config_dir: Path = Path("config"), output_dir: Path = Path("outputs")):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.context_optimizer = ContextOptimizer(self.config_dir)
        self.visualizer = LessonRelationshipVisualizer(self.config_dir, self.output_dir)
        
        # Test results
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
    
    def test_advanced_context_optimization(self) -> Dict[str, Any]:
        """Test advanced context optimization features."""
        print("\n🔍 Testing Advanced Context Optimization...")
        
        test_results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Get available lessons
            available_lessons = list(self.context_optimizer.cross_lesson_data["content_index"].keys())
            
            if not available_lessons:
                test_results["errors"].append("No lessons available for testing")
                return test_results
            
            # Test 1: Basic context optimization
            print("  Testing basic context optimization...")
            source_lesson = available_lessons[0]
            result = self.context_optimizer.get_optimized_context(source_lesson)
            
            if result and "selected_context" in result:
                test_results["passed"] += 1
                print(f"    ✓ Basic optimization: {len(result['selected_context'])} context sources")
            else:
                test_results["failed"] += 1
                test_results["errors"].append("Basic context optimization failed")
            
            # Test 2: Content type-specific optimization
            print("  Testing content type-specific optimization...")
            for content_type in ["definitions", "procedures", "concepts"]:
                try:
                    result = self.context_optimizer.optimize_for_content_type(source_lesson, content_type)
                    if result and "selected_context" in result:
                        test_results["passed"] += 1
                        print(f"    ✓ {content_type} optimization: {len(result['selected_context'])} sources")
                    else:
                        test_results["failed"] += 1
                        test_results["errors"].append(f"{content_type} optimization failed")
                except Exception as e:
                    test_results["failed"] += 1
                    test_results["errors"].append(f"{content_type} optimization error: {e}")
            
            # Test 3: Context pattern analysis
            print("  Testing context pattern analysis...")
            result = self.context_optimizer.get_optimized_context(source_lesson)
            if result and "context_analysis" in result:
                analysis = result["context_analysis"]
                if "concept_clusters" in analysis and "content_complexity" in analysis:
                    test_results["passed"] += 1
                    print(f"    ✓ Pattern analysis: {len(analysis.get('concept_clusters', {}))} clusters")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Context pattern analysis incomplete")
            
            # Test 4: Adaptive recommendations
            print("  Testing adaptive recommendations...")
            if result and "adaptive_recommendations" in result:
                recommendations = result["adaptive_recommendations"]
                if isinstance(recommendations, list):
                    test_results["passed"] += 1
                    print(f"    ✓ Adaptive recommendations: {len(recommendations)} suggestions")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Adaptive recommendations format error")
            
            # Test 5: Context evolution analysis
            print("  Testing context evolution analysis...")
            try:
                evolution = self.context_optimizer.get_context_evolution_analysis(source_lesson)
                if evolution and "current_context" in evolution:
                    test_results["passed"] += 1
                    print(f"    ✓ Evolution analysis: {evolution['current_context']['context_sources']} sources")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Context evolution analysis failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Evolution analysis error: {e}")
            
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Advanced optimization test error: {e}")
        
        print(f"  Advanced Context Optimization: {test_results['passed']} passed, {test_results['failed']} failed")
        return test_results
    
    def test_visualization_features(self) -> Dict[str, Any]:
        """Test visualization features."""
        print("\n🎨 Testing Visualization Features...")
        
        test_results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "generated_files": []
        }
        
        try:
            # Test 1: Lesson relationship network
            print("  Testing lesson relationship network...")
            try:
                output_file = "test_relationships.png"
                result = self.visualizer.visualize_lesson_relationships(output_file)
                if result and Path(result).exists():
                    test_results["passed"] += 1
                    test_results["generated_files"].append(result)
                    print(f"    ✓ Relationship network: {result}")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Lesson relationship visualization failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Relationship visualization error: {e}")
            
            # Test 2: Concept correlation network
            print("  Testing concept correlation network...")
            try:
                output_file = "test_concepts.png"
                result = self.visualizer.visualize_concept_correlations(output_file)
                if result and Path(result).exists():
                    test_results["passed"] += 1
                    test_results["generated_files"].append(result)
                    print(f"    ✓ Concept correlations: {result}")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Concept correlation visualization failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Concept visualization error: {e}")
            
            # Test 3: Semantic space visualization
            print("  Testing semantic space visualization...")
            try:
                output_file = "test_semantic.png"
                result = self.visualizer.visualize_semantic_space(output_file)
                if result and Path(result).exists():
                    test_results["passed"] += 1
                    test_results["generated_files"].append(result)
                    print(f"    ✓ Semantic space: {result}")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Semantic space visualization failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Semantic visualization error: {e}")
            
            # Test 4: Comprehensive visualization
            print("  Testing comprehensive visualization...")
            try:
                results = self.visualizer.create_comprehensive_visualization("test_comprehensive")
                if results and len(results) >= 4:  # Should have relationships, concepts, semantic, insights
                    test_results["passed"] += 1
                    test_results["generated_files"].extend(results.values())
                    print(f"    ✓ Comprehensive visualization: {len(results)} outputs")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Comprehensive visualization incomplete")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Comprehensive visualization error: {e}")
            
            # Test 5: Curriculum insights generation
            print("  Testing curriculum insights generation...")
            try:
                insights = self.visualizer.generate_curriculum_insights()
                if insights and "total_lessons" in insights and "recommendations" in insights:
                    test_results["passed"] += 1
                    print(f"    ✓ Curriculum insights: {insights['total_lessons']} lessons, {len(insights['recommendations'])} recommendations")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Curriculum insights generation failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Insights generation error: {e}")
            
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Visualization test error: {e}")
        
        print(f"  Visualization Features: {test_results['passed']} passed, {test_results['failed']} failed")
        return test_results
    
    def test_advanced_algorithms(self) -> Dict[str, Any]:
        """Test advanced algorithms and analysis features."""
        print("\n🧠 Testing Advanced Algorithms...")
        
        test_results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Test 1: Adaptive context selection
            print("  Testing adaptive context selection...")
            available_lessons = list(self.context_optimizer.cross_lesson_data["content_index"].keys())
            if available_lessons:
                source_lesson = available_lessons[0]
                
                # Test with different max lengths
                for max_length in [1000, 2000, 3000]:
                    try:
                        result = self.context_optimizer.optimize_context_selection(
                            source_lesson, "general", max_length
                        )
                        if isinstance(result, list):
                            test_results["passed"] += 1
                            print(f"    ✓ Adaptive selection ({max_length} chars): {len(result)} sources")
                        else:
                            test_results["failed"] += 1
                            test_results["errors"].append(f"Adaptive selection failed for {max_length} chars")
                    except Exception as e:
                        test_results["failed"] += 1
                        test_results["errors"].append(f"Adaptive selection error: {e}")
            
            # Test 2: Quality assessment
            print("  Testing quality assessment...")
            try:
                # Create test context
                test_context = [
                    {"lesson_id": "test1", "weight": 0.8, "content_length": 1000},
                    {"lesson_id": "test2", "weight": 0.6, "content_length": 1500}
                ]
                
                assessment = self.context_optimizer.assess_context_quality(test_context)
                if assessment and "quality_score" in assessment:
                    test_results["passed"] += 1
                    print(f"    ✓ Quality assessment: score {assessment['quality_score']:.2f}")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Quality assessment failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Quality assessment error: {e}")
            
            # Test 3: Network graph creation
            print("  Testing network graph creation...")
            try:
                relationship_network = self.visualizer.create_relationship_network()
                if relationship_network and hasattr(relationship_network, 'nodes'):
                    test_results["passed"] += 1
                    print(f"    ✓ Relationship network: {len(relationship_network.nodes())} nodes")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Relationship network creation failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Network creation error: {e}")
            
            # Test 4: Concept correlation network
            print("  Testing concept correlation network...")
            try:
                concept_network = self.visualizer.create_concept_correlation_network()
                if concept_network and hasattr(concept_network, 'nodes'):
                    test_results["passed"] += 1
                    print(f"    ✓ Concept network: {len(concept_network.nodes())} nodes")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Concept network creation failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Concept network error: {e}")
            
            # Test 5: Semantic space analysis
            print("  Testing semantic space analysis...")
            try:
                coords, lesson_ids = self.visualizer.create_semantic_space_visualization()
                if len(coords) > 0 and len(lesson_ids) > 0:
                    test_results["passed"] += 1
                    print(f"    ✓ Semantic space: {len(lesson_ids)} lessons, {coords.shape} dimensions")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Semantic space analysis failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Semantic space error: {e}")
            
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Advanced algorithms test error: {e}")
        
        print(f"  Advanced Algorithms: {test_results['passed']} passed, {test_results['failed']} failed")
        return test_results
    
    def test_integration_features(self) -> Dict[str, Any]:
        """Test integration with existing Phase 2B components."""
        print("\n🔗 Testing Integration Features...")
        
        test_results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Test 1: Integration with cross-lesson analyzer
            print("  Testing cross-lesson analyzer integration...")
            try:
                analyzer = CrossLessonAnalyzer(self.config_dir)
                if hasattr(analyzer, 'analyze_lesson_relationships'):
                    test_results["passed"] += 1
                    print(f"    ✓ Cross-lesson analyzer integration")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Cross-lesson analyzer integration failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Cross-lesson analyzer error: {e}")
            
            # Test 2: Integration with content indexer
            print("  Testing content indexer integration...")
            try:
                indexer = LessonContentIndexer(self.config_dir)
                if hasattr(indexer, 'index_lesson_content'):
                    test_results["passed"] += 1
                    print(f"    ✓ Content indexer integration")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Content indexer integration failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Content indexer error: {e}")
            
            # Test 3: Data consistency
            print("  Testing data consistency...")
            try:
                # Check if all components can access the same data
                optimizer_data = self.context_optimizer.cross_lesson_data
                visualizer_data = self.visualizer.cross_lesson_data
                
                if (len(optimizer_data["content_index"]) == len(visualizer_data["content_index"]) and
                    len(optimizer_data["lesson_relationships"]) == len(visualizer_data["lesson_relationships"])):
                    test_results["passed"] += 1
                    print(f"    ✓ Data consistency: {len(optimizer_data['content_index'])} lessons")
                else:
                    test_results["failed"] += 1
                    test_results["errors"].append("Data consistency check failed")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"Data consistency error: {e}")
            
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Integration test error: {e}")
        
        print(f"  Integration Features: {test_results['passed']} passed, {test_results['failed']} failed")
        return test_results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 3 tests."""
        print("🚀 Starting Phase 3 Implementation Tests...")
        print("=" * 60)
        
        # Run individual test suites
        self.test_results["tests"]["advanced_optimization"] = self.test_advanced_context_optimization()
        self.test_results["tests"]["visualization"] = self.test_visualization_features()
        self.test_results["tests"]["advanced_algorithms"] = self.test_advanced_algorithms()
        self.test_results["tests"]["integration"] = self.test_integration_features()
        
        # Calculate summary
        total_passed = sum(test["passed"] for test in self.test_results["tests"].values())
        total_failed = sum(test["failed"] for test in self.test_results["tests"].values())
        total_errors = sum(len(test["errors"]) for test in self.test_results["tests"].values())
        
        self.test_results["summary"] = {
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "success_rate": total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Phase 3 Test Results Summary")
        print("=" * 60)
        
        for test_name, results in self.test_results["tests"].items():
            print(f"{test_name.replace('_', ' ').title()}:")
            print(f"  Passed: {results['passed']}")
            print(f"  Failed: {results['failed']}")
            if results['errors']:
                print(f"  Errors: {len(results['errors'])}")
                for error in results['errors'][:3]:  # Show first 3 errors
                    print(f"    - {error}")
            print()
        
        print(f"Overall Success Rate: {self.test_results['summary']['success_rate']:.1%}")
        print(f"Total Tests: {total_passed + total_failed}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        
        if total_failed == 0:
            print("\n✅ All Phase 3 tests passed! Advanced context features are working correctly.")
        else:
            print(f"\n⚠️  {total_failed} tests failed. Please review the errors above.")
        
        return self.test_results
    
    def save_test_results(self, output_file: str = "phase3_test_results.json"):
        """Save test results to file."""
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Test results saved to: {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Test Phase 3 Implementation")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--save-results", action="store_true", help="Save test results to file")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = Phase3Tester(Path(args.config_dir), Path(args.output_dir))
    
    # Run tests
    results = tester.run_all_tests()
    
    # Save results if requested
    if args.save_results:
        tester.save_test_results()
    
    # Exit with appropriate code
    if results["summary"]["total_failed"] == 0:
        print("\n🎉 Phase 3 implementation is ready for production!")
        sys.exit(0)
    else:
        print(f"\n❌ {results['summary']['total_failed']} tests failed. Please fix issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
