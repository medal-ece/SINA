import os
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import json
from collections import defaultdict

@dataclass
class Component:
    """Represents a circuit component with name and connected nodes"""
    name: str
    nodes: List[int]
    
    def __repr__(self):
        return f"{self.name} {self.nodes}"
    
    @property
    def component_type(self):
        """Extract component type from name (e.g., 'Resistor' from 'Resistor_1')"""
        return self.name.split('_')[0]
    
    @property
    def component_id(self):
        """Extract component ID from name"""
        parts = self.name.split('_')
        return parts[1] if len(parts) > 1 else "0"


@dataclass
class MatchingResult:
    """Detailed result of netlist comparison"""
    file_name: str
    is_isomorphic: bool
    accuracy_score: float  # 0-100
    
    # Component metrics
    component_precision: float  # Correct components / Total detected components
    component_recall: float     # Correct components / Total true components
    component_f1: float        # Harmonic mean of precision and recall
    
    # Connection metrics
    connection_accuracy: float  # Percentage of correct connections
    topology_score: float      # Graph similarity score (0-100)
    
    # Detailed errors
    missing_components: List[str]
    extra_components: List[str]
    wrong_connections: List[str]
    
    # Graph properties
    true_graph_stats: Dict
    test_graph_stats: Dict
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'file_name': self.file_name,
            'is_isomorphic': self.is_isomorphic,
            'accuracy_score': round(self.accuracy_score, 2),
            'component_precision': round(self.component_precision, 3),
            'component_recall': round(self.component_recall, 3),
            'component_f1': round(self.component_f1, 3),
            'connection_accuracy': round(self.connection_accuracy, 2),
            'topology_score': round(self.topology_score, 2),
            'missing_components': self.missing_components,
            'extra_components': self.extra_components,
            'wrong_connections': self.wrong_connections,
            'true_graph_stats': self.true_graph_stats,
            'test_graph_stats': self.test_graph_stats
        }


class ImprovedNetlistMatcher:
    """Enhanced netlist comparison with comprehensive metrics"""
    
    def __init__(self, true_path: str = None, test_path: str = None):
        self.true_path = true_path or os.path.join(os.getcwd(), 'inputs', 'true_netlists_structure')
        self.test_path = test_path or os.path.join(os.getcwd(), 'outputs', 'adjusted_netlist_structure')
        self.results: List[MatchingResult] = []
        
    def read_netlist(self, file_path: str) -> List[Component]:
        """Read and parse a netlist file"""
        components = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.strip() == "===":
                        break
                    if ',' not in line:
                        continue
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue
                    name = parts[0].split(':')[0].strip()
                    nodes = list(map(int, parts[1].strip().split()))
                    components.append(Component(name, nodes))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        return components
    
    def components_to_graph(self, components: List[Component]) -> nx.Graph:
        """Convert component list to NetworkX graph"""
        G = nx.Graph()
        
        for comp in components:
            # Add component as a node with attributes
            G.add_node(comp.name, type='component', component_type=comp.component_type)
            
            # Add circuit nodes and edges
            for node in comp.nodes:
                node_name = f"node_{node}"
                G.add_node(node_name, type='circuit_node')
                G.add_edge(comp.name, node_name)
                
        return G
    
    def calculate_graph_stats(self, G: nx.Graph) -> Dict:
        """Calculate comprehensive graph statistics"""
        component_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'component']
        circuit_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'circuit_node']
        
        stats = {
            'num_components': len(component_nodes),
            'num_circuit_nodes': len(circuit_nodes),
            'num_edges': G.number_of_edges(),
            'avg_degree': np.mean([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
            'density': nx.density(G) if G.number_of_nodes() > 1 else 0,
            'components_by_type': defaultdict(int)
        }
        
        # Count components by type
        for node in component_nodes:
            comp_type = G.nodes[node].get('component_type', 'Unknown')
            stats['components_by_type'][comp_type] += 1
            
        stats['components_by_type'] = dict(stats['components_by_type'])
        
        # Add connectivity metrics
        if G.number_of_nodes() > 0:
            stats['is_connected'] = nx.is_connected(G)
            stats['num_connected_components'] = nx.number_connected_components(G)
            
            # Calculate clustering coefficient for circuit structure
            try:
                stats['avg_clustering'] = nx.average_clustering(G)
            except:
                stats['avg_clustering'] = 0
                
        return stats
    
    def match_components(self, true_comps: List[Component], test_comps: List[Component]) -> Tuple[Dict, List, List]:
        """
        Match components between true and test netlists
        Returns: (matches, missing, extra)
        """
        matches = {}
        true_by_type = defaultdict(list)
        test_by_type = defaultdict(list)
        
        # Group by component type
        for comp in true_comps:
            true_by_type[comp.component_type].append(comp)
        for comp in test_comps:
            test_by_type[comp.component_type].append(comp)
        
        # Match components of same type by node connectivity
        for comp_type in true_by_type:
            true_type_comps = true_by_type[comp_type]
            test_type_comps = test_by_type[comp_type]
            
            # Try to match based on node sets
            for true_comp in true_type_comps:
                best_match = None
                best_score = 0
                
                for test_comp in test_type_comps:
                    if test_comp.name in [m[1].name for m in matches.values()]:
                        continue  # Already matched
                        
                    # Calculate similarity based on node overlap
                    true_nodes = set(true_comp.nodes)
                    test_nodes = set(test_comp.nodes)
                    
                    if len(true_nodes) == len(test_nodes):
                        # For components with same number of nodes, check if nodes can be mapped
                        score = len(true_nodes.intersection(test_nodes)) / len(true_nodes)
                        if score > best_score:
                            best_score = score
                            best_match = test_comp
                
                if best_match:
                    matches[true_comp.name] = (true_comp, best_match)
        
        # Find missing and extra components
        matched_true = set(matches.keys())
        matched_test = set(m[1].name for m in matches.values())
        
        missing = [c.name for c in true_comps if c.name not in matched_true]
        extra = [c.name for c in test_comps if c.name not in matched_test]
        
        return matches, missing, extra
    
    def calculate_connection_accuracy(self, true_comps: List[Component], test_comps: List[Component], 
                                     matches: Dict) -> Tuple[float, List[str]]:
        """
        Calculate accuracy of connections between matched components
        Returns: (accuracy_percentage, list_of_wrong_connections)
        """
        if not matches:
            return 0.0, []
        
        correct_connections = 0
        total_connections = 0
        wrong_connections = []
        
        # Build adjacency info for true netlist
        true_adjacency = defaultdict(set)
        for comp in true_comps:
            for node in comp.nodes:
                # Find other components connected to same node
                for other_comp in true_comps:
                    if comp.name != other_comp.name and node in other_comp.nodes:
                        true_adjacency[comp.name].add(other_comp.name)
        
        # Build adjacency info for test netlist
        test_adjacency = defaultdict(set)
        test_comp_dict = {c.name: c for c in test_comps}
        for comp in test_comps:
            for node in comp.nodes:
                for other_comp in test_comps:
                    if comp.name != other_comp.name and node in other_comp.nodes:
                        test_adjacency[comp.name].add(other_comp.name)
        
        # Check connections for matched components
        for true_name, (true_comp, test_comp) in matches.items():
            true_neighbors = true_adjacency.get(true_name, set())
            test_neighbors = test_adjacency.get(test_comp.name, set())
            
            total_connections += len(true_neighbors)
            
            # Map neighbor names through matches
            mapped_test_neighbors = set()
            for test_neighbor in test_neighbors:
                # Find if this neighbor corresponds to a true component
                for t_name, (t_comp, matched_test) in matches.items():
                    if matched_test.name == test_neighbor:
                        mapped_test_neighbors.add(t_name)
                        break
            
            # Count correct connections
            correct = len(true_neighbors.intersection(mapped_test_neighbors))
            correct_connections += correct
            
            # Record wrong connections
            missing_conns = true_neighbors - mapped_test_neighbors
            for miss in missing_conns:
                wrong_connections.append(f"{true_name} should connect to {miss}")
            
        accuracy = (correct_connections / total_connections * 100) if total_connections > 0 else 0
        return accuracy, wrong_connections
    
    def calculate_topology_score(self, G1: nx.Graph, G2: nx.Graph) -> float:
        """
        Calculate topology similarity score (0-100)
        Uses multiple graph metrics for comprehensive comparison
        """
        if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
            return 0.0
        
        scores = []
        
        # 1. Node count similarity
        node_score = 1 - abs(G1.number_of_nodes() - G2.number_of_nodes()) / max(G1.number_of_nodes(), G2.number_of_nodes())
        scores.append(node_score)
        
        # 2. Edge count similarity  
        if G1.number_of_edges() > 0 or G2.number_of_edges() > 0:
            edge_score = 1 - abs(G1.number_of_edges() - G2.number_of_edges()) / max(G1.number_of_edges(), G2.number_of_edges())
            scores.append(edge_score)
        
        # 3. Degree sequence similarity
        deg1 = sorted([d for n, d in G1.degree()])
        deg2 = sorted([d for n, d in G2.degree()])
        
        if deg1 and deg2:
            # Pad shorter sequence
            max_len = max(len(deg1), len(deg2))
            deg1 = deg1 + [0] * (max_len - len(deg1))
            deg2 = deg2 + [0] * (max_len - len(deg2))
            
            # Calculate correlation
            if np.std(deg1) > 0 and np.std(deg2) > 0:
                degree_score = np.corrcoef(deg1, deg2)[0, 1]
                degree_score = (degree_score + 1) / 2  # Normalize to 0-1
                scores.append(degree_score)
        
        # Skip expensive computations (spectral and GED) for faster processing
        # These can be enabled for detailed analysis when needed
        
        # Return weighted average
        return (sum(scores) / len(scores) * 100) if scores else 0.0
    
    def compare_netlists(self, true_comps: List[Component], test_comps: List[Component], 
                        file_name: str) -> MatchingResult:
        """Perform comprehensive comparison of two netlists"""
        
        # Convert to graphs
        G_true = self.components_to_graph(true_comps)
        G_test = self.components_to_graph(test_comps)
        
        # Calculate graph statistics
        true_stats = self.calculate_graph_stats(G_true)
        test_stats = self.calculate_graph_stats(G_test)
        
        # Match components
        matches, missing, extra = self.match_components(true_comps, test_comps)
        
        # Calculate component metrics
        true_count = len(true_comps)
        test_count = len(test_comps)
        matched_count = len(matches)
        
        precision = (matched_count / test_count) if test_count > 0 else 0
        recall = (matched_count / true_count) if true_count > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        # Calculate connection accuracy
        connection_acc, wrong_conns = self.calculate_connection_accuracy(true_comps, test_comps, matches)
        
        # Calculate topology score
        topology_score = self.calculate_topology_score(G_true, G_test)
        
        # Check isomorphism
        is_isomorphic = False
        if true_stats['components_by_type'] == test_stats['components_by_type']:
            # Only check isomorphism if component types match
            is_isomorphic = nx.is_isomorphic(G_true, G_test)
        
        # Calculate overall accuracy score (weighted combination)
        accuracy_score = (
            f1 * 40 +                    # Component detection (40%)
            connection_acc * 0.35 +      # Connection accuracy (35%)
            topology_score * 0.25        # Topology similarity (25%)
        )
        
        return MatchingResult(
            file_name=file_name,
            is_isomorphic=is_isomorphic,
            accuracy_score=accuracy_score,
            component_precision=precision,
            component_recall=recall,
            component_f1=f1,
            connection_accuracy=connection_acc,
            topology_score=topology_score,
            missing_components=missing,
            extra_components=extra,
            wrong_connections=wrong_conns[:10],  # Limit to first 10
            true_graph_stats=true_stats,
            test_graph_stats=test_stats
        )
    
    def process_all_files(self, verbose: bool = False) -> None:
        """Process all netlist files and generate comprehensive report"""
        
        # Get all files
        true_files = {}
        test_files = {}
        
        if os.path.exists(self.true_path):
            for file_name in os.listdir(self.true_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(self.true_path, file_name)
                    true_files[file_name] = self.read_netlist(file_path)
        
        if os.path.exists(self.test_path):
            for file_name in os.listdir(self.test_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(self.test_path, file_name)
                    test_files[file_name] = self.read_netlist(file_path)
        
        print(f"Found {len(true_files)} reference files and {len(test_files)} test files")
        
        # Compare matching files
        for i, file_name in enumerate(test_files):
            if file_name in true_files:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Analyzing: {file_name}")
                    print(f"{'='*60}")

                result = self.compare_netlists(
                    true_files[file_name],
                    test_files[file_name],
                    file_name
                )

                self.results.append(result)
                if verbose:
                    self._print_result(result)
                else:
                    self._print_file_errors(result)
        
        # Generate summary report
        self._generate_summary_report()
        
        # Save detailed results to JSON
        self._save_results_to_json()
    
    def _print_result(self, result: MatchingResult) -> None:
        """Print detailed result for a single comparison"""

        status = "[MATCH]" if result.is_isomorphic else "[MISMATCH]"
        print(f"\nStatus: {status}")
        print(f"Overall Accuracy: {result.accuracy_score:.1f}%")

        print(f"\nComponent Metrics:")
        print(f"  Precision: {result.component_precision:.3f}")
        print(f"  Recall: {result.component_recall:.3f}")
        print(f"  F1 Score: {result.component_f1:.3f}")

        print(f"\nConnection Metrics:")
        print(f"  Connection Accuracy: {result.connection_accuracy:.1f}%")
        print(f"  Topology Score: {result.topology_score:.1f}%")

        if result.missing_components:
            print(f"\n[WARNING] Missing Components: {', '.join(result.missing_components)}")
        if result.extra_components:
            print(f"[WARNING] Extra Components: {', '.join(result.extra_components)}")
        if result.wrong_connections:
            print(f"[WARNING] Wrong Connections: {len(result.wrong_connections)} issues")
            for conn in result.wrong_connections[:3]:  # Show first 3
                print(f"    - {conn}")

    def _print_file_errors(self, result: MatchingResult) -> None:
        """Print concise error summary for a single file"""
        has_errors = (result.missing_components or
                      result.extra_components or
                      result.wrong_connections or
                      not result.is_isomorphic)

        if not has_errors:
            print(f"[OK] {result.file_name} (Accuracy: {result.accuracy_score:.1f}%)")
        else:
            print(f"\n[ERRORS] {result.file_name} (Accuracy: {result.accuracy_score:.1f}%)")
            if result.missing_components:
                print(f"  Missing components: {', '.join(result.missing_components)}")
            if result.extra_components:
                print(f"  Extra components: {', '.join(result.extra_components)}")
            if result.wrong_connections:
                print(f"  Wrong connections ({len(result.wrong_connections)}):")
                for conn in result.wrong_connections:
                    print(f"    - {conn}")

    def _generate_summary_report(self) -> None:
        """Generate comprehensive summary report"""
        
        if not self.results:
            print("\n[ERROR] No results to summarize")
            return
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print(f"{'='*80}")
        
        # Overall metrics
        total = len(self.results)
        perfect_matches = sum(1 for r in self.results if r.is_isomorphic)
        avg_accuracy = np.mean([r.accuracy_score for r in self.results])
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Files Analyzed: {total}")
        print(f"  Perfect Matches (Isomorphic): {perfect_matches}/{total} ({perfect_matches/total*100:.1f}%)")
        print(f"  Average Accuracy Score: {avg_accuracy:.1f}%")
        
        # Component detection metrics
        avg_precision = np.mean([r.component_precision for r in self.results])
        avg_recall = np.mean([r.component_recall for r in self.results])
        avg_f1 = np.mean([r.component_f1 for r in self.results])
        
        # RESEARCH PAPER METRICS
        # Component Detection Accuracy: Using F1 score as it balances precision and recall
        component_detection_accuracy = avg_f1 * 100
        
        # Circuit Topology Accuracy: Average of connection accuracy and topology score
        avg_connection_acc = np.mean([r.connection_accuracy for r in self.results])
        avg_topology_score = np.mean([r.topology_score for r in self.results])
        circuit_topology_accuracy = (avg_connection_acc + avg_topology_score) / 2
        
        print(f"\n" + "="*50)
        print("RESEARCH PAPER METRICS:")
        print("="*50)
        print(f"Component Detection Accuracy: {component_detection_accuracy:.1f}%")
        print(f"Circuit Topology Accuracy: {circuit_topology_accuracy:.1f}%")
        print("="*50)
        
        print(f"\nCOMPONENT DETECTION:")
        print(f"  Average Precision: {avg_precision:.3f}")
        print(f"  Average Recall: {avg_recall:.3f}")
        print(f"  Average F1 Score: {avg_f1:.3f}")
        
        # Connection and topology metrics
        avg_connection = np.mean([r.connection_accuracy for r in self.results])
        avg_topology = np.mean([r.topology_score for r in self.results])
        
        print(f"\nCONNECTIVITY ANALYSIS:")
        print(f"  Average Connection Accuracy: {avg_connection:.1f}%")
        print(f"  Average Topology Score: {avg_topology:.1f}%")
        
        # Error analysis
        total_missing = sum(len(r.missing_components) for r in self.results)
        total_extra = sum(len(r.extra_components) for r in self.results)
        total_wrong_conns = sum(len(r.wrong_connections) for r in self.results)
        
        print(f"\nERROR ANALYSIS:")
        print(f"  Total Missing Components: {total_missing}")
        print(f"  Total Extra Components: {total_extra}")
        print(f"  Total Wrong Connections: {total_wrong_conns}")
        
        # Performance distribution
        score_ranges = {
            "Excellent (90-100%)": 0,
            "Good (75-90%)": 0,
            "Fair (60-75%)": 0,
            "Poor (40-60%)": 0,
            "Very Poor (<40%)": 0
        }
        
        for r in self.results:
            if r.accuracy_score >= 90:
                score_ranges["Excellent (90-100%)"] += 1
            elif r.accuracy_score >= 75:
                score_ranges["Good (75-90%)"] += 1
            elif r.accuracy_score >= 60:
                score_ranges["Fair (60-75%)"] += 1
            elif r.accuracy_score >= 40:
                score_ranges["Poor (40-60%)"] += 1
            else:
                score_ranges["Very Poor (<40%)"] += 1
        
        print(f"\nPERFORMANCE DISTRIBUTION:")
        for range_name, count in score_ranges.items():
            percentage = (count / total * 100) if total > 0 else 0
            bar = "#" * int(percentage / 2)
            print(f"  {range_name}: {count:2d} ({percentage:5.1f}%) {bar}")
        
        # Component type analysis
        component_type_accuracy = defaultdict(list)
        for r in self.results:
            for comp_type in r.true_graph_stats['components_by_type']:
                true_count = r.true_graph_stats['components_by_type'].get(comp_type, 0)
                test_count = r.test_graph_stats['components_by_type'].get(comp_type, 0)
                if true_count > 0:
                    accuracy = min(test_count, true_count) / true_count
                    component_type_accuracy[comp_type].append(accuracy)
        
        print(f"\nCOMPONENT TYPE ACCURACY:")
        for comp_type, accuracies in component_type_accuracy.items():
            avg_acc = np.mean(accuracies) * 100
            print(f"  {comp_type}: {avg_acc:.1f}%")
        
        # Worst performing files
        worst_files = sorted(self.results, key=lambda x: x.accuracy_score)[:5]
        print(f"\n[WARNING] FILES NEEDING ATTENTION (Lowest Scores):")
        for r in worst_files:
            print(f"  {r.file_name}: {r.accuracy_score:.1f}%")
            if r.missing_components:
                print(f"    Missing: {', '.join(r.missing_components[:3])}")
        
        print(f"\n{'='*80}")
    
    def _save_results_to_json(self) -> None:
        """Save detailed results to JSON file"""
        
        output_dir = os.path.join(os.getcwd(), 'outputs', 'validation_results')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'detailed_matching_results.json')
        
        results_dict = {
            'summary': {
                'total_files': len(self.results),
                'perfect_matches': sum(1 for r in self.results if r.is_isomorphic),
                'average_accuracy': np.mean([r.accuracy_score for r in self.results]),
                'average_precision': np.mean([r.component_precision for r in self.results]),
                'average_recall': np.mean([r.component_recall for r in self.results]),
                'average_f1': np.mean([r.component_f1 for r in self.results]),
                'average_connection_accuracy': np.mean([r.connection_accuracy for r in self.results]),
                'average_topology_score': np.mean([r.topology_score for r in self.results])
            },
            'detailed_results': [r.to_dict() for r in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n[INFO] Detailed results saved to: {output_file}")


def main():
    """Main execution function"""
    print("Starting Improved Netlist Graph Matcher")
    print("="*80)
    
    matcher = ImprovedNetlistMatcher()
    matcher.process_all_files()
    
    print("\nAnalysis Complete!")


if __name__ == "__main__":
    main()