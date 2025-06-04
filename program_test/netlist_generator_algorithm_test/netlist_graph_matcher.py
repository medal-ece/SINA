import os
import networkx as nx
import numpy as np

class Component:
    def __init__(self, name, nodes):
        self.name = name
        self.nodes = nodes

    def __repr__(self):
        return f"{self.name} {self.nodes}"


def read_netlist(file_path):
    components = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            name = parts[0]
            nodes = list(map(int, parts[1:]))
            components.append(Component(name, nodes))
    return components


def process_the_folder(folder_path):
    netlists = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            netlists[file_name] = read_netlist(file_path)
    return netlists


def parse_input(component_list):
    adjacency_dict = {}
    for component in component_list:
        if not isinstance(component, Component):
            continue
        node_raw = component.name
        neighbors = component.nodes if isinstance(component.nodes, list) else [component.nodes]
        node = "_".join(node_raw.split("_")[:-1]) if "_" in node_raw else node_raw
        adjacency_dict.setdefault(node, []).extend(neighbors)
    return adjacency_dict


def check_isomorphism(input1, input2):
    G1, G2 = nx.Graph(), nx.Graph()
    adj_dict1 = parse_input(input1)
    adj_dict2 = parse_input(input2)

    for node, neighbors in adj_dict1.items():
        for neighbor in neighbors:
            G1.add_edge(str(node), str(neighbor))
    for node, neighbors in adj_dict2.items():
        for neighbor in neighbors:
            G2.add_edge(str(node), str(neighbor))

    return nx.is_isomorphic(G1, G2)


def spectral_similarity_fixed(G1, G2):
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        print("⚠️ One or both graphs are empty. Cannot compute spectral similarity.")
        return float('inf')  # Return a high dissimilarity score
    A1 = nx.adjacency_matrix(G1).todense()
    A2 = nx.adjacency_matrix(G2).todense()
    eig1 = np.sort(np.linalg.eigvals(A1))
    eig2 = np.sort(np.linalg.eigvals(A2))
    max_length = max(len(eig1), len(eig2))
    eig1 = np.pad(eig1, (0, max_length - len(eig1)), mode='constant')
    eig2 = np.pad(eig2, (0, max_length - len(eig2)), mode='constant')
    return np.linalg.norm(eig1 - eig2)


def explain_graph_differences(input1, input2):
    G1, G2 = nx.Graph(), nx.Graph()
    adj_dict1 = parse_input(input1)
    adj_dict2 = parse_input(input2)

    for node, neighbors in adj_dict1.items():
        for neighbor in neighbors:
            G1.add_edge(str(node), str(neighbor))
    for node, neighbors in adj_dict2.items():
        for neighbor in neighbors:
            G2.add_edge(str(node), str(neighbor))

    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return "❌ One or both graphs are empty. Cannot compute differences."

    ged = nx.graph_edit_distance(G1, G2)
    spectral_sim = spectral_similarity_fixed(G1, G2)

    explanation = f"**Graph Analysis for non-isomorphic graphs:**\n"
    explanation += f"- Graph Edit Distance (GED): {ged:.1f} changes needed.\n"
    explanation += f"- Spectral Similarity: {spectral_sim:.2f} (lower is better).\n"

    severity_score = (ged / 3) + (spectral_sim / 5)
    if severity_score < 0.01:
        explanation += "✅ Graphs are effectively isomorphic.\n"
    elif severity_score < 0.5:
        explanation += "🔄 Graphs are close to being isomorphic.\n"
    elif severity_score < 1.5:
        explanation += "⚠️ Graphs are structurally similar with moderate differences.\n"
    else:
        explanation += "❌ Graphs are structurally quite different.\n"

    return explanation


def compare_all_images_using_graph_theory():
    current_dir = os.getcwd()
    test_path = os.path.join(current_dir, 'outputs', 'test_results/')
    true_path = os.path.join(current_dir, 'inputs', 'true_netlists/')
    true_files = process_the_folder(true_path)
    test_files = process_the_folder(test_path)

    missing_files = set()

    for file_name, test_netlist in test_files.items():
        if file_name in true_files:
            true_netlist = true_files[file_name]
            print(f"\nComparing: {file_name}")
            isomorphic = check_isomorphism(true_netlist, test_netlist)
            print(f"Isomorphic: {isomorphic}")
            if not isomorphic:
                print(explain_graph_differences(true_netlist, test_netlist))
        else:
            missing_files.add(file_name)

    for missing in missing_files:
        print(f"⚠️ Missing true netlist for: {missing}")


if __name__ == "__main__":
    compare_all_images_using_graph_theory()
