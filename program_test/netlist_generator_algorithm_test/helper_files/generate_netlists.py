import numpy as np
import os
from scipy.spatial import distance
import cv2
import json

def find_nearest_edge(node_labels, px, py):
    edge_points = np.argwhere(node_labels > 0)
    if edge_points.size == 0:
        return None
    distances = distance.cdist([(py, px)], edge_points)
    nearest_index = np.argmin(distances)
    return edge_points[nearest_index]  # (y, x)

def is_point_connected_or_nearest(node_labels, px, py):
    nearest = find_nearest_edge(node_labels, px, py)
    if nearest is not None:
        return True, (nearest[1], nearest[0])  # (x, y)
    print(f"No connection found for point ({px}, {py})")
    return False, None

def overlay_and_find_nodes_with_connected_regions(node_labels, components, test_results_path, image_file):
    region_to_node = {}
    current_node_id = 1
    gnd_regions = set()
    height, width = node_labels.shape
    num_regions = node_labels.max()

    for component in components:
        comp_id = component["id"]
        for point in component["connection_points"]:
            px, py = point
            if py >= height or px >= width:
                continue
            is_connected, connection_point = is_point_connected_or_nearest(node_labels, px, py)
            if is_connected:
                connected_px, connected_py = connection_point
                region = node_labels[connected_py, connected_px]
                if region > 0:
                    if region not in region_to_node:
                        region_to_node[region] = current_node_id
                        current_node_id += 1
                    if component["label"].upper() == "GND":
                        gnd_regions.add(region)

    # Sort regions by top-left-most pixel
    region_top_left = {}
    for region in range(1, num_regions + 1):
        pixels = np.argwhere(node_labels == region)
        if len(pixels) > 0:
            top_left_pixel = pixels[np.lexsort((pixels[:, 1], pixels[:, 0]))][0]
            region_top_left[region] = top_left_pixel

    sorted_regions = sorted(region_top_left.items(), key=lambda x: (x[1][0], x[1][1]))
    new_region_to_node = {}
    new_node_id = 1
    for region, _ in sorted_regions:
        if region in region_to_node:
            new_region_to_node[region] = new_node_id
            new_node_id += 1

    # Normalize GND nodes
    gnd_nodes = [new_region_to_node[region] for region in gnd_regions if region in new_region_to_node]
    if gnd_nodes:
        smallest_gnd_node = min(gnd_nodes)
        for region in gnd_regions:
            if region in new_region_to_node:
                new_region_to_node[region] = smallest_gnd_node

    # Save node connection results
    results_file = os.path.join(test_results_path, os.path.splitext(image_file)[0] + '.txt')
    with open(results_file, 'w') as results:
        for component in components:
            if component["label"].upper() == "GND":
                continue
            comp_id = component["id"]
            connected_nodes = []
            for point in component["connection_points"]:
                px, py = point
                if py >= height or px >= width:
                    continue
                is_connected, connection_point = is_point_connected_or_nearest(node_labels, px, py)
                if is_connected:
                    connected_px, connected_py = connection_point
                    region = node_labels[connected_py, connected_px]
                    if region > 0 and region in new_region_to_node:
                        node_id = new_region_to_node[region]
                        connected_nodes.append(node_id)
            if connected_nodes:
                results.write(f"{comp_id} {' '.join(map(str, connected_nodes))}\n")

def generate_netlist_for_image(image_file, output_files_path, test_results_path):
    image_name = os.path.splitext(image_file)[0]
    image_output_folder = os.path.join(output_files_path, image_name)
    os.makedirs(image_output_folder, exist_ok=True)

    # Load preprocessed circuit info
    json_path = os.path.join(image_output_folder, 'circuit_info.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing circuit_info.json at: {json_path}")
    with open(json_path, 'r') as json_file:
        components = json.load(json_file)

    masked_edges = cv2.imread(os.path.join(image_output_folder, "6_circuit_cut.png"))
    if masked_edges is not None and len(masked_edges.shape) == 3:
        masked_edges = cv2.cvtColor(masked_edges, cv2.COLOR_BGR2GRAY)
    
    node_labels = np.load(os.path.join(image_output_folder, "node_labels.npy"))

    overlay_and_find_nodes_with_connected_regions(node_labels, components, test_results_path, image_file)

def generate_all_netlists(test_images_folder, output_files_path, test_results_path):
    os.makedirs(test_results_path, exist_ok=True)
    image_files = [f for f in os.listdir(test_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:    
        generate_netlist_for_image(image_file, output_files_path, test_results_path)

