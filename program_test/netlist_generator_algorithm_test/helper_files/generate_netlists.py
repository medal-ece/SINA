# import numpy as np
# import cv2
# import os
# from scipy.ndimage import label as connected_label
# from scipy.spatial import distance

# def detect_edges(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(grayscale, 50, 150)
#     kernel = np.ones((5, 5), np.uint8)
#     connected_edges = cv2.dilate(edges, kernel, iterations=2)

#     return connected_edges

# def mask_components(connected_edges, components):
#     masked_edges = connected_edges.copy()
#     for component in components:
#         bbox = component["bounding_box"]
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(masked_edges, (x1, y1), (x2, y2), 0, -1)

#     return masked_edges

# def find_nearest_edge(masked_edges, px, py):
#     edge_points = np.argwhere(masked_edges > 0)  # Get all edge points
#     if edge_points.size == 0:
#         return None  # No edges in the mask
#     distances = distance.cdist([(py, px)], edge_points)  # Compute distances from the point to all edges
#     nearest_index = np.argmin(distances)  # Index of the nearest edge

#     return edge_points[nearest_index]  # Coordinates of the nearest edge (y, x)

# def is_point_connected_or_nearest(masked_edges, px, py):

#     # Find the nearest edge
#     nearest_edge = find_nearest_edge(masked_edges, px, py)
#     if nearest_edge is not None:
#         return True, (nearest_edge[1], nearest_edge[0])  # Return x, y of the nearest edge

#     print(f"No connection found for point ({px}, {py})")  # Debug message if something went wrong
#     return False, None  # No connection and no nearest edge

# def overlay_and_find_nodes_with_connected_regions(connected_edges, masked_edges, components, test_results_path, image_file, output_files_path):
#     labeled_edges, num_regions = connected_label(masked_edges)
#     region_to_node = {}
#     current_node_id = 1
#     gnd_regions = set()

#     for component in components:
#         for point in component["connection_points"]:
#             px, py = point
#             if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
#                 continue

#             is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
#             if is_connected:
#                 connected_px, connected_py = connection_point
#                 region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
#                 if region > 0:
#                     if region not in region_to_node:
#                         region_to_node[region] = current_node_id
#                         current_node_id += 1
#                     if component["label"].upper() == "GND":
#                         gnd_regions.add(region)

#     # Rearrange node IDs based on top-left-most pixel
#     region_top_left = {}
#     for region in range(1, num_regions + 1):
#         pixels = np.argwhere(labeled_edges == region)  # Get all pixels in the region
#         if len(pixels) > 0:
#             top_left_pixel = pixels[np.lexsort((pixels[:, 1], pixels[:, 0]))][0]  # Sort by (y, then x) and pick the first
#             region_top_left[region] = top_left_pixel

#     # Sort regions by top-left-most pixel
#     sorted_regions = sorted(region_top_left.items(), key=lambda x: (x[1][0], x[1][1]))  # Sort by (y, x)
#     # Reassign node IDs
#     new_region_to_node = {}
#     new_node_id = 1
#     for region, _ in sorted_regions:
#         if region in region_to_node:
#             new_region_to_node[region] = new_node_id
#             new_node_id += 1

#     # Adjust for gnd_regions
#     gnd_nodes = [new_region_to_node[region] for region in gnd_regions if region in new_region_to_node]
#     if gnd_nodes:  # If there are any ground nodes
#         # Find the smallest node in gnd_nodes
#         smallest_gnd_node = min(gnd_nodes)
        
#         # Update new_region_to_node so that all gnd_regions point to the smallest_gnd_node
#         for region in gnd_regions:
#             if region in new_region_to_node:
#                 new_region_to_node[region] = smallest_gnd_node

#     # Create a new text file for node positions
#     results_file = os.path.join(test_results_path, os.path.splitext(image_file)[0] + '.txt')
#     with open(results_file, 'w') as results:
#         # Dictionary to keep track of label counts
#         label_counts = {}

#         for component in components:
#             # Skip GND components
#             if component["label"].upper() == "GND":
#                 continue
#             connected_nodes = []
#             for point in component["connection_points"]:
#                 px, py = point
#                 if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
#                     continue
#                 is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
#                 if is_connected:
#                     connected_px, connected_py = connection_point
#                     region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
#                     if region > 0 and region in new_region_to_node:
#                         node_id = new_region_to_node[region]
#                         connected_nodes.append(node_id)
#             connected_nodes = list(connected_nodes)
#             if connected_nodes:  # Check if there are any connected nodes
#                 unique_label = component["label"]
#                 # Increment the count for the current label
#                 if unique_label not in label_counts:
#                     label_counts[unique_label] = 0
#                 label_counts[unique_label] += 1
#                 # Create a new label with numbering
#                 numbered_label = f"{unique_label}_{label_counts[unique_label]}"
#                 # Write to the results file
#                 results.write(f"{numbered_label} {' '.join(map(str, connected_nodes))}\n")

# def parse_connections_file(file_path):
#     connections_list = []
    
#     with open(file_path, 'r') as f:
#         for line in f:
#             # Extract component name and connection points
#             parts = line.strip().split(' ', 1)
#             if len(parts) < 2:
#                 continue
                
#             # Extract the connection points using regex
#             import re
#             points = re.findall(r'\((\d+), (\d+)\)', parts[1])
            
#             # Convert to the format we need
#             connection_points = [[int(x), int(y)] for x, y in points]
#             connections_list.append(connection_points)
    
#     return connections_list
    
# def generate_netlist_for_image(image_file, model, test_images_folder, output_files_path, test_results_path):
#     import json

#     image_path = os.path.join(test_images_folder, image_file)
#     image_name = os.path.splitext(image_file)[0]
#     image_output_folder = os.path.join(output_files_path, image_name)
#     json_path = os.path.join(image_output_folder, 'circuit_info.json')
#     connections_file_path = os.path.join(image_output_folder, "component_connections.txt")

#     # Parse connection points from new detection pipeline
#     component_connections = parse_connections_file(connections_file_path)

#     # Detect components with YOLO
#     results = model(image_path)[0]
#     circuit_info = []
#     index = 0
#     for cls, bbox in zip(results.boxes.cls.cpu().numpy(), results.boxes.xyxy.cpu().numpy()):
#         class_idx = int(cls)
#         object_name = results.names[class_idx]
#         x_min, y_min, x_max, y_max = map(int, bbox)
#         bounding_box = [x_min, y_min, x_max, y_max]

#         # Use mapped connection points
#         if index < len(component_connections):
#             connection_points = component_connections[index]
#             index += 1
#         else:
#             connection_points = []

#         circuit_info.append({
#             "label": object_name,
#             "bounding_box": bounding_box,
#             "connection_points": connection_points
#         })

#     # Save JSON info
#     with open(json_path, 'w') as json_file:
#         json.dump(circuit_info, json_file, indent=4)

#     # Reload and run netlist generator
#     with open(json_path, "r") as json_file:
#         components = json.load(json_file)

#     connected_edges = detect_edges(image_path)
#     masked_edges = mask_components(connected_edges, components)
#     overlay_and_find_nodes_with_connected_regions(
#         connected_edges, masked_edges, components,
#         test_results_path, image_file, output_files_path
#     )

# def generate_all_netlists(test_images_folder, output_files_path, test_results_path, model_path):
#     from ultralytics import YOLO
#     os.makedirs(test_results_path, exist_ok=True)

#     model = YOLO(model_path)
#     image_files = [f for f in os.listdir(test_images_folder)
#                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

#     for image_file in image_files:
#         print(f"Generating netlist for {image_file}...")
#         generate_netlist_for_image(image_file, model, test_images_folder,
#                                    output_files_path, test_results_path)



import numpy as np
import cv2
import os
from scipy.ndimage import label as connected_label
from scipy.spatial import distance
import json
import re
from ultralytics import YOLO

# def detect_edges(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(grayscale, 50, 150)
#     kernel = np.ones((5, 5), np.uint8)
#     connected_edges = cv2.dilate(edges, kernel, iterations=2)
#     return connected_edges

# def mask_components(connected_edges, components):
#     masked_edges = connected_edges.copy()
#     for component in components:
#         x1, y1, x2, y2 = component["bounding_box"]
#         cv2.rectangle(masked_edges, (x1, y1), (x2, y2), 0, -1)
#     return masked_edges

import cv2
import numpy as np
import os

def detect_edges(image_path):
    output_folder = os.path.join(os.path.dirname(__file__), "output_images")
    print("saved in ", output_folder)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(output_folder, "a_original_color.png"), image)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, "b_grayscale.png"), grayscale)

    edges = cv2.Canny(grayscale, 50, 150)
    cv2.imwrite(os.path.join(output_folder, "c_canny_edges.png"), edges)

    kernel = np.ones((5, 5), np.uint8)
    connected_edges = cv2.dilate(edges, kernel, iterations=2)
    cv2.imwrite(os.path.join(output_folder, "d_connected_edges.png"), connected_edges)

    return connected_edges

def mask_components(connected_edges, components):
    output_folder = os.path.join(os.path.dirname(__file__), "output_images")
    print("saved in ", output_folder)
    masked_edges = connected_edges.copy()
    for component in components:
        x1, y1, x2, y2 = component["bounding_box"]
        cv2.rectangle(masked_edges, (x1, y1), (x2, y2), 0, -1)

    cv2.imwrite(os.path.join(output_folder, "e_masked_edges.png"), masked_edges)

    return masked_edges


def find_nearest_edge(masked_edges, px, py):
    edge_points = np.argwhere(masked_edges > 0)
    if edge_points.size == 0:
        return None
    distances = distance.cdist([(py, px)], edge_points)
    nearest_index = np.argmin(distances)
    return edge_points[nearest_index]  # (y, x)

def is_point_connected_or_nearest(masked_edges, px, py):
    nearest_edge = find_nearest_edge(masked_edges, px, py)
    if nearest_edge is not None:
        return True, (nearest_edge[1], nearest_edge[0])  # (x, y)
    print(f"No connection found for point ({px}, {py})")
    return False, None

def overlay_and_find_nodes_with_connected_regions(connected_edges, masked_edges, components, test_results_path, image_file, output_files_path):
    labeled_edges, num_regions = connected_label(masked_edges)
    region_to_node = {}
    current_node_id = 1
    gnd_regions = set()

    for component in components:
        comp_id = component["id"]
        for point in component["connection_points"]:
            px, py = point
            if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
                continue
            is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
            if is_connected:
                connected_px, connected_py = connection_point
                region = labeled_edges[connected_py, connected_px]
                if region > 0:
                    if region not in region_to_node:
                        region_to_node[region] = current_node_id
                        current_node_id += 1
                    if component["label"].upper() == "GND":
                        gnd_regions.add(region)

    # Sort regions by top-left-most pixel
    region_top_left = {}
    for region in range(1, num_regions + 1):
        pixels = np.argwhere(labeled_edges == region)
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
                if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
                    continue
                is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
                if is_connected:
                    connected_px, connected_py = connection_point
                    region = labeled_edges[connected_py, connected_px]
                    if region > 0 and region in new_region_to_node:
                        node_id = new_region_to_node[region]
                        connected_nodes.append(node_id)
            if connected_nodes:
                results.write(f"{comp_id} {' '.join(map(str, connected_nodes))}\n")

def parse_connections_file(file_path):
    connections_list = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
            points = re.findall(r'\((\d+), (\d+)\)', parts[1])
            connection_points = [[int(x), int(y)] for x, y in points]
            connections_list.append(connection_points)
    return connections_list

def generate_netlist_for_image(image_file, model, test_images_folder, output_files_path, test_results_path):
    image_path = os.path.join(test_images_folder, image_file)
    image_name = os.path.splitext(image_file)[0]
    image_output_folder = os.path.join(output_files_path, image_name)
    os.makedirs(image_output_folder, exist_ok=True)

    json_path = os.path.join(image_output_folder, 'circuit_info.json')
    connections_file_path = os.path.join(image_output_folder, "component_connections.txt")

    component_connections = parse_connections_file(connections_file_path)
    results = model(image_path)[0]

    circuit_info = []
    type_counters = {}
    index = 0
    for cls, bbox in zip(results.boxes.cls.cpu().numpy(), results.boxes.xyxy.cpu().numpy()):
        class_idx = int(cls)
        object_name = results.names[class_idx]
        x_min, y_min, x_max, y_max = map(int, bbox)
        bounding_box = [x_min, y_min, x_max, y_max]

        # Per-type unique ID
        if object_name not in type_counters:
            type_counters[object_name] = 1
        else:
            type_counters[object_name] += 1
        comp_id = f"{object_name}_{type_counters[object_name]}"

        connection_points = component_connections[index] if index < len(component_connections) else []
        index += 1

        circuit_info.append({
            "id": comp_id,
            "label": object_name,
            "bounding_box": bounding_box,
            "connection_points": connection_points
        })

    with open(json_path, 'w') as json_file:
        json.dump(circuit_info, json_file, indent=4)

    with open(json_path, "r") as json_file:
        components = json.load(json_file)

    connected_edges = detect_edges(image_path)
    masked_edges = mask_components(connected_edges, components)
    overlay_and_find_nodes_with_connected_regions(
        connected_edges, masked_edges, components,
        test_results_path, image_file, output_files_path
    )

def generate_all_netlists(test_images_folder, output_files_path, test_results_path, model_path):
    os.makedirs(test_results_path, exist_ok=True)
    model = YOLO(model_path)
    image_files = [f for f in os.listdir(test_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        print(f"Generating netlist for {image_file}...")
        generate_netlist_for_image(image_file, model, test_images_folder, output_files_path, test_results_path)
