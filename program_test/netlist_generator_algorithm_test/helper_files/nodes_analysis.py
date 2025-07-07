from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import os
import random
from collections import defaultdict
import json
import matplotlib
matplotlib.use('Agg') 

def save_circuit_info_json(components, component_connections, output_folder):
    circuit_info = []
    for comp in components:
        comp_id = comp["id"]
        circuit_info.append({
            "id": comp_id,
            "label": comp["type"],
            "bounding_box": comp["bbox"],
            "connection_points": component_connections.get(comp_id, [])
        })

    json_path = os.path.join(output_folder, "circuit_info.json")
    with open(json_path, 'w') as f:
        json.dump(circuit_info, f, indent=4)
    print(f"Saved circuit_info.json to {json_path}")

def identify_circuit_nodes(merged_labels, components, output_folder, original_image):
    h, w = merged_labels.shape
    circuit_image = merged_labels.copy()

    # STEP 1: Cut out the component areas to isolate wiring nodes
    component_mask = np.zeros_like(circuit_image)
    shrink_amount = 3
    for component in components:
        x_min, y_min, x_max, y_max = component["bbox"]
        x_min_r = x_min + shrink_amount
        y_min_r = y_min + shrink_amount
        x_max_r = x_max - shrink_amount
        y_max_r = y_max - shrink_amount
        if x_min_r < x_max_r and y_min_r < y_max_r:
            component_mask[y_min_r:y_max_r, x_min_r:x_max_r] = 1

    circuit_cut = circuit_image.copy()
    circuit_cut[component_mask == 1] = 0
    cv2.imwrite(os.path.join(output_folder, "6_circuit_cut.png"), circuit_cut * 255)

    # STEP 2: Connected component labeling to identify node blobs
    num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(
        circuit_cut.astype(np.uint8), connectivity=8)

    # saving node labels to be later used for netlist generation 
    np.save(os.path.join(output_folder, "node_labels.npy"), node_labels)

    step2_node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id in range(1, num_labels):
        color = np.random.randint(0, 255, 3).tolist()
        step2_node_vis[node_labels == label_id] = color
    cv2.imwrite(os.path.join(output_folder, "7_node_labels.png"), step2_node_vis)

    # STEP 3: Identify valid nodes and collect raw contact points
    valid_nodes = set()
    node_to_components = defaultdict(set)
    component_to_nodes = defaultdict(set)
    raw_contact_points = defaultdict(list)  # (component_id -> list of (x, y))

    for component in components:
        x_min, y_min, x_max, y_max = component["bbox"]
        comp_id = component["id"]
        comp_perimeter = np.zeros((h, w), dtype=np.uint8)
        if y_min < h: comp_perimeter[y_min, max(0, x_min):min(w, x_max + 1)] = 1
        if y_max < h: comp_perimeter[y_max, max(0, x_min):min(w, x_max + 1)] = 1
        if x_min < w: comp_perimeter[max(0, y_min):min(h, y_max + 1), x_min] = 1
        if x_max < w: comp_perimeter[max(0, y_min):min(h, y_max + 1), x_max] = 1

        for label_id in range(1, num_labels):
            node_mask = (node_labels == label_id)
            intersection = np.logical_and(comp_perimeter, node_mask)
            if np.any(intersection):
                valid_nodes.add(label_id)
                node_to_components[label_id].add(comp_id)
                component_to_nodes[comp_id].add(label_id)

                y_points, x_points = np.where(intersection)
                for cx, cy in zip(x_points, y_points):
                    raw_contact_points[comp_id].append((cx, cy))

    # STEP 4: Show valid nodes and all raw contact points
    step4_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for node_id in valid_nodes:
        color = np.random.randint(0, 255, 3).tolist()
        step4_vis[node_labels == node_id] = color

    component_colors = {
        component["id"]: (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        for component in components
    }

    for component in components:
        comp_id = component["id"]
        x_min, y_min, x_max, y_max = component["bbox"]
        color = component_colors[comp_id]
        cv2.rectangle(step4_vis, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
        cv2.putText(step4_vis, comp_id, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for (cx, cy) in raw_contact_points[comp_id]:
            cv2.circle(step4_vis, (cx, cy), 2, color, -1)

    cv2.imwrite(os.path.join(output_folder, "8_valid_nodes.png"), step4_vis)

    # STEP 5: Cluster raw contact points into logical connection points
    clustered_contact_points = defaultdict(list)
    cluster_vis = original_image.copy()

    for comp_id, points in raw_contact_points.items():
        if len(points) == 0:
            continue

        points_array = np.array(points)
        clustering = DBSCAN(eps=10, min_samples=1).fit(points_array)  # eps = pixel radius
        labels = clustering.labels_

        unique_labels = set(labels)
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue  # skip noise

            cluster_pts = points_array[labels == cluster_id]
            if cluster_pts.shape[0] == 0:
                continue  # skip empty cluster just in case

            cx = int(np.mean(cluster_pts[:, 0]))
            cy = int(np.mean(cluster_pts[:, 1]))
            clustered_contact_points[comp_id].append((cx, cy))

            # Draw the clustered point
            cv2.circle(cluster_vis, (cx, cy), 4, component_colors[comp_id], -1)
            cv2.putText(cluster_vis, comp_id, (cx + 4, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, component_colors[comp_id], 1)

    cv2.imwrite(os.path.join(output_folder, "9_clustered_connection_points.png"), cluster_vis)

    # STEP 6: Filter clustered contact points by real node intersection
    real_nodes = {n for n, comps in node_to_components.items() if len(comps) >= 2}
    component_to_real_nodes = {
        comp_id: [n for n in nodes if n in real_nodes]
        for comp_id, nodes in component_to_nodes.items()
    }

    step6_intersections = original_image.copy()
    component_connections = defaultdict(list)

    for component in components:
        comp_id = component["id"]
        if comp_id not in clustered_contact_points or comp_id not in component_to_real_nodes:
            continue

        relevant_nodes = component_to_real_nodes[comp_id]
        for (x, y) in clustered_contact_points[comp_id]:
            for node_id in relevant_nodes:
                node_mask = (node_labels == node_id)
                if node_mask[y, x]:  # pixel overlaps node
                    component_connections[comp_id].append((x, y))
                    cv2.circle(step6_intersections, (x, y), 3, (0, 0, 255), -1)
                    break  # avoid duplicating same point for multiple nodes

    cv2.imwrite(os.path.join(output_folder, "10_node_intersections.png"), step6_intersections)

    # STEP 6a: Visualize only real (valid) nodes with connection points and bounding boxes
    valid_node_only_vis = original_image.copy()

    # Use a single color for all valid nodes (cyan)
    valid_node_color = (0, 225, 0)
    for node_id in real_nodes:
        valid_node_only_vis[node_labels == node_id] = valid_node_color

    # Overlay the actual confirmed connection points
    for comp_id, points in component_connections.items():
        for (x, y) in points:
            cv2.circle(valid_node_only_vis, (x, y), 3, (0, 0, 255), -1)  # red dot

    # Draw bounding boxes for components
    for component in components:
        x_min, y_min, x_max, y_max = component["bbox"]
        comp_id = component["id"]
        cv2.rectangle(valid_node_only_vis, (x_min, y_min), (x_max, y_max), (255, 155, 0) , 1)

    # Save the visualization
    cv2.imwrite(os.path.join(output_folder, "10a_valid_nodes_with_contacts_and_boxes.png"), valid_node_only_vis)

    # STEP 7: Final annotated result — clean output with labeled connection points
    final_image = original_image.copy()
    for comp_id, points in component_connections.items():
        for (x, y) in points:
            cv2.circle(final_image, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(final_image, comp_id, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imwrite(os.path.join(output_folder, "11_final_connection_points.png"), final_image)

    # prepare circuit_info.json for netlist generation
    save_circuit_info_json(components, component_connections, output_folder)
