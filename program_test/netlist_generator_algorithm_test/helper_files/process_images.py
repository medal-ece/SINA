# from sklearn.cluster import DBSCAN
# import numpy as np
# import cv2
# import os
# import random
# from collections import defaultdict
# from ultralytics import YOLO

# def identify_circuit_nodes(merged_labels, components, output_folder, original_image):
#     h, w = merged_labels.shape
#     circuit_image = merged_labels.copy()

#     # STEP 1: Cut out the component areas to isolate wiring nodes
#     component_mask = np.zeros_like(circuit_image)
#     shrink_amount = 3
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         x_min_r = x_min + shrink_amount
#         y_min_r = y_min + shrink_amount
#         x_max_r = x_max - shrink_amount
#         y_max_r = y_max - shrink_amount
#         if x_min_r < x_max_r and y_min_r < y_max_r:
#             component_mask[y_min_r:y_max_r, x_min_r:x_max_r] = 1

#     circuit_cut = circuit_image.copy()
#     circuit_cut[component_mask == 1] = 0
#     cv2.imwrite(os.path.join(output_folder, "6_circuit_cut.png"), circuit_cut * 255)

#     # STEP 2: Connected component labeling to identify node blobs
#     num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(
#         circuit_cut.astype(np.uint8), connectivity=8)

#     step2_node_vis = np.zeros((h, w, 3), dtype=np.uint8)
#     for label_id in range(1, num_labels):
#         color = np.random.randint(0, 255, 3).tolist()
#         step2_node_vis[node_labels == label_id] = color
#     cv2.imwrite(os.path.join(output_folder, "7_node_labels.png"), step2_node_vis)

#     # STEP 3: Identify valid nodes and collect raw contact points
#     valid_nodes = set()
#     node_to_components = defaultdict(set)
#     component_to_nodes = defaultdict(set)
#     raw_contact_points = defaultdict(list)  # (component_id -> list of (x, y))

#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         comp_id = component["id"]
#         comp_perimeter = np.zeros((h, w), dtype=np.uint8)
#         if y_min < h: comp_perimeter[y_min, max(0, x_min):min(w, x_max + 1)] = 1
#         if y_max < h: comp_perimeter[y_max, max(0, x_min):min(w, x_max + 1)] = 1
#         if x_min < w: comp_perimeter[max(0, y_min):min(h, y_max + 1), x_min] = 1
#         if x_max < w: comp_perimeter[max(0, y_min):min(h, y_max + 1), x_max] = 1

#         for label_id in range(1, num_labels):
#             node_mask = (node_labels == label_id)
#             intersection = np.logical_and(comp_perimeter, node_mask)
#             if np.any(intersection):
#                 valid_nodes.add(label_id)
#                 node_to_components[label_id].add(comp_id)
#                 component_to_nodes[comp_id].add(label_id)

#                 y_points, x_points = np.where(intersection)
#                 for cx, cy in zip(x_points, y_points):
#                     raw_contact_points[comp_id].append((cx, cy))

#     # STEP 4: Show valid nodes and all raw contact points
#     step4_vis = np.zeros((h, w, 3), dtype=np.uint8)
#     for node_id in valid_nodes:
#         color = np.random.randint(0, 255, 3).tolist()
#         step4_vis[node_labels == node_id] = color

#     component_colors = {
#         component["id"]: (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
#         for component in components
#     }

#     for component in components:
#         comp_id = component["id"]
#         x_min, y_min, x_max, y_max = component["bbox"]
#         color = component_colors[comp_id]
#         cv2.rectangle(step4_vis, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
#         cv2.putText(step4_vis, comp_id, (x_min, y_min - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
#         for (cx, cy) in raw_contact_points[comp_id]:
#             cv2.circle(step4_vis, (cx, cy), 2, color, -1)

#     cv2.imwrite(os.path.join(output_folder, "8_valid_nodes.png"), step4_vis)

#     # STEP 5: Cluster raw contact points into logical connection points

#     clustered_contact_points = defaultdict(list)
#     cluster_vis = original_image.copy()

#     for comp_id, points in raw_contact_points.items():
#         if len(points) == 0:
#             continue

#         points_array = np.array(points)
#         clustering = DBSCAN(eps=10, min_samples=1).fit(points_array)  # eps = pixel radius
#         labels = clustering.labels_

#         unique_labels = set(labels)
#         for cluster_id in unique_labels:
#             if cluster_id == -1:
#                 continue  # skip noise

#             cluster_pts = points_array[labels == cluster_id]
#             if cluster_pts.shape[0] == 0:
#                 continue  # skip empty cluster just in case

#             cx = int(np.mean(cluster_pts[:, 0]))
#             cy = int(np.mean(cluster_pts[:, 1]))
#             clustered_contact_points[comp_id].append((cx, cy))

#             # Draw the clustered point
#             cv2.circle(cluster_vis, (cx, cy), 4, component_colors[comp_id], -1)
#             cv2.putText(cluster_vis, comp_id, (cx + 4, cy - 4),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, component_colors[comp_id], 1)

#     cv2.imwrite(os.path.join(output_folder, "9_clustered_connection_points.png"), cluster_vis)

#     # STEP 6: Filter clustered contact points by real node intersection
#     real_nodes = {n for n, comps in node_to_components.items() if len(comps) >= 2}
#     component_to_real_nodes = {
#         comp_id: [n for n in nodes if n in real_nodes]
#         for comp_id, nodes in component_to_nodes.items()
#     }

#     step6_intersections = original_image.copy()
#     component_connections = defaultdict(list)

#     for component in components:
#         # comp_id = component["id"]
#         print(f"Processing component: {comp_id}")
#         if comp_id not in clustered_contact_points or comp_id not in component_to_real_nodes:
#             continue

#         relevant_nodes = component_to_real_nodes[comp_id]
#         for (x, y) in clustered_contact_points[comp_id]:
#             for node_id in relevant_nodes:
#                 node_mask = (node_labels == node_id)
#                 if node_mask[y, x]:  # pixel overlaps node
#                     component_connections[comp_id].append((x, y))
#                     cv2.circle(step6_intersections, (x, y), 3, (0, 0, 255), -1)
#                     break  # avoid duplicating same point for multiple nodes

#     cv2.imwrite(os.path.join(output_folder, "10_node_intersections.png"), step6_intersections)

#     # STEP 7: Final annotated result — clean output with labeled connection points
#     final_image = original_image.copy()
#     for comp_id, points in component_connections.items():
#         for (x, y) in points:
#             cv2.circle(final_image, (x, y), 4, (0, 0, 255), -1)
#             cv2.putText(final_image, comp_id, (x + 5, y - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

#     cv2.imwrite(os.path.join(output_folder, "11_final_connection_points.png"), final_image)

#     # Save text data
#     with open(os.path.join(output_folder, "component_connections.txt"), 'w') as f:
#         for comp_id, points in component_connections.items():
#             line = f"{comp_id} " + ", ".join(f"({x}, {y})" for (x, y) in points)
#             # print("comp_id:", comp_id, "points:", points)
#             f.write(line + "\n")

# def process_circuit_image(image_path, model_path, output_folder, first):
#     os.makedirs(output_folder, exist_ok=True)
#     model = YOLO(model_path)
#     original_image = cv2.imread(image_path)

#     cv2.imwrite(os.path.join(output_folder, "1_original_image.png"), original_image)

#     if original_image is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None, None

#     detection_image = original_image.copy()
#     results = model(image_path)[0]
#     components = []
#     for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(),
#                                         results.boxes.xyxy.cpu().numpy())):
#         class_idx = int(cls)
#         class_name = results.names[class_idx]
#         x_min, y_min, x_max, y_max = map(int, bbox)
#         component_id = f"{class_name}_{i}"
#         color = (random.randint(100, 255),
#                  random.randint(100, 255),
#                  random.randint(100, 255))
#         cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 2)
#         cv2.putText(detection_image, f"{class_name}",
#                     (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, color, 2)
#         # Save image with model-drawn bounding boxes
#         cv2.imwrite(os.path.join(output_folder, "2_model_detected_boxes.png"), detection_image)
#         components.append({
#             "id": component_id,
#             "type": class_name,
#             "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
#             "color": [int(c) for c in color]
#         })

#     gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
#     cv2.imwrite(os.path.join(output_folder, "3_original_binary.png"), binary)

#     kernel = np.ones((3, 3), np.uint8)
#     thickened_binary = cv2.dilate(binary, kernel, iterations=1)
#     cv2.imwrite(os.path.join(output_folder, "4_thickened_binary.png"), thickened_binary)

#     merged_labels = np.zeros_like(thickened_binary)
#     merged_labels[thickened_binary > 0] = 1
#     merged_label_vis = np.zeros_like(original_image)
#     merged_label_vis[merged_labels > 0] = (0, 255, 0)
#     cv2.imwrite(os.path.join(output_folder, "5_merged_labels_green.png"), merged_label_vis)

#     identify_circuit_nodes(merged_labels, components, output_folder, original_image)

# def process_all_images(images_folder, model_path, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     image_files = [f for f in os.listdir(images_folder)
#                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     first = True
#     for image_file in image_files:
#         print(f"Processing {image_file}...")
#         image_path = os.path.join(images_folder, image_file)
#         image_output_folder = os.path.join(output_folder,
#                                            os.path.splitext(image_file)[0])
#         os.makedirs(image_output_folder, exist_ok=True)
#         try:
#             process_circuit_image(image_path, model_path, image_output_folder, first)
#             first = False
#         except Exception as e:
#             print(f"  Error processing {image_file}: {str(e)}")
#             import traceback
#             traceback.print_exc()


from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import os
import random
from collections import defaultdict
from ultralytics import YOLO

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

    # STEP 7: Final annotated result — clean output with labeled connection points
    final_image = original_image.copy()
    for comp_id, points in component_connections.items():
        for (x, y) in points:
            cv2.circle(final_image, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(final_image, comp_id, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imwrite(os.path.join(output_folder, "11_final_connection_points.png"), final_image)

    # Save text data
    with open(os.path.join(output_folder, "component_connections.txt"), 'w') as f:
        for comp_id, points in component_connections.items():
            line = f"{comp_id} " + ", ".join(f"({x}, {y})" for (x, y) in points)
            f.write(line + "\n")

def process_circuit_image(image_path, model_path, output_folder, first):
    os.makedirs(output_folder, exist_ok=True)
    model = YOLO(model_path)
    original_image = cv2.imread(image_path)

    cv2.imwrite(os.path.join(output_folder, "1_original_image.png"), original_image)

    if original_image is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None

    detection_image = original_image.copy()
    results = model(image_path)[0]
    components = []
    type_counters = {}  # Dictionary to track IDs per type
    for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(),
                                        results.boxes.xyxy.cpu().numpy())):
        class_idx = int(cls)
        class_name = results.names[class_idx]
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Increment counter for this class type
        if class_name not in type_counters:
            type_counters[class_name] = 1
        else:
            type_counters[class_name] += 1

        component_id = f"{class_name}_{type_counters[class_name]}"  # ID per type

        color = (random.randint(100, 255),
                 random.randint(100, 255),
                 random.randint(100, 255))

        cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(detection_image, f"{class_name}",
                    (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        # Save image with model-drawn bounding boxes
        cv2.imwrite(os.path.join(output_folder, "2_model_detected_boxes.png"), detection_image)

        components.append({
            "id": component_id,
            "type": class_name,
            "bbox": [x_min, y_min, x_max, y_max],
            "color": [int(c) for c in color]
        })

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(output_folder, "3_original_binary.png"), binary)

    kernel = np.ones((3, 3), np.uint8)
    thickened_binary = cv2.dilate(binary, kernel, iterations=1)
    cv2.imwrite(os.path.join(output_folder, "4_thickened_binary.png"), thickened_binary)

    merged_labels = np.zeros_like(thickened_binary)
    merged_labels[thickened_binary > 0] = 1
    merged_label_vis = np.zeros_like(original_image)
    merged_label_vis[merged_labels > 0] = (0, 255, 0)
    cv2.imwrite(os.path.join(output_folder, "5_merged_labels_green.png"), merged_label_vis)

    identify_circuit_nodes(merged_labels, components, output_folder, original_image)

def process_all_images(images_folder, model_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(images_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    first = True
    for image_file in image_files:
        print(f"Processing {image_file}...")
        image_path = os.path.join(images_folder, image_file)
        image_output_folder = os.path.join(output_folder,
                                           os.path.splitext(image_file)[0])
        os.makedirs(image_output_folder, exist_ok=True)
        try:
            process_circuit_image(image_path, model_path, image_output_folder, first)
            first = False
        except Exception as e:
            print(f"  Error processing {image_file}: {str(e)}")
            import traceback
            traceback.print_exc()
