# from sklearn.cluster import DBSCAN
# import numpy as np
# import cv2
# import os
# import random
# from collections import defaultdict
# from ultralytics import YOLO
# import traceback
# import json

# def save_circuit_info_json(components, component_connections, output_folder):
#     circuit_info = []
#     for comp in components:
#         comp_id = comp["id"]
#         circuit_info.append({
#             "id": comp_id,
#             "label": comp["type"],
#             "bounding_box": comp["bbox"],
#             "connection_points": component_connections.get(comp_id, [])
#         })

#     json_path = os.path.join(output_folder, "circuit_info.json")
#     with open(json_path, 'w') as f:
#         json.dump(circuit_info, f, indent=4)
#     print(f"Saved circuit_info.json to {json_path}")

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

#     # saving node labels to be later used for netlist generation 
#     np.save(os.path.join(output_folder, "node_labels.npy"), node_labels)

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
#         comp_id = component["id"]
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

#     # STEP 6a: Visualize only real (valid) nodes with connection points and bounding boxes
#     valid_node_only_vis = original_image.copy()

#     # Use a single color for all valid nodes (cyan)
#     valid_node_color = (0, 225, 0)
#     for node_id in real_nodes:
#         valid_node_only_vis[node_labels == node_id] = valid_node_color

#     # Overlay the actual confirmed connection points
#     for comp_id, points in component_connections.items():
#         for (x, y) in points:
#             cv2.circle(valid_node_only_vis, (x, y), 3, (0, 0, 255), -1)  # red dot
#             # cv2.putText(valid_node_only_vis, comp_id, (x + 5, y - 5),
#             #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

#     # Draw bounding boxes for components
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         comp_id = component["id"]
#         cv2.rectangle(valid_node_only_vis, (x_min, y_min), (x_max, y_max), (255, 155, 0) , 1)
#         # cv2.putText(valid_node_only_vis, comp_id, (x_min, y_min - 5),
#         #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

#     # Save the visualization
#     cv2.imwrite(os.path.join(output_folder, "10a_valid_nodes_with_contacts_and_boxes.png"), valid_node_only_vis)

#     # STEP 7: Final annotated result — clean output with labeled connection points
#     final_image = original_image.copy()
#     for comp_id, points in component_connections.items():
#         for (x, y) in points:
#             cv2.circle(final_image, (x, y), 4, (0, 0, 255), -1)
#             cv2.putText(final_image, comp_id, (x + 5, y - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

#     cv2.imwrite(os.path.join(output_folder, "11_final_connection_points.png"), final_image)

#     # prepare circuit_info.json for netlist generation
#     save_circuit_info_json(components, component_connections, output_folder)

# def process_circuit_image(image_path, model_path, output_folder):
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
#     type_counters = {}  # Dictionary to track IDs per type
#     for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(),
#                                         results.boxes.xyxy.cpu().numpy())):
#         class_idx = int(cls)
#         class_name = results.names[class_idx]
#         x_min, y_min, x_max, y_max = map(int, bbox)

#         # Increment counter for this class type
#         if class_name not in type_counters:
#             type_counters[class_name] = 1
#         else:
#             type_counters[class_name] += 1

#         component_id = f"{class_name}_{type_counters[class_name]}"  # ID per type

#         # color = (random.randint(100, 255),
#         #          random.randint(100, 255),
#         #          random.randint(100, 255))
#         color = (255, 155, 0)  # Use one color (cyan) for all boxes

#         cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 1)
#         cv2.putText(detection_image, f"{class_name}",
#                     (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, color, 1)

#         # Save image with model-drawn bounding boxes
#         cv2.imwrite(os.path.join(output_folder, "2_model_detected_boxes.png"), detection_image)

#         components.append({
#             "id": component_id,
#             "type": class_name,
#             "bbox": [x_min, y_min, x_max, y_max],
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
#     for image_file in image_files:
#         print("\n")
#         print(f"Processing {image_file}...")
#         image_path = os.path.join(images_folder, image_file)
#         image_output_folder = os.path.join(output_folder,
#                                            os.path.splitext(image_file)[0])
#         os.makedirs(image_output_folder, exist_ok=True)
#         try:
#             process_circuit_image(image_path, model_path, image_output_folder)
#         except Exception as e:
#             print(f"  Error processing {image_file}: {str(e)}")
            
#             traceback.print_exc()








# import cv2
# import os
# import traceback

# def process_circuit_image(image_path, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     original_image = cv2.imread(image_path)
#     cv2.imwrite(os.path.join(output_folder, "1_original_image.png"), original_image)

# def process_all_images(images_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)

#     image_files = [f for f in os.listdir(images_folder)
#                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
#     for image_file in image_files:
#         print("\n")
#         print(f"Processing {image_file}...")

#         image_path = os.path.join(images_folder, image_file)
#         image_output_folder = os.path.join(output_folder,
#                                            os.path.splitext(image_file)[0])
        
#         os.makedirs(image_output_folder, exist_ok=True)

#         try:
#             process_circuit_image(image_path, image_output_folder)
#         except Exception as e:
#             print(f"  Error processing {image_file}: {str(e)}")
#             traceback.print_exc()

# import cv2
# import os
# import traceback
# import pytesseract
# import numpy as np
# import re
# from PIL import Image

# def preprocess_image(image):
#     """Preprocess image for better OCR results"""
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply denoising
#     denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
#     # Apply adaptive thresholding for better text extraction
#     thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                    cv2.THRESH_BINARY, 11, 2)
    
#     # Apply morphological operations to clean up the image
#     kernel = np.ones((2,2), np.uint8)
#     cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
#     # Invert if needed (make text black on white background)
#     # Check if background is mostly black
#     if np.mean(cleaned) < 127:
#         cleaned = cv2.bitwise_not(cleaned)
    
#     return cleaned

# def extract_component_labels(image):
#     """Extract component labels using OCR"""
#     # Configure tesseract for better results
#     custom_config = r'--oem 3 --psm 11'
    
#     # Perform OCR
#     text = pytesseract.image_to_string(image, config=custom_config)
    
#     # Define patterns for common component labels
#     component_patterns = [
#         r'\b[RLC]\d+\b',           # R1, R2, L1, C1, etc.
#         r'\b[UV]\d+\b',             # U1, U2 (ICs), V1, V2 (voltage sources)
#         r'\bQ\d+\b',                # Q1, Q2 (transistors)
#         r'\bD\d+\b',                # D1, D2 (diodes)
#         r'\b[JK]\d+\b',             # J1, J2 (connectors), K1, K2 (relays)
#         r'\bTP\d+\b',               # TP1, TP2 (test points)
#         r'\bX\d+\b',                # X1, X2 (crystals)
#         r'\bT\d+\b',                # T1, T2 (transformers)
#         r'\bF\d+\b',                # F1, F2 (fuses)
#         r'\bSW\d+\b',               # SW1, SW2 (switches)
#         r'\b[RLC]\d+[A-Z]?\b',      # R1A, R1B (multi-part components)
#     ]
    
#     # Combine all patterns
#     combined_pattern = '|'.join(component_patterns)
    
#     # Find all component labels
#     components = re.findall(combined_pattern, text, re.IGNORECASE)
    
#     # Convert to uppercase and remove duplicates while preserving order
#     seen = set()
#     unique_components = []
#     for comp in components:
#         comp_upper = comp.upper()
#         if comp_upper not in seen:
#             seen.add(comp_upper)
#             unique_components.append(comp_upper)
    
#     return unique_components

# def enhance_for_ocr(image):
#     """Apply additional enhancements specifically for OCR"""
#     # Resize image if too small (OCR works better on larger images)
#     height, width = image.shape[:2]
#     if width < 1000:
#         scale = 1500 / width
#         new_width = int(width * scale)
#         new_height = int(height * scale)
#         image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
#     # Apply sharpening
#     kernel = np.array([[-1,-1,-1],
#                        [-1, 9,-1],
#                        [-1,-1,-1]])
#     sharpened = cv2.filter2D(image, -1, kernel)
    
#     return sharpened

# def process_circuit_image(image_path, output_folder):
#     """Process a single circuit image and extract component labels"""
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Read the original image
#     original_image = cv2.imread(image_path)
#     cv2.imwrite(os.path.join(output_folder, "1_original_image.png"), original_image)
    
#     # Preprocess the image
#     preprocessed = preprocess_image(original_image)
#     cv2.imwrite(os.path.join(output_folder, "2_preprocessed.png"), preprocessed)
    
#     # Enhance for OCR
#     enhanced = enhance_for_ocr(preprocessed)
#     cv2.imwrite(os.path.join(output_folder, "3_enhanced_for_ocr.png"), enhanced)
    
#     # Extract component labels
#     components = extract_component_labels(enhanced)
    
#     # Try different preprocessing methods if no components found
#     if not components:
#         print("  Trying alternative preprocessing...")
        
#         # Try with simple threshold
#         gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#         _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         components.extend(extract_component_labels(binary))
        
#         # Try with inverted image
#         inverted = cv2.bitwise_not(gray)
#         components.extend(extract_component_labels(inverted))
        
#         # Remove duplicates
#         components = list(dict.fromkeys(components))
    
#     # Sort components by type and number
#     def sort_key(comp):
#         match = re.match(r'([A-Z]+)(\d+)([A-Z]?)', comp)
#         if match:
#             prefix, number, suffix = match.groups()
#             return (prefix, int(number), suffix)
#         return (comp, 0, '')
    
#     components.sort(key=sort_key)
    
#     # Save results to text file
#     output_file = os.path.join(output_folder, "components.txt")
#     with open(output_file, 'w') as f:
#         if components:
#             f.write('\n'.join(components))
#             print(f"  Found {len(components)} components: {', '.join(components[:10])}{'...' if len(components) > 10 else ''}")
#         else:
#             f.write("No components detected")
#             print("  No components detected")
    
#     # Also save a summary
#     summary_file = os.path.join(output_folder, "summary.txt")
#     with open(summary_file, 'w') as f:
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total components found: {len(components)}\n\n")
        
#         # Group by component type
#         component_types = {}
#         for comp in components:
#             comp_type = re.match(r'([A-Z]+)', comp).group(1)
#             if comp_type not in component_types:
#                 component_types[comp_type] = []
#             component_types[comp_type].append(comp)
        
#         for comp_type, comps in sorted(component_types.items()):
#             f.write(f"{comp_type}: {', '.join(comps)}\n")
    
#     return components

# def process_all_images(images_folder, output_folder):
#     """Process all images in the folder"""
#     os.makedirs(output_folder, exist_ok=True)

#     image_files = [f for f in os.listdir(images_folder)
#                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
#     if not image_files:
#         print(f"No image files found in {images_folder}")
#         return
    
#     all_components = []
#     components_per_image = {}
    
#     for image_file in image_files:
#         print("\n" + "="*50)
#         print(f"Processing {image_file}...")

#         image_path = os.path.join(images_folder, image_file)
#         image_output_folder = os.path.join(output_folder,
#                                            os.path.splitext(image_file)[0])
        
#         os.makedirs(image_output_folder, exist_ok=True)

#         try:
#             components = process_circuit_image(image_path, image_output_folder)
#             all_components.extend(components)
#             components_per_image[image_file] = components
#         except Exception as e:
#             print(f"  Error processing {image_file}: {str(e)}")
#             traceback.print_exc()
#             components_per_image[image_file] = []
    
#     # Create overall summary
#     print("\n" + "="*50)
#     print("OVERALL SUMMARY")
#     print("="*50)
    
#     unique_components = list(dict.fromkeys(all_components))
#     print(f"Total components found across all images: {len(all_components)}")
#     print(f"Total unique components across all images: {len(unique_components)}")
    
#     # Save overall summary
#     summary_path = os.path.join(output_folder, "overall_summary.txt")
#     with open(summary_path, 'w') as f:
#         f.write("Circuit Schematic OCR Results\n")
#         f.write("="*30 + "\n\n")
#         f.write(f"Total images processed: {len(image_files)}\n")
#         f.write(f"Total components found: {len(all_components)}\n")
#         f.write(f"Total unique components found: {len(unique_components)}\n\n")
        
#         # Add breakdown by image
#         f.write("Components per image:\n")
#         f.write("-"*30 + "\n")
#         for img, comps in components_per_image.items():
#             f.write(f"{img}: {len(comps)} components\n")
        
#         f.write("\n" + "-"*30 + "\n")
#         f.write("All unique components:\n")
#         f.write('\n'.join(sorted(unique_components)))

# if __name__ == "__main__":
#     # Example usage
#     images_folder = "circuit_images"  # Folder containing circuit schematic images
#     output_folder = "outputs"         # Folder for results
    
#     # Check if tesseract is installed
#     try:
#         pytesseract.get_tesseract_version()
#     except:
#         print("ERROR: Tesseract is not installed or not in PATH")
#         print("Please install Tesseract OCR:")
#         print("- Ubuntu/Debian: sudo apt-get install tesseract-ocr")
#         print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
#         print("- macOS: brew install tesseract")
#         exit(1)
    
#     if os.path.exists(images_folder):
#         process_all_images(images_folder, output_folder)
#     else:
#         print(f"Images folder '{images_folder}' not found!")
#         print("Please create the folder and add circuit schematic images.")




# # Version one with compoennts patters

# import cv2
# import os
# import traceback
# import numpy as np
# import re
# from PIL import Image
# import easyocr
# import pytesseract

# # Set matplotlib to use non-interactive backend before importing pyplot
# import matplotlib
# matplotlib.use('Agg')  # Use non-GUI backend
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# try:
#     from paddleocr import PaddleOCR
#     PADDLE_AVAILABLE = True
# except ImportError:
#     PADDLE_AVAILABLE = False
#     print("PaddleOCR not available, will use EasyOCR + Tesseract")

# class CircuitOCR:
#     def __init__(self, use_gpu=True):
#         """Initialize OCR engines"""
#         print("Initializing OCR engines...")
        
#         # Initialize EasyOCR (this works with your current setup)
#         self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
        
#         # Initialize PaddleOCR only if available
#         if PADDLE_AVAILABLE:
#             self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
#         else:
#             self.paddle_ocr = None
        
#         # Enhanced component patterns
#         self.component_patterns = [
#             r'\b[RLC]\d+[A-Z]?\b',      # R1, R2, L1, C1, R1A, R1B
#             r'\b[UV]\d+[A-Z]?\b',        # U1, U2, V1, V2
#             r'\bQ\d+[A-Z]?\b',           # Q1, Q2 (transistors)
#             r'\bD\d+[A-Z]?\b',           # D1, D2 (diodes)
#             r'\b[JK]\d+[A-Z]?\b',        # J1, J2, K1, K2
#             r'\bTP\d+\b',                # TP1, TP2 (test points)
#             r'\bX\d+[A-Z]?\b',           # X1, X2 (crystals)
#             r'\bT\d+[A-Z]?\b',           # T1, T2 (transformers)
#             r'\bF\d+[A-Z]?\b',           # F1, F2 (fuses)
#             r'\bSW\d+\b',                # SW1, SW2 (switches)
#             r'\bM\d+[A-Z]?\b',           # M1, M2 (MOSFETs)
#             r'\bY\d+\b',                 # Y1, Y2 (crystals/oscillators)
#             r'\bP\d+\b',                 # P1, P2 (connectors/ports)
#             r'\bCR\d+\b',                # CR1, CR2 (diodes, alternative)
#             r'\bLED\d+\b',               # LED1, LED2
#             r'\bBT\d+\b',                # BT1, BT2 (batteries)
#             r'\bRV\d+\b',                # RV1, RV2 (variable resistors)
#             r'\bIC\d+\b',                # IC1, IC2 (integrated circuits)
#             r'\bZD\d+\b',                # ZD1, ZD2 (zener diodes)
#             r'\bRN\d+\b',                # RN1, RN2 (resistor networks)
#             r'\bCN\d+\b',                # CN1, CN2 (connectors)
#             r'\bFL\d+\b',                # FL1, FL2 (filters)
#             r'\bE\d+\b',                 # E1, E2 (test points)
#             r'\bG\d+\b',                 # G1, G2 (generators)
#             r'\bH\d+\b',                 # H1, H2 (holes/hardware)
#             r'\bPS\d+\b',                # PS1, PS2 (power supplies)
#             r'\bS\d+\b',                 # S1, S2 (switches alternative)
#         ]
        
#         self.combined_pattern = '|'.join(self.component_patterns)
        
#         # Common OCR misreads
#         self.ocr_corrections = {
#             'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8',
#             'G': '6', 'l': '1', 'o': '0', 'i': '1', 's': '5'
#         }

#     def advanced_preprocessing(self, image):
#         """Create multiple preprocessing versions optimized for OCR"""
#         versions = {}
        
#         # Original
#         versions['original'] = image.copy()
        
#         # Grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         versions['gray'] = gray
        
#         # High contrast binary (Otsu's method)
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         versions['binary_otsu'] = binary
        
#         # Adaptive threshold (good for varying lighting)
#         adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY, 21, 10)
#         versions['adaptive'] = adaptive
        
#         # Inverted versions (for white text on dark background)
#         versions['inverted_gray'] = cv2.bitwise_not(gray)
#         versions['inverted_binary'] = cv2.bitwise_not(binary)
#         versions['inverted_adaptive'] = cv2.bitwise_not(adaptive)
        
#         # Enhanced contrast using CLAHE
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
#         versions['enhanced_contrast'] = enhanced
        
#         # Denoised version
#         denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#         versions['denoised'] = denoised
        
#         # Edge-preserved smoothing
#         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
#         versions['bilateral'] = bilateral
        
#         # Morphological operations to clean text
#         kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
#         versions['morph_close'] = morph_close
        
#         # Dilated version (makes text thicker)
#         kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#         dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
#         versions['dilated'] = dilated
        
#         # Super resolution attempt (resize)
#         height, width = gray.shape
#         if width < 2000:
#             scale = 2000 / width
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#             versions['upscaled'] = resized
        
#         return versions

#     def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7):
#         """Extract text using EasyOCR with configurable parameters"""
#         try:
#             # Run EasyOCR with specific settings
#             results = self.easy_reader.readtext(
#                 image, 
#                 detail=detail,
#                 width_ths=width_ths,
#                 height_ths=height_ths,
#                 paragraph=False,
#                 text_threshold=0.3,
#                 low_text=0.3,
#                 mag_ratio=1.5
#             )
            
#             if detail == 1:
#                 texts = [result[1].upper() for result in results if result[2] > 0.2]
#             else:
#                 texts = [text.upper() for text in results]
            
#             return texts
#         except Exception as e:
#             print(f"    EasyOCR error: {e}")
#             return []

#     def extract_text_with_tesseract(self, image):
#         """Extract text using Tesseract with multiple configurations"""
#         texts = []
        
#         # Different Tesseract configurations for circuit diagrams
#         configs = [
#             r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Only alphanumeric
#             r'--psm 8',   # Single word
#             r'--psm 11',  # Sparse text
#             r'--psm 12',  # Sparse text with OSD
#             r'--psm 13',  # Raw line
#             r'--psm 6',   # Uniform block
#         ]
        
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(image, config=config)
#                 if text.strip():
#                     texts.append(text.upper())
#             except:
#                 pass
        
#         # Also try getting individual words with bounding boxes
#         try:
#             data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#             for i, word in enumerate(data['text']):
#                 if word.strip():
#                     texts.append(word.upper())
#         except:
#             pass
        
#         return texts

#     def extract_text_with_paddle(self, image):
#         """Extract text using PaddleOCR if available"""
#         if not self.paddle_ocr:
#             return []
        
#         try:
#             result = self.paddle_ocr.ocr(image, cls=True)
#             texts = []
#             if result and result[0]:
#                 for line in result[0]:
#                     if line[1][1] > 0.3:  # Confidence threshold
#                         texts.append(line[1][0].upper())
#             return texts
#         except Exception as e:
#             print(f"    PaddleOCR error: {e}")
#             return []

#     def correct_ocr_errors(self, text):
#         """Correct common OCR misreads"""
#         corrected = text
#         for wrong, right in self.ocr_corrections.items():
#             corrected = corrected.replace(wrong, right)
#         return corrected

#     def extract_components_from_texts(self, texts):
#         """Extract valid component labels from OCR text results"""
#         all_text = ' '.join(texts)
        
#         # Also create version without spaces
#         no_space_text = ''.join(texts).replace(' ', '')
        
#         # Combine both versions
#         combined_text = all_text + ' ' + no_space_text
        
#         # Apply OCR corrections
#         corrected_text = self.correct_ocr_errors(combined_text)
        
#         # Find all matches
#         matches = re.findall(self.combined_pattern, corrected_text, re.IGNORECASE)
        
#         # Clean and deduplicate
#         components = set()
#         for match in matches:
#             comp = match.upper().strip()
#             # Validate component format
#             if re.match(r'^[A-Z]{1,3}\d{1,3}[A-Z]?$', comp):
#                 components.add(comp)
        
#         return list(components)

#     def process_image(self, image_path):
#         """Main processing function"""
#         # Read image
#         image = cv2.imread(image_path)
        
#         # Get all preprocessing versions
#         versions = self.advanced_preprocessing(image)
        
#         all_components = set()
        
#         print("  Processing with multiple OCR engines and preprocessing...")
        
#         # Process each version
#         for version_name, processed_img in versions.items():
#             # Convert grayscale to BGR for EasyOCR if needed
#             if len(processed_img.shape) == 2:
#                 img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 img_for_ocr = processed_img
            
#             # EasyOCR
#             texts = self.extract_text_with_easyocr(img_for_ocr)
#             components = self.extract_components_from_texts(texts)
#             if components:
#                 all_components.update(components)
            
#             # EasyOCR with different parameters
#             texts = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
#             components = self.extract_components_from_texts(texts)
#             if components:
#                 all_components.update(components)
            
#             # Tesseract
#             texts = self.extract_text_with_tesseract(processed_img)
#             components = self.extract_components_from_texts(texts)
#             if components:
#                 all_components.update(components)
            
#             # PaddleOCR if available
#             if PADDLE_AVAILABLE:
#                 texts = self.extract_text_with_paddle(img_for_ocr)
#                 components = self.extract_components_from_texts(texts)
#                 if components:
#                     all_components.update(components)
        
#         # Try cropping and processing regions
#         height, width = image.shape[:2]
#         # Process in quadrants
#         quadrants = [
#             (0, 0, width//2, height//2),
#             (width//2, 0, width, height//2),
#             (0, height//2, width//2, height),
#             (width//2, height//2, width, height)
#         ]
        
#         for x1, y1, x2, y2 in quadrants:
#             cropped = image[y1:y2, x1:x2]
#             texts = self.extract_text_with_easyocr(cropped)
#             components = self.extract_components_from_texts(texts)
#             all_components.update(components)
        
#         return list(all_components), versions

#     def visualize_results(self, image, components, output_path):
#         """Create visualization of detected components"""
#         try:
#             # Set matplotlib to use non-interactive backend
#             import matplotlib
#             matplotlib.use('Agg')  # Use non-GUI backend
#             import matplotlib.pyplot as plt
#             import matplotlib.patches as patches
            
#             # Run EasyOCR to get bounding boxes
#             results = self.easy_reader.readtext(image, detail=1)
            
#             fig, ax = plt.subplots(1, figsize=(15, 10))
#             ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
#             # Draw bounding boxes for detected components
#             found_count = 0
#             for (bbox, text, prob) in results:
#                 text_upper = text.upper()
#                 # Check if this text matches any component
#                 for comp in components:
#                     if comp in text_upper or text_upper in comp:
#                         # Draw bounding box
#                         points = bbox
#                         rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=3)
#                         ax.add_patch(rect)
                        
#                         # Add label
#                         ax.text(points[0][0], points[0][1]-5, comp, 
#                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
#                                fontsize=10, fontweight='bold')
#                         found_count += 1
#                         break
            
#             ax.set_title(f'Circuit OCR Results - Found {len(components)} unique components ({found_count} visualized)', 
#                         fontsize=16, fontweight='bold')
#             ax.axis('off')
#             plt.tight_layout()
#             plt.savefig(output_path, dpi=200, bbox_inches='tight')
#             plt.close()
#         except Exception as e:
#             print(f"    Warning: Could not create visualization: {e}")
#             # Create a simple text visualization instead
#             with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
#                 f.write(f"Found {len(components)} components:\n")
#                 f.write('\n'.join(components))

# def process_circuit_image(image_path, output_folder, ocr_engine):
#     """Process a single circuit image"""
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Save original
#     image = cv2.imread(image_path)
#     cv2.imwrite(os.path.join(output_folder, "1_original.png"), image)
    
#     # Extract components
#     components, versions = ocr_engine.process_image(image_path)
    
#     # Save some preprocessing versions for debugging
#     for i, (name, img) in enumerate(list(versions.items())[:4]):
#         cv2.imwrite(os.path.join(output_folder, f"2_preprocess_{name}.png"), img)
    
#     # Sort components
#     def sort_key(comp):
#         match = re.match(r'([A-Z]+)(\d+)([A-Z]?)', comp)
#         if match:
#             prefix, number, suffix = match.groups()
#             return (prefix, int(number), suffix or '')
#         return (comp, 0, '')
    
#     components.sort(key=sort_key)
    
#     # Visualize
#     if components:
#         ocr_engine.visualize_results(image, components, 
#                                    os.path.join(output_folder, "3_detected_components.png"))
    
#     # Save results
#     with open(os.path.join(output_folder, "components.txt"), 'w') as f:
#         if components:
#             f.write('\n'.join(components))
#             print(f"  Found {len(components)} components: {', '.join(components[:15])}{'...' if len(components) > 15 else ''}")
#         else:
#             f.write("No components detected")
#             print("  No components detected")
    
#     # Save detailed summary
#     with open(os.path.join(output_folder, "summary.txt"), 'w') as f:
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total components found: {len(components)}\n\n")
        
#         # Group by type
#         component_types = {}
#         for comp in components:
#             comp_type = re.match(r'([A-Z]+)', comp).group(1)
#             if comp_type not in component_types:
#                 component_types[comp_type] = []
#             component_types[comp_type].append(comp)
        
#         for comp_type, comps in sorted(component_types.items()):
#             f.write(f"{comp_type} ({len(comps)}): {', '.join(comps)}\n")
    
#     return components

# def process_all_images(images_folder, output_folder, use_gpu=True):
#     """Process all images in folder"""
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Initialize OCR engine
#     ocr_engine = CircuitOCR(use_gpu=use_gpu)
    
#     # Get image files
#     image_files = [f for f in os.listdir(images_folder)
#                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
    
#     if not image_files:
#         print(f"No image files found in {images_folder}")
#         return
    
#     all_components = []
#     components_per_image = {}
    
#     for image_file in image_files:
#         print(f"\nProcessing {image_file}...")
        
#         image_path = os.path.join(images_folder, image_file)
#         image_output_folder = os.path.join(output_folder, os.path.splitext(image_file)[0])
        
#         try:
#             components = process_circuit_image(image_path, image_output_folder, ocr_engine)
#             all_components.extend(components)
#             components_per_image[image_file] = components
#         except Exception as e:
#             print(f"  Error: {str(e)}")
#             traceback.print_exc()
#             components_per_image[image_file] = []
    
#     # Summary
#     print("\n" + "="*60)
#     print("OVERALL SUMMARY")
#     print("="*60)
    
#     unique_components = list(dict.fromkeys(all_components))
#     print(f"Total components found: {len(all_components)}")
#     print(f"Total unique components: {len(unique_components)}")
#     print(f"Images processed: {len(image_files)}")
    
#     # Save overall summary
#     with open(os.path.join(output_folder, "overall_summary.txt"), 'w') as f:
#         f.write("Circuit Schematic OCR Results\n")
#         f.write("="*40 + "\n\n")
#         f.write(f"OCR Engines used: EasyOCR, Tesseract{', PaddleOCR' if PADDLE_AVAILABLE else ''}\n")
#         f.write(f"Total images processed: {len(image_files)}\n")
#         f.write(f"Total components found: {len(all_components)}\n")
#         f.write(f"Total unique components: {len(unique_components)}\n\n")
        
#         f.write("Components per image:\n")
#         f.write("-"*40 + "\n")
#         for img, comps in components_per_image.items():
#             f.write(f"{img}: {len(comps)} components\n")
        
#         f.write("\n" + "-"*40 + "\n")
#         f.write("All unique components (sorted):\n")
#         f.write('\n'.join(sorted(unique_components)))

# version that works well but no confidence rates

# import cv2
# import os
# import traceback
# import numpy as np
# import re
# from PIL import Image
# import easyocr
# import pytesseract

# # Set matplotlib to use non-interactive backend before importing pyplot
# import matplotlib
# matplotlib.use('Agg')  # Use non-GUI backend
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# try:
#     from paddleocr import PaddleOCR
#     PADDLE_AVAILABLE = True
# except ImportError:
#     PADDLE_AVAILABLE = False
#     print("PaddleOCR not available, will use EasyOCR + Tesseract")

# class CircuitOCR:
#     def __init__(self, use_gpu=True):
#         """Initialize OCR engines"""
#         print("Initializing OCR engines...")
        
#         # Initialize EasyOCR (this works with your current setup)
#         self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
        
#         # Initialize PaddleOCR only if available
#         if PADDLE_AVAILABLE:
#             self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
#         else:
#             self.paddle_ocr = None
        
#         # No filtering - capture all text
#         self.component_patterns = []  # Empty - we'll capture everything
#         self.combined_pattern = None  # No pattern matching
        
#         # Common OCR misreads
#         self.ocr_corrections = {
#             'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8',
#             'G': '6', 'l': '1', 'o': '0', 'i': '1', 's': '5'
#         }

#     def advanced_preprocessing(self, image):
#         """Create multiple preprocessing versions optimized for OCR"""
#         versions = {}
        
#         # Original
#         versions['original'] = image.copy()
        
#         # Grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         versions['gray'] = gray
        
#         # High contrast binary (Otsu's method)
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         versions['binary_otsu'] = binary
        
#         # Adaptive threshold (good for varying lighting)
#         adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY, 21, 10)
#         versions['adaptive'] = adaptive
        
#         # Inverted versions (for white text on dark background)
#         versions['inverted_gray'] = cv2.bitwise_not(gray)
#         versions['inverted_binary'] = cv2.bitwise_not(binary)
#         versions['inverted_adaptive'] = cv2.bitwise_not(adaptive)
        
#         # Enhanced contrast using CLAHE
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
#         versions['enhanced_contrast'] = enhanced
        
#         # Denoised version
#         denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#         versions['denoised'] = denoised
        
#         # Edge-preserved smoothing
#         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
#         versions['bilateral'] = bilateral
        
#         # Morphological operations to clean text
#         kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
#         versions['morph_close'] = morph_close
        
#         # Dilated version (makes text thicker)
#         kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#         dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
#         versions['dilated'] = dilated
        
#         # Super resolution attempt (resize)
#         height, width = gray.shape
#         if width < 2000:
#             scale = 2000 / width
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#             versions['upscaled'] = resized
        
#         return versions

#     def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7):
#         """Extract text using EasyOCR with configurable parameters"""
#         try:
#             # Run EasyOCR with specific settings
#             results = self.easy_reader.readtext(
#                 image, 
#                 detail=detail,
#                 width_ths=width_ths,
#                 height_ths=height_ths,
#                 paragraph=False,
#                 text_threshold=0.3,
#                 low_text=0.3,
#                 mag_ratio=1.5
#             )
            
#             if detail == 1:
#                 texts = [result[1].upper() for result in results if result[2] > 0.2]
#             else:
#                 texts = [text.upper() for text in results]
            
#             return texts
#         except Exception as e:
#             print(f"    EasyOCR error: {e}")
#             return []

#     def extract_text_with_tesseract(self, image):
#         """Extract text using Tesseract with multiple configurations"""
#         texts = []
        
#         # Different Tesseract configurations for circuit diagrams
#         configs = [
#             r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Only alphanumeric
#             r'--psm 8',   # Single word
#             r'--psm 11',  # Sparse text
#             r'--psm 12',  # Sparse text with OSD
#             r'--psm 13',  # Raw line
#             r'--psm 6',   # Uniform block
#         ]
        
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(image, config=config)
#                 if text.strip():
#                     texts.append(text.upper())
#             except:
#                 pass
        
#         # Also try getting individual words with bounding boxes
#         try:
#             data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#             for i, word in enumerate(data['text']):
#                 if word.strip():
#                     texts.append(word.upper())
#         except:
#             pass
        
#         return texts

#     def extract_text_with_paddle(self, image):
#         """Extract text using PaddleOCR if available"""
#         if not self.paddle_ocr:
#             return []
        
#         try:
#             result = self.paddle_ocr.ocr(image, cls=True)
#             texts = []
#             if result and result[0]:
#                 for line in result[0]:
#                     if line[1][1] > 0.3:  # Confidence threshold
#                         texts.append(line[1][0].upper())
#             return texts
#         except Exception as e:
#             print(f"    PaddleOCR error: {e}")
#             return []

#     def correct_ocr_errors(self, text):
#         """Correct common OCR misreads"""
#         corrected = text
#         for wrong, right in self.ocr_corrections.items():
#             corrected = corrected.replace(wrong, right)
#         return corrected

#     def extract_components_from_texts(self, texts):
#         """Extract ALL text from OCR results (no filtering)"""
#         # Clean and process all text
#         all_text_items = []
        
#         for text in texts:
#             # Split by common separators and clean
#             items = re.split(r'[\s,;:\n\r\t]+', text)
#             for item in items:
#                 item = item.strip()
#                 # Apply OCR corrections
#                 corrected = self.correct_ocr_errors(item)
#                 if corrected and len(corrected) > 0:  # Keep everything non-empty
#                     all_text_items.append(corrected)
        
#         # Remove exact duplicates while preserving order
#         seen = set()
#         unique_items = []
#         for item in all_text_items:
#             if item not in seen:
#                 seen.add(item)
#                 unique_items.append(item)
        
#         return unique_items

#     def process_image(self, image_path):
#         """Main processing function"""
#         # Read image
#         image = cv2.imread(image_path)
        
#         # Get all preprocessing versions
#         versions = self.advanced_preprocessing(image)
        
#         all_components = set()
        
#         print("  Processing with multiple OCR engines and preprocessing...")
        
#         # Process each version
#         for version_name, processed_img in versions.items():
#             # Convert grayscale to BGR for EasyOCR if needed
#             if len(processed_img.shape) == 2:
#                 img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 img_for_ocr = processed_img
            
#             # EasyOCR
#             texts = self.extract_text_with_easyocr(img_for_ocr)
#             components = self.extract_components_from_texts(texts)
#             if components:
#                 all_components.update(components)
            
#             # EasyOCR with different parameters
#             texts = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
#             components = self.extract_components_from_texts(texts)
#             if components:
#                 all_components.update(components)
            
#             # Tesseract
#             texts = self.extract_text_with_tesseract(processed_img)
#             components = self.extract_components_from_texts(texts)
#             if components:
#                 all_components.update(components)
            
#             # PaddleOCR if available
#             if PADDLE_AVAILABLE:
#                 texts = self.extract_text_with_paddle(img_for_ocr)
#                 components = self.extract_components_from_texts(texts)
#                 if components:
#                     all_components.update(components)
        
#         # Try cropping and processing regions
#         height, width = image.shape[:2]
#         # Process in quadrants
#         quadrants = [
#             (0, 0, width//2, height//2),
#             (width//2, 0, width, height//2),
#             (0, height//2, width//2, height),
#             (width//2, height//2, width, height)
#         ]
        
#         for x1, y1, x2, y2 in quadrants:
#             cropped = image[y1:y2, x1:x2]
#             texts = self.extract_text_with_easyocr(cropped)
#             components = self.extract_components_from_texts(texts)
#             all_components.update(components)
        
#         return list(all_components), versions

#     def visualize_results(self, image, all_text, output_path):
#         """Create visualization of detected text"""
#         try:
#             # Set matplotlib to use non-interactive backend
#             import matplotlib
#             matplotlib.use('Agg')  # Use non-GUI backend
#             import matplotlib.pyplot as plt
#             import matplotlib.patches as patches
            
#             # Run EasyOCR to get bounding boxes
#             results = self.easy_reader.readtext(image, detail=1)
            
#             fig, ax = plt.subplots(1, figsize=(15, 10))
#             ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
#             # Draw bounding boxes for all detected text
#             for i, (bbox, text, prob) in enumerate(results):
#                 if prob > 0.2:  # Show all text with reasonable confidence
#                     # Draw bounding box
#                     points = bbox
#                     rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=2)
#                     ax.add_patch(rect)
                    
#                     # Add label
#                     ax.text(points[0][0], points[0][1]-5, text, 
#                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
#                            fontsize=8)
            
#             ax.set_title(f'Circuit OCR Results - Found {len(all_text)} text items', 
#                         fontsize=16, fontweight='bold')
#             ax.axis('off')
#             plt.tight_layout()
#             plt.savefig(output_path, dpi=200, bbox_inches='tight')
#             plt.close()
#         except Exception as e:
#             print(f"    Warning: Could not create visualization: {e}")
#             # Create a simple text visualization instead
#             with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
#                 f.write(f"Found {len(all_text)} text items:\n")
#                 f.write('\n'.join(all_text))

# def process_circuit_image(image_path, output_folder, ocr_engine):
#     """Process a single circuit image and extract ALL text"""
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Save original
#     image = cv2.imread(image_path)
#     cv2.imwrite(os.path.join(output_folder, "1_original.png"), image)
    
#     # Extract all text
#     all_text, versions = ocr_engine.process_image(image_path)
    
#     # Save some preprocessing versions for debugging
#     for i, (name, img) in enumerate(list(versions.items())[:4]):
#         cv2.imwrite(os.path.join(output_folder, f"2_preprocess_{name}.png"), img)
    
#     # No sorting needed for general text
    
#     # Visualize
#     if all_text:
#         ocr_engine.visualize_results(image, all_text, 
#                                    os.path.join(output_folder, "3_detected_text.png"))
    
#     # Save results - all extracted text
#     with open(os.path.join(output_folder, "extracted_text.txt"), 'w', encoding='utf-8') as f:
#         if all_text:
#             f.write('\n'.join(all_text))
#             print(f"  Found {len(all_text)} text items: {', '.join(all_text[:10])}{'...' if len(all_text) > 10 else ''}")
#         else:
#             f.write("No text detected")
#             print("  No text detected")
    
#     # Save detailed summary
#     with open(os.path.join(output_folder, "summary.txt"), 'w', encoding='utf-8') as f:
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total text items found: {len(all_text)}\n\n")
#         f.write("All extracted text:\n")
#         f.write("-" * 40 + "\n")
#         for item in all_text:
#             f.write(f"{item}\n")
    
#     return all_text

# def process_all_images(images_folder, output_folder, use_gpu=True):
#     """Process all images in folder and extract ALL text"""
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Initialize OCR engine
#     ocr_engine = CircuitOCR(use_gpu=use_gpu)
    
#     # Get image files
#     image_files = [f for f in os.listdir(images_folder)
#                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
    
#     if not image_files:
#         print(f"No image files found in {images_folder}")
#         return
    
#     all_text_items = []
#     text_per_image = {}
    
#     for image_file in image_files:
#         print(f"\nProcessing {image_file}...")
        
#         image_path = os.path.join(images_folder, image_file)
#         image_output_folder = os.path.join(output_folder, os.path.splitext(image_file)[0])
        
#         try:
#             text_items = process_circuit_image(image_path, image_output_folder, ocr_engine)
#             all_text_items.extend(text_items)
#             text_per_image[image_file] = text_items
#         except Exception as e:
#             print(f"  Error: {str(e)}")
#             traceback.print_exc()
#             text_per_image[image_file] = []
    
#     # Summary
#     print("\n" + "="*60)
#     print("OVERALL SUMMARY")
#     print("="*60)
    
#     unique_text_items = list(dict.fromkeys(all_text_items))
#     print(f"Total text items found: {len(all_text_items)}")
#     print(f"Total unique text items: {len(unique_text_items)}")
#     print(f"Images processed: {len(image_files)}")
    
#     # Save overall summary
#     with open(os.path.join(output_folder, "overall_summary.txt"), 'w', encoding='utf-8') as f:
#         f.write("Circuit Image Text Extraction Results\n")
#         f.write("="*40 + "\n\n")
#         f.write(f"OCR Engines used: EasyOCR, Tesseract{', PaddleOCR' if PADDLE_AVAILABLE else ''}\n")
#         f.write(f"Total images processed: {len(image_files)}\n")
#         f.write(f"Total text items found: {len(all_text_items)}\n")
#         f.write(f"Total unique text items: {len(unique_text_items)}\n\n")
        
#         f.write("Text items per image:\n")
#         f.write("-"*40 + "\n")
#         for img, items in text_per_image.items():
#             f.write(f"{img}: {len(items)} text items\n")
        
#         f.write("\n" + "-"*40 + "\n")
#         f.write("All unique text items:\n")
#         for item in sorted(unique_text_items):
#             f.write(f"{item}\n")


import cv2
import os
import traceback
import numpy as np
import re
from PIL import Image
import easyocr
import pytesseract

# Set matplotlib to use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("PaddleOCR not available, will use EasyOCR + Tesseract")

class CircuitOCR:
    def __init__(self, use_gpu=True):
        """Initialize OCR engines"""
        print("Initializing OCR engines...")
        
        # Initialize EasyOCR (this works with your current setup)
        self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
        
        # Initialize PaddleOCR only if available
        if PADDLE_AVAILABLE:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
        else:
            self.paddle_ocr = None
        
        # No filtering - capture all text
        self.component_patterns = []  # Empty - we'll capture everything
        self.combined_pattern = None  # No pattern matching
        
        # Common OCR misreads
        self.ocr_corrections = {
            'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8',
            'G': '6', 'l': '1', 'o': '0', 'i': '1', 's': '5'
        }

    def advanced_preprocessing(self, image):
        """Create multiple preprocessing versions optimized for OCR"""
        versions = {}
        
        # Original
        versions['original'] = image.copy()
        
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        versions['gray'] = gray
        
        # High contrast binary (Otsu's method)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions['binary_otsu'] = binary
        
        # Adaptive threshold (good for varying lighting)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 10)
        versions['adaptive'] = adaptive
        
        # Inverted versions (for white text on dark background)
        versions['inverted_gray'] = cv2.bitwise_not(gray)
        versions['inverted_binary'] = cv2.bitwise_not(binary)
        versions['inverted_adaptive'] = cv2.bitwise_not(adaptive)
        
        # Enhanced contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        versions['enhanced_contrast'] = enhanced
        
        # Denoised version
        denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
        versions['denoised'] = denoised
        
        # Edge-preserved smoothing
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        versions['bilateral'] = bilateral
        
        # Morphological operations to clean text
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        versions['morph_close'] = morph_close
        
        # Dilated version (makes text thicker)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
        versions['dilated'] = dilated
        
        # Super resolution attempt (resize)
        height, width = gray.shape
        if width < 2000:
            scale = 2000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            versions['upscaled'] = resized
        
        return versions

    def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7):
        """Extract text using EasyOCR with confidence scores"""
        try:
            # Run EasyOCR with specific settings
            results = self.easy_reader.readtext(
                image, 
                detail=detail,
                width_ths=width_ths,
                height_ths=height_ths,
                paragraph=False,
                text_threshold=0.3,
                low_text=0.3,
                mag_ratio=1.5
            )
            
            if detail == 1:
                # Return tuples of (text, confidence)
                text_with_conf = [(result[1].upper(), result[2]) for result in results if result[2] > 0.2]
            else:
                # When detail=0, no confidence scores available
                text_with_conf = [(text.upper(), 0.0) for text in results]
            
            return text_with_conf
        except Exception as e:
            print(f"    EasyOCR error: {e}")
            return []

    def extract_text_with_tesseract(self, image):
        """Extract text using Tesseract with confidence scores"""
        text_with_conf = []
        
        # Try getting individual words with confidence scores
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            for i, word in enumerate(data['text']):
                if word.strip():
                    confidence = float(data['conf'][i])
                    if confidence > 0:  # Tesseract uses -1 for no confidence
                        text_with_conf.append((word.upper(), confidence / 100.0))  # Normalize to 0-1
        except:
            pass
        
        # Also try different configurations without confidence scores
        configs = [
            r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--psm 8',
            r'--psm 12',
            r'--psm 13',
        ]
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config)
                if text.strip():
                    # Split into words and assign default confidence
                    words = text.upper().split()
                    for word in words:
                        if word.strip():
                            text_with_conf.append((word, 0.5))  # Default confidence
            except:
                pass
        
        return text_with_conf

    def extract_text_with_paddle(self, image):
        """Extract text using PaddleOCR with confidence scores"""
        if not self.paddle_ocr:
            return []
        
        try:
            result = self.paddle_ocr.ocr(image, cls=True)
            text_with_conf = []
            if result and result[0]:
                for line in result[0]:
                    if line[1][1] > 0.3:  # Confidence threshold
                        text_with_conf.append((line[1][0].upper(), line[1][1]))
            return text_with_conf
        except Exception as e:
            print(f"    PaddleOCR error: {e}")
            return []

    def correct_ocr_errors(self, text):
        """Correct common OCR misreads"""
        corrected = text
        for wrong, right in self.ocr_corrections.items():
            corrected = corrected.replace(wrong, right)
        return corrected

    def extract_components_from_texts(self, text_with_conf_list):
        """Extract ALL text with confidence scores (no filtering)"""
        # Combine all results
        all_items = {}  # Use dict to track best confidence for each text
        
        for text, conf in text_with_conf_list:
            # Split by common separators and clean
            items = re.split(r'[\s,;:\n\r\t]+', text)
            for item in items:
                item = item.strip()
                # Apply OCR corrections
                corrected = self.correct_ocr_errors(item)
                if corrected and len(corrected) > 0:
                    # Keep the highest confidence for each unique text
                    if corrected not in all_items or conf > all_items[corrected]:
                        all_items[corrected] = conf
        
        # Convert to list of tuples sorted by confidence
        return sorted([(text, conf) for text, conf in all_items.items()], 
                     key=lambda x: x[1], reverse=True)

    def process_image(self, image_path):
        """Main processing function that returns text with confidence scores"""
        # Read image
        image = cv2.imread(image_path)
        
        # Get all preprocessing versions
        versions = self.advanced_preprocessing(image)
        
        all_text_with_conf = []  # List of (text, confidence) tuples
        
        print("  Processing with multiple OCR engines and preprocessing...")
        
        # Process each version
        for version_name, processed_img in versions.items():
            # Convert grayscale to BGR for EasyOCR if needed
            if len(processed_img.shape) == 2:
                img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
            else:
                img_for_ocr = processed_img
            
            # EasyOCR
            text_conf_list = self.extract_text_with_easyocr(img_for_ocr)
            all_text_with_conf.extend(text_conf_list)
            
            # EasyOCR with different parameters
            text_conf_list = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
            all_text_with_conf.extend(text_conf_list)
            
            # Tesseract
            text_conf_list = self.extract_text_with_tesseract(processed_img)
            all_text_with_conf.extend(text_conf_list)
            
            # PaddleOCR if available
            if PADDLE_AVAILABLE:
                text_conf_list = self.extract_text_with_paddle(img_for_ocr)
                all_text_with_conf.extend(text_conf_list)
        
        # Try cropping and processing regions
        height, width = image.shape[:2]
        quadrants = [
            (0, 0, width//2, height//2),
            (width//2, 0, width, height//2),
            (0, height//2, width//2, height),
            (width//2, height//2, width, height)
        ]
        
        for x1, y1, x2, y2 in quadrants:
            cropped = image[y1:y2, x1:x2]
            text_conf_list = self.extract_text_with_easyocr(cropped)
            all_text_with_conf.extend(text_conf_list)
        
        # Process and deduplicate
        final_results = self.extract_components_from_texts(all_text_with_conf)
        
        return final_results, versions

    def visualize_results(self, image, all_text, output_path):
        """Create visualization of detected text"""
        try:
            # Set matplotlib to use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')  # Use non-GUI backend
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # Run EasyOCR to get bounding boxes
            results = self.easy_reader.readtext(image, detail=1)
            
            fig, ax = plt.subplots(1, figsize=(15, 10))
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Draw bounding boxes for all detected text
            for i, (bbox, text, prob) in enumerate(results):
                if prob > 0.2:  # Show all text with reasonable confidence
                    # Draw bounding box
                    points = bbox
                    rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=2)
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(points[0][0], points[0][1]-5, text, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                           fontsize=8)
            
            ax.set_title(f'Circuit OCR Results - Found {len(all_text)} text items', 
                        fontsize=16, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not create visualization: {e}")
            # Create a simple text visualization instead
            with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
                f.write(f"Found {len(all_text)} text items:\n")
                f.write('\n'.join(all_text))

def process_circuit_image(image_path, output_folder, ocr_engine):
    """Process a single circuit image and extract ALL text with confidence scores"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Save original
    image = cv2.imread(image_path)
    cv2.imwrite(os.path.join(output_folder, "1_original.png"), image)
    
    # Extract all text with confidence
    text_with_conf, versions = ocr_engine.process_image(image_path)
    
    # Save some preprocessing versions for debugging
    for i, (name, img) in enumerate(list(versions.items())[:4]):
        cv2.imwrite(os.path.join(output_folder, f"2_preprocess_{name}.png"), img)
    
    # Visualize (pass just the text for visualization)
    all_text = [text for text, conf in text_with_conf]
    if all_text:
        ocr_engine.visualize_results(image, all_text, 
                                   os.path.join(output_folder, "3_detected_text.png"))
    
    # Save results with confidence scores
    with open(os.path.join(output_folder, "extracted_text_with_confidence.txt"), 'w', encoding='utf-8') as f:
        if text_with_conf:
            for text, conf in text_with_conf:
                f.write(f"{text}\t{conf:.3f}\n")
            print(f"  Found {len(text_with_conf)} text items")
            # Show top 10 with confidence
            print("  Top detections:")
            for text, conf in text_with_conf[:10]:
                print(f"    {text} (confidence: {conf:.3f})")
            if len(text_with_conf) > 10:
                print(f"    ... and {len(text_with_conf) - 10} more")
        else:
            f.write("No text detected")
            print("  No text detected")
    
    # Save just the text (without confidence) for compatibility
    with open(os.path.join(output_folder, "extracted_text.txt"), 'w', encoding='utf-8') as f:
        if all_text:
            f.write('\n'.join(all_text))
    
    # Save detailed summary
    with open(os.path.join(output_folder, "summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write(f"Total text items found: {len(text_with_conf)}\n\n")
        f.write("All extracted text with confidence scores:\n")
        f.write("-" * 50 + "\n")
        f.write("Text\t\t\tConfidence\n")
        f.write("-" * 50 + "\n")
        for text, conf in text_with_conf:
            f.write(f"{text:<20}\t{conf:.3f}\n")
    
    return text_with_conf

def process_all_images(images_folder, output_folder, use_gpu=True):
    """Process all images in folder and extract ALL text with confidence"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize OCR engine
    ocr_engine = CircuitOCR(use_gpu=use_gpu)
    
    # Get image files
    image_files = [f for f in os.listdir(images_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
    
    if not image_files:
        print(f"No image files found in {images_folder}")
        return
    
    all_text_with_conf = []
    text_per_image = {}
    
    for image_file in image_files:
        print(f"\nProcessing {image_file}...")
        
        image_path = os.path.join(images_folder, image_file)
        image_output_folder = os.path.join(output_folder, os.path.splitext(image_file)[0])
        
        try:
            text_conf_items = process_circuit_image(image_path, image_output_folder, ocr_engine)
            all_text_with_conf.extend(text_conf_items)
            text_per_image[image_file] = text_conf_items
        except Exception as e:
            print(f"  Error: {str(e)}")
            traceback.print_exc()
            text_per_image[image_file] = []
    
    # Summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    # Get unique text items
    unique_text_dict = {}
    for text, conf in all_text_with_conf:
        if text not in unique_text_dict or conf > unique_text_dict[text]:
            unique_text_dict[text] = conf
    
    print(f"Total text items found: {len(all_text_with_conf)}")
    print(f"Total unique text items: {len(unique_text_dict)}")
    print(f"Images processed: {len(image_files)}")
    
    # Save overall summary with confidence scores
    with open(os.path.join(output_folder, "overall_summary_with_confidence.txt"), 'w', encoding='utf-8') as f:
        f.write("Circuit Image Text Extraction Results (with Confidence)\n")
        f.write("="*60 + "\n\n")
        f.write(f"OCR Engines used: EasyOCR, Tesseract{', PaddleOCR' if PADDLE_AVAILABLE else ''}\n")
        f.write(f"Total images processed: {len(image_files)}\n")
        f.write(f"Total text items found: {len(all_text_with_conf)}\n")
        f.write(f"Total unique text items: {len(unique_text_dict)}\n\n")
        
        f.write("Text items per image:\n")
        f.write("-"*60 + "\n")
        for img, items in text_per_image.items():
            f.write(f"{img}: {len(items)} text items\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("All unique text items (sorted by confidence):\n")
        f.write("-"*60 + "\n")
        f.write("Text\t\t\t\tBest Confidence\n")
        f.write("-"*60 + "\n")
        
        # Sort by confidence
        sorted_items = sorted(unique_text_dict.items(), key=lambda x: x[1], reverse=True)
        for text, conf in sorted_items:
            f.write(f"{text:<30}\t{conf:.3f}\n")
    
    # Also save simple text-only version
    with open(os.path.join(output_folder, "overall_summary.txt"), 'w', encoding='utf-8') as f:
        f.write("All unique text items:\n")
        for text, _ in sorted_items:
            f.write(f"{text}\n")
