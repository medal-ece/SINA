from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import os
import random
from collections import defaultdict
from ultralytics import YOLO
import traceback
import json
import re
import easyocr
import pytesseract
import matplotlib
matplotlib.use('Agg') 


# ===== OCR CLASS DEFINITION =====
class CircuitOCR:
    def __init__(self, use_gpu=True):
        """Initialize OCR engines"""
        print("Initializing OCR engines...")
        
        # Initialize EasyOCR (this works with your current setup)
        self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
         
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
        """Extract text using EasyOCR with confidence scores and bounding boxes"""
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
                # Return tuples of (text, confidence, bbox)
                text_with_conf_bbox = []
                for result in results:
                    if result[2] > 0.2:  # confidence threshold
                        bbox_points = result[0]
                        # Calculate center point of bbox
                        x_coords = [p[0] for p in bbox_points]
                        y_coords = [p[1] for p in bbox_points]
                        center_x = sum(x_coords) / len(x_coords)
                        center_y = sum(y_coords) / len(y_coords)
                        
                        text_with_conf_bbox.append({
                            'text': result[1].upper(),
                            'confidence': result[2],
                            'bbox': bbox_points,
                            'center': (center_x, center_y)
                        })
            else:
                # When detail=0, no bbox available
                text_with_conf_bbox = [{'text': text.upper(), 'confidence': 0.0, 'bbox': None, 'center': None} 
                                      for text in results]
            
            return text_with_conf_bbox
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

    def correct_ocr_errors(self, text):
        """Correct common OCR misreads"""
        corrected = text
        for wrong, right in self.ocr_corrections.items():
            corrected = corrected.replace(wrong, right)
        return corrected

    def extract_components_from_texts(self, text_with_conf_bbox_list):
        """Extract ALL text with confidence scores and locations (no filtering)"""
        # Combine all results
        all_items = {}  # Use dict to track best result for each text
        
        for item in text_with_conf_bbox_list:
            text = item.get('text', '')
            conf = item.get('confidence', 0.0)
            bbox = item.get('bbox', None)
            center = item.get('center', None)
            
            # Split by common separators and clean
            items = re.split(r'[\s,;:\n\r\t]+', text)
            for sub_item in items:
                sub_item = sub_item.strip()
                # Apply OCR corrections
                corrected = self.correct_ocr_errors(sub_item)
                if corrected and len(corrected) > 0:
                    # Keep the highest confidence version with location info
                    if corrected not in all_items or conf > all_items[corrected]['confidence']:
                        all_items[corrected] = {
                            'text': corrected,
                            'confidence': conf,
                            'bbox': bbox,
                            'center': center
                        }
        
        # Convert to list sorted by confidence
        return sorted(all_items.values(), key=lambda x: x['confidence'], reverse=True)

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

# ===== HELPER FUNCTIONS FOR TEXT-COMPONENT PROXIMITY =====
def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    if point1 is None or point2 is None:
        return float('inf')
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_component_center(bbox):
    """Get center point of a component bounding box"""
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)

def assign_text_to_nearest_component(ocr_results, components):
    """Assign each text to its nearest component (each text appears only once)"""
    # First, find the nearest component for each text
    text_assignments = {}  # text -> (component_id, distance)
    
    for ocr_item in ocr_results:
        text = ocr_item['text']
        text_center = ocr_item.get('center', None)
        
        if text_center:
            min_distance = float('inf')
            nearest_component = None
            
            # Find the closest component
            for component in components:
                comp_center = get_component_center(component["bbox"])
                distance = calculate_distance(comp_center, text_center)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_component = component["id"]
            
            if nearest_component:
                text_assignments[text] = {
                    'component_id': nearest_component,
                    'distance': min_distance,
                    'confidence': ocr_item['confidence'],
                    'center': text_center
                }
    
    # Now organize by component
    component_text_mapping = {}
    for component in components:
        comp_id = component["id"]
        component_text_mapping[comp_id] = []
    
    # Assign texts to their nearest components
    for text, assignment in text_assignments.items():
        comp_id = assignment['component_id']
        component_text_mapping[comp_id].append({
            'text': text,
            'confidence': assignment['confidence'],
            'distance': assignment['distance'],
            'center': assignment['center']
        })
    
    # Sort each component's texts by distance
    for comp_id in component_text_mapping:
        component_text_mapping[comp_id].sort(key=lambda x: x['distance'])
    
    return component_text_mapping

# ===== OCR PROCESSING FUNCTION =====
def process_circuit_ocr(image_path, output_folder, ocr_engine, components=None):
    """Process OCR for circuit image and extract ALL text with confidence scores and proximity to components"""
    print("\n  Running OCR text extraction...")
    
    # Create OCR subfolder
    ocr_output_folder = os.path.join(output_folder, "ocr_results")
    os.makedirs(ocr_output_folder, exist_ok=True)
    
    # Read image
    image = cv2.imread(image_path)
    
    # Extract all text with confidence and location
    text_with_info, versions = ocr_engine.process_image(image_path)
    
    # Save some preprocessing versions for debugging
    for i, (name, img) in enumerate(list(versions.items())[:4]):
        cv2.imwrite(os.path.join(ocr_output_folder, f"ocr_preprocess_{name}.png"), img)
    
    # Visualize (pass just the text for visualization)
    all_text = [item['text'] for item in text_with_info]
    if all_text:
        ocr_engine.visualize_results(image, all_text, 
                                   os.path.join(ocr_output_folder, "ocr_detected_text.png"))
    
    # Save results with confidence scores
    with open(os.path.join(ocr_output_folder, "extracted_text_with_confidence.txt"), 'w', encoding='utf-8') as f:
        if text_with_info:
            for item in text_with_info:
                f.write(f"{item['text']}\t{item['confidence']:.3f}\n")
            print(f"    Found {len(text_with_info)} text items")
            # Show top 10 with confidence
            print("    Top OCR detections:")
            for item in text_with_info[:10]:
                print(f"      {item['text']} (confidence: {item['confidence']:.3f})")
            if len(text_with_info) > 10:
                print(f"      ... and {len(text_with_info) - 10} more")
        else:
            f.write("No text detected")
            print("    No text detected")
    
    # Save just the text (without confidence) for compatibility
    with open(os.path.join(ocr_output_folder, "extracted_text.txt"), 'w', encoding='utf-8') as f:
        if all_text:
            f.write('\n'.join(all_text))
    
    # If components are provided, create proximity-based mapping
    if components and text_with_info:
        print("    Analyzing text proximity to components...")
        component_text_mapping = assign_text_to_nearest_component(text_with_info, components)
        
        # Save proximity-based summary
        with open(os.path.join(ocr_output_folder, "text_by_component_proximity.txt"), 'w', encoding='utf-8') as f:
            f.write("OCR Text Assigned to Nearest Components\n")
            f.write("="*60 + "\n")
            f.write("Note: Each text appears only under its closest component\n")
            f.write("="*60 + "\n\n")
            
            for comp_id, text_items in component_text_mapping.items():
                # Find component info
                comp_info = next((c for c in components if c['id'] == comp_id), None)
                if comp_info:
                    f.write(f"Component: {comp_id} (Type: {comp_info['type']})\n")
                    f.write(f"Bounding Box: {comp_info['bbox']}\n")
                    f.write("-"*40 + "\n")
                    
                    if text_items:
                        f.write(f"Assigned Text ({len(text_items)} items, sorted by distance):\n")
                        for i, item in enumerate(text_items):
                            f.write(f"  {i+1}. {item['text']:<15} (conf: {item['confidence']:.3f}, dist: {item['distance']:.1f}px)\n")
                    else:
                        f.write("  No text assigned to this component\n")
                    f.write("\n")
        
        # Create visual proximity map showing assignments
        proximity_vis = image.copy()
        
        # Color map for components
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # Draw components and their assigned texts
        for idx, (comp_id, text_items) in enumerate(component_text_mapping.items()):
            comp_info = next((c for c in components if c['id'] == comp_id), None)
            if comp_info:
                x_min, y_min, x_max, y_max = comp_info['bbox']
                color = colors[idx % len(colors)]
                
                # Draw component box
                cv2.rectangle(proximity_vis, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(proximity_vis, comp_id, (x_min, y_min - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw lines to all assigned texts
                comp_center = get_component_center(comp_info['bbox'])
                for text_item in text_items:
                    if text_item['center']:
                        # Draw line from component to text
                        cv2.line(proximity_vis, 
                                (int(comp_center[0]), int(comp_center[1])), 
                                (int(text_item['center'][0]), int(text_item['center'][1])), 
                                color, 1)
                        # Draw text with same color as component
                        cv2.putText(proximity_vis, text_item['text'], 
                                   (int(text_item['center'][0]), int(text_item['center'][1]) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite(os.path.join(ocr_output_folder, "text_component_assignments.png"), proximity_vis)
        
        # Also create a summary statistics file
        with open(os.path.join(ocr_output_folder, "assignment_statistics.txt"), 'w', encoding='utf-8') as f:
            f.write("Text Assignment Statistics\n")
            f.write("="*40 + "\n\n")
            
            total_texts = sum(len(items) for items in component_text_mapping.values())
            f.write(f"Total texts detected: {len(text_with_info)}\n")
            f.write(f"Total texts with locations: {total_texts}\n")
            f.write(f"Total components: {len(components)}\n\n")
            
            f.write("Texts per component:\n")
            for comp_id, text_items in component_text_mapping.items():
                comp_info = next((c for c in components if c['id'] == comp_id), None)
                if comp_info:
                    f.write(f"  {comp_id} ({comp_info['type']}): {len(text_items)} texts\n")
    
    # Save detailed summary
    with open(os.path.join(ocr_output_folder, "ocr_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write(f"Total text items found: {len(text_with_info)}\n\n")
        f.write("All extracted text with confidence scores:\n")
        f.write("-" * 50 + "\n")
        f.write("Text\t\t\tConfidence\n")
        f.write("-" * 50 + "\n")
        for item in text_with_info:
            f.write(f"{item['text']:<20}\t{item['confidence']:.3f}\n")
    
    return text_with_info

# ===== ORIGINAL CIRCUIT PROCESSING FUNCTIONS =====
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

def process_circuit_image(image_path, model_path, output_folder, use_ocr=True, use_gpu=True):
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

        color = (255, 155, 0)  # Use one color (cyan) for all boxes

        cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 1)
        cv2.putText(detection_image, f"{class_name}",
                    (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

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

    # Call the original circuit node identification
    identify_circuit_nodes(merged_labels, components, output_folder, original_image)
    
    # ===== ADD OCR PROCESSING HERE =====
    if use_ocr:
        try:
            # Initialize OCR engine (only once)
            ocr_engine = CircuitOCR(use_gpu=use_gpu)
            
            # Process OCR with components for proximity analysis
            ocr_results = process_circuit_ocr(image_path, output_folder, ocr_engine, components)
            
            # Save OCR results to JSON for potential future integration
            ocr_json_path = os.path.join(output_folder, "ocr_results.json")
            ocr_data = {
                "total_text_items": len(ocr_results),
                "text_with_confidence": [
                    {
                        "text": item['text'], 
                        "confidence": float(item['confidence']),
                        "center": item.get('center', None)
                    } 
                    for item in ocr_results
                ]
            }
            
            # Add proximity mapping to JSON if components exist
            if components:
                component_text_mapping = assign_text_to_nearest_component(ocr_results, components)
                ocr_data["text_by_component"] = {}
                for comp_id, text_items in component_text_mapping.items():
                    ocr_data["text_by_component"][comp_id] = [
                        {
                            "text": item['text'],
                            "confidence": float(item['confidence']),
                            "distance": float(item['distance'])
                        }
                        for item in text_items
                    ]
            
            with open(ocr_json_path, 'w') as f:
                json.dump(ocr_data, f, indent=4)
            print(f"  Saved OCR results to {ocr_json_path}")
            
        except Exception as e:
            print(f"  Warning: OCR processing failed: {str(e)}")
            traceback.print_exc()

def process_all_images(images_folder, model_path, output_folder, use_ocr=True, use_gpu=True):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(images_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Initialize OCR engine once for all images (if using OCR)
    ocr_engine = None
    if use_ocr:
        try:
            ocr_engine = CircuitOCR(use_gpu=use_gpu)
        except Exception as e:
            print(f"Warning: Could not initialize OCR engine: {str(e)}")
            use_ocr = False
    
    for image_file in image_files:
        print("\n")
        print(f"Processing {image_file}...")
        image_path = os.path.join(images_folder, image_file)
        image_output_folder = os.path.join(output_folder,
                                           os.path.splitext(image_file)[0])
        os.makedirs(image_output_folder, exist_ok=True)
        try:
            process_circuit_image(image_path, model_path, image_output_folder, use_ocr=use_ocr, use_gpu=use_gpu)
        except Exception as e:
            print(f"  Error processing {image_file}: {str(e)}")
            traceback.print_exc()


