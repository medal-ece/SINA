import numpy as np
import cv2
import os
from helper_files.nodes_analysis import identify_circuit_nodes
from helper_files.ocr_module import CircuitOCR, process_circuit_ocr
from helper_files.proximity_analysis import assign_text_to_nearest_component
from ultralytics import YOLO
import traceback
import json
import matplotlib
matplotlib.use('Agg') 

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


