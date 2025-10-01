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

import cv2
import numpy as np
import base64
import os
from helper_files.proximity_analysis import assign_text_to_nearest_component
import easyocr
import matplotlib
matplotlib.use('Agg')
import pytesseract
import re
import base64
import requests
import json
from collections import Counter
import shutil
from dotenv import load_dotenv

SUPPORTED_COMPONENTS = {
    "Resistor", "Capacitor", "Inductor", "Transistor_BJT", "Transistor_MOSFET",
    "Voltage_src", "Current_src", "Op-Amp", "Bridge"
}

def create_bridge_component(x, y, bridge_id, bridge_size=20):
    """
    Create a bridge component at the specified coordinates.
    Bridge components mark cut wire endpoints for crossing disambiguation.

    Args:
        x (int): X coordinate of the bridge center
        y (int): Y coordinate of the bridge center
        bridge_id (str): Unique identifier for the bridge (e.g., "Bridge_1_line1_start")
        bridge_size (int): Size of the bridge bounding box (default: 6 pixels)

    Returns:
        dict: Bridge component dictionary compatible with the component pipeline
    """
    half_size = bridge_size // 2
    return {
        "id": bridge_id,
        "type": "Bridge",
        "bbox": [x - half_size, y - half_size, x + half_size, y + half_size],
        "color": [255, 0, 255],  # Magenta color for bridge components
        "center_point": [x, y]  # Store center point for connection purposes
    }

def inject_bridge_components(components, crossing_data, disambiguation_context=None):
    """
    Inject bridge components at cut wire endpoints based on crossing data and disambiguation context.

    Args:
        components (list): List of existing components from YOLO detection
        crossing_data (dict): Crossing detection data containing line endpoints
        disambiguation_context (str): Context indicating which line should be cut ("line1_connected")

    Returns:
        list: Components list with bridge components added
    """
    if not crossing_data or not crossing_data.get('crossings') or disambiguation_context != "line1_connected":
        return components

    bridge_components = []

    for i, crossing in enumerate(crossing_data['crossings']):
        crossing_id = crossing['id']

        # Only add bridges for line1_connected processing (line2 is cut)
        cut_line = crossing['line2']
        bridge_line_name = "line2"

        # Create bridge components at both endpoints of the cut line
        start_x, start_y = cut_line['start']
        end_x, end_y = cut_line['end']

        bridge_start_id = f"Bridge_{crossing_id}_{bridge_line_name}_start"
        bridge_end_id = f"Bridge_{crossing_id}_{bridge_line_name}_end"

        bridge_start = create_bridge_component(start_x, start_y, bridge_start_id)
        bridge_end = create_bridge_component(end_x, end_y, bridge_end_id)

        bridge_components.extend([bridge_start, bridge_end])

    print(f"Added {len(bridge_components)} bridge components for disambiguation context: {disambiguation_context}")
    return components + bridge_components

# Global tracking for summary
approved_images = []
failed_images = []

def check_component_consistency(api_response_lines, circuit_json_path, image_path, output_folder, approved_folder):
    """
    Compare component type counts from OpenAI API response vs. circuit_info.json.
    Write explanation to consistency_check.txt.
    Move approved image to approved_folder and print path.
    Return True if matched, False otherwise.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "consistency_check.txt")

    # === Step 1: Parse JSON ===
    with open(circuit_json_path, "r", encoding="utf-8") as f:
        circuit_components = json.load(f)

    json_counts = Counter()
    for comp in circuit_components:
        label = comp.get("label")
        if label in SUPPORTED_COMPONENTS:
            json_counts[label] += 1

    # === Step 2: Parse API ===
    api_counts = Counter()
    inside_types = False
    for line in api_response_lines:
        if line.strip().startswith("types:"):
            inside_types = True
            continue
        if inside_types:
            if not line.strip() or ":" not in line:
                break
            try:
                comp_type, count = line.strip().split(":")
                comp_type = comp_type.strip()
                count = int(count.strip())
                if comp_type in SUPPORTED_COMPONENTS:
                    api_counts[comp_type] = count
            except Exception:
                continue

    # === Step 3: Compare and Save Output ===
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Component Consistency Check\n")
        f.write("=" * 40 + "\n")
        f.write(f"API Component Types ({sum(api_counts.values())} total):\n")
        for k, v in sorted(api_counts.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write(f"Object Detection Types ({sum(json_counts.values())} total):\n")
        for k, v in sorted(json_counts.items()):
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        # Compare differences
        passed = True
        all_keys = sorted(set(api_counts) | set(json_counts))
        for key in all_keys:
            if api_counts[key] != json_counts[key]:
                passed = False
                f.write(f"Mismatch in '{key}': API={api_counts[key]}, Detection={json_counts[key]}\n")

        f.write("\n")
        if passed:
            f.write("Result: PASS - Component counts match.\n")
        else:
            f.write("Result: FAIL - Mismatched component types or counts.\n")

    # === Step 4: Move image if passed ===
    approved_folder = os.path.join(approved_folder, "approved_images")
    os.makedirs(approved_folder, exist_ok=True)

    image_name = os.path.basename(image_path)
    if passed and os.path.isfile(image_path):
        destination = os.path.join(approved_folder, image_name)
        shutil.copy2(image_path, destination)
        approved_images.append(image_name)
    else:
        failed_images.append(image_name)

    # === Step 5: Write Summary ===
    summary_path = os.path.join(approved_folder, "approval_summary.txt")
    total = len(approved_images) + len(failed_images)
    passed_count = len(approved_images)
    failed_count = len(failed_images)
    pass_percent = (passed_count / total * 100) if total > 0 else 0

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Approval Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Passed: {passed_count}\n")
        f.write(f"Failed: {failed_count}\n")
        f.write(f"Pass Rate: {pass_percent:.2f}%\n\n")
        if failed_images:
            f.write("Failed Images:\n")
            for name in failed_images:
                f.write(f"  - {name}\n")

    return passed

def extract_components_from_image_api(image_path, output_path=None):
    load_dotenv()   
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")

    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    prompt = (
    "You are a vision model with expert knowledge in electronics and circuit schematics.\n\n"
    "Your task is to analyze the provided circuit schematic image and extract components that are actually *drawn* using proper circuit symbols.\n\n"
    
    "CRITICAL RULES:\n"
    "- ONLY include components with drawn symbols (zig-zag lines, parallel lines, transistor symbols, etc.)\n"
    "- DO NOT include text-only references like 'use 10V supply' or 'connect resistor here'\n"
    "- DO NOT include ground symbols, connection dots, wire junctions, or connecting wires\n"
    "- Skip components with no visible labels entirely\n"
    "- Think like a circuit simulator — only drawn symbols that can be electrically connected\n\n"
    
    "COMPONENT SYMBOLS TO RECOGNIZE:\n"
    "- Resistor: zig-zag line OR rectangular box\n"
    "- Capacitor: two parallel lines OR curved line with straight line (treat all as 'Capacitor')\n"
    "- Inductor: series of loops/coils\n"
    "- Transistor_BJT: triangle with lines (any BJT symbol regardless of NPN/PNP)\n"
    "- Transistor_MOSFET: gate symbol with channel\n"
    "- Voltage_src: circle with +/- OR battery symbol\n"
    "- Current_src: circle with arrow\n"
    "- Op-Amp: triangle amplifier symbol\n\n"
    
    "LABEL AND VALUE EXTRACTION:\n"
    "- For each drawn component, find nearby text that serves as its label (R1, C2, Tr1, V_supply, etc.)\n"
    "- Accept any naming convention that appears to be a component label\n"
    "- Extract the component value if visible (10kΩ, 1µF, 12V, etc.)\n"
    "- If multiple values are present, include all of them\n"
    "- If labels are partially obscured, make your best guess\n"
    "- Extract values in their exact format as shown (don't convert '10000 ohms' to '10kΩ')\n\n"
    
    "MANDATORY FORMAT:\n"
    "For each component, you MUST include a description of how it's drawn:\n"
    "\"(This component was drawn like [description of the actual symbol you see])\"\n\n"
    
    "=== OUTPUT FORMAT ===\n"
    "number of components: <TOTAL_COUNT>\n"
    "types:\n"
    "  <Type>: <count_of_that_type>\n\n"
    
    "For each component:\n"
    "(This component was drawn like [symbol description])ComponentLabel: ComponentValue\n\n"
    
    "EXAMPLES:\n"
    "(This component was drawn like zig-zag line which fits resistor description)R1: 10kΩ\n"
    "(This component was drawn like two parallel lines which fits capacitor description)C2: 1µF\n"
    "(This component was drawn like a series of loops which fits inductor description)L1: 100mH\n"
    "(This component was drawn like a triangle with three lines which fits BJT description)Q1: 2N3904\n"
    "(This component was drawn like a circle with + and - symbols which fits voltage source description)V_bat: 9V\n"
    "(This component was drawn like zig-zag line which fits resistor description)R_load:\n"
    
    "EXAMPLE OUTPUT:\n"
    "number of components: 6\n"
    "types:\n"
    "  Resistor: 2\n"
    "  Capacitor: 1\n"
    "  Inductor: 1\n"
    "  Transistor_BJT: 1\n"
    "  Voltage_src: 1\n\n"
    
    "IMPORTANT NOTES:\n"
    "- If component value is unknown/unclear, leave blank after colon (e.g., 'C1:')\n"
    "- Keep original label names exactly as written (don't standardize them)\n"
    "- If multiple values are near one component, include all: 'R1: 10kΩ, 5%, 0.25W'\n"
    "- Only count actual drawn symbols, not text references\n\n"
    
    "Return only the formatted output. No extra explanations."
)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 1800,
        "temperature": 0
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    response.raise_for_status()

    result_text = response.json()['choices'][0]['message']['content'].strip()

    # Save response
    if output_path is None:
        output_path = os.path.join(os.path.dirname(image_path), "API_first_response.txt")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result_text)

    return result_text.splitlines()


def process_circuit_image(image_path, model_path, output_folder, use_ocr=True, use_gpu=True, detect_crossings=True, crossing_data=None, disambiguation_context=None):
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

# Bridge components will be added after circuit cut in identify_circuit_nodes()

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

    # Call the circuit node identification with crossing detection parameters
    identify_circuit_nodes(merged_labels, components, output_folder, original_image, model_path, detect_crossings=detect_crossings, crossing_data=crossing_data, disambiguation_context=disambiguation_context)
    
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


    
def process_all_images(images_folder, model_path, output_folder, use_ocr=True, use_gpu=True, detect_crossings=True):
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
            process_circuit_image(image_path, model_path, image_output_folder, use_ocr=use_ocr, use_gpu=use_gpu, detect_crossings=detect_crossings, crossing_data=None, disambiguation_context=None)
        except Exception as e:
            print(f"  Error processing {image_file}: {str(e)}")
            traceback.print_exc()


