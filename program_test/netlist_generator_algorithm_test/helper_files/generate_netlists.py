import numpy as np
import os
from scipy.spatial import distance
import cv2
import json
import re
from dotenv import load_dotenv
import base64
import requests

def generate_named_netlist(raw_netlist_lines, api_lines, ocr_text, image_path=None):
    """
    Replace internal component IDs in the netlist with real labels and values
    using the API response and OCR results. Includes reasoning per component.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")

    # Combine raw netlist into text
    raw_netlist_text = "\n".join(raw_netlist_lines)

    # Extract only lines that contain label:value format
    label_lines = [line for line in api_lines if re.search(r"\)\s*[A-Za-z]+\d*:", line)]

    # Remove any parenthetical description before the label
    cleaned_label_lines = [re.sub(r"^\(.*?\)", "", line).strip() for line in label_lines]
    component_label_text = "\n".join(cleaned_label_lines)

    # Optional image attachment
    image_block = []
    if image_path and os.path.isfile(image_path):
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        image_block.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    prompt = (
    "You are a circuit assistant tasked with transforming a raw netlist into a labeled and value-annotated version.\n\n"
    "You are provided with the following data:\n"
    "- A RAW NETLIST showing internal component IDs and their node connections. Example: `Resistor_1 1 2`\n"
    "- A LABEL:VALUE list from an API vision model. Each line may have parenthetical descriptions before the label — ignore these completely. Example: `(This component was drawn like two parallel lines which fits capacitor description)C3: 100uF`\n"
    "- OCR results showing detected text with confidence scores, organized by component and proximity.\n\n"

    "Your job is to:\n"
    "1. Parse the LABEL:VALUE list. For each line, extract only the component label and its value after the colon. Completely ignore any parenthetical text before the label.\n\n"
    
    "2. For each component in the RAW NETLIST:\n"
    "   a) LABEL MATCHING: Find the real-world label using this process:\n"
    "      - Start with API labels for components of the same type\n"
    "      - Use OCR text near that component ID as hints to help match which API label belongs to which raw netlist component\n"
    "      - OCR may be noisy/inaccurate, so use it as supporting evidence, not definitive proof\n"
    "      - Match OCR hints to API labels using case-insensitive comparison and partial matching\n"
    "      - For components without useful OCR hints, assign remaining unused API labels of the same type\n"
    "      - Example: Transistor_BJT_1 has nearby OCR 'T_' and '3' → API has Tr1,Tr2,Tr3 → OCR '3' suggests this might be Tr3\n\n"
    
    "   b) VALUE ASSIGNMENT: Determine the component value using this priority:\n"
    "      - PRIMARY: Use the API's value for the matched label as the default\n"
    "      - VALIDATION: Check if the API value makes sense for the component type:\n"
    "        * Resistors should have resistance values (Ω, kΩ, MΩ)\n"
    "        * Capacitors should have capacitance values (F, uF, nF, pF)\n"
    "        * Voltage sources should have voltage values (V, mV)\n"
    "        * Current sources should have current values (A, mA, uA)\n"
    "      - ASSISTANCE: Use OCR text near the component as hints to help confirm, enhance, or correct values\n"
    "      - CORRECTION: If API value doesn't make sense, look for value hints in OCR near the component\n"
    "      - ENHANCEMENT: If API has label but no value, use OCR value hints if they make sense for the component\n\n"
    
    "3. Output format for each component:\n"
    "   - `component ID:<LABEL>, <NODE1> <NODE2>, <VALUE>` (for 2-terminal components)\n"
    "   - `component ID:<LABEL>, <NODE1> <NODE2> <NODE3>, <VALUE>` (for 3-terminal components)\n"
    "   - Include the value only if confidently determined and sensible\n"
    "   - If no confident label/value match found, copy the original raw netlist line\n\n"
    
    "4. IMPORTANT GUIDELINES:\n"
    "   - The API is your PRIMARY source, OCR is only for HINTS and ASSISTANCE\n"
    "   - OCR is ~90% accurate and noisy — use it as supporting evidence, not definitive answers\n"
    "   - Use OCR hints to help match API labels to raw netlist components (TR1 suggests Tr1)\n"
    "   - For components without useful OCR hints, assign remaining unused API labels of the same type\n"
    "   - Do NOT require minimum confidence thresholds for OCR — treat all OCR as potentially useful hints\n"
    "   - Do NOT invent any labels or values not supported by the data\n"
    "   - Do NOT include any reasoning, comments, or explanations in your output\n"
    "   - Always validate that values make sense for the component type\n\n"

    "=== RAW NETLIST ===\n"
    f"{raw_netlist_text}\n\n"
    "=== LABEL:VALUE MAPPING ===\n"
    f"{component_label_text}\n\n"
    "=== OCR RESULTS ===\n"
    f"{ocr_text}\n\n"
    "Output only the final netlist, one line per component. No comments, no reasoning, no extra text."

    "After finishing add === to a new line then provide a netlist compatible with LTspice. IMPORTANT: keep nodes names the same (the provided numbers) don't use the N### convention."
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
                "content": [{"type": "text", "text": prompt}] + image_block
            }
        ],
        "max_tokens": 1500,
        "temperature": 0
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    result = response.json()['choices'][0]['message']['content']

    return [line.strip() for line in result.splitlines() if line.strip()]


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
    results_file = os.path.join(test_results_path, os.path.splitext(image_file)[0], 'netlist_structure.txt')
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

def generate_all_netlists(approved_images_path, output_files_path, test_results_path):
    os.makedirs(test_results_path, exist_ok=True)
    image_files = [f for f in os.listdir(approved_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_basename = os.path.splitext(image_file)[0]

        # Run your custom function
        generate_netlist_for_image(image_file, output_files_path, output_files_path)

        # === Define input paths ===
        raw_netlist_path = os.path.join(output_files_path, image_basename, "netlist_structure.txt")
        api_response_path = os.path.join(output_files_path, image_basename, "API_first_response.txt")
        ocr_text_path = os.path.join(output_files_path, image_basename, "ocr_results", "circuit_components_with_text.txt")

        # === Read input files ===
        with open(raw_netlist_path, "r", encoding="utf-8") as f:
            raw_netlist = f.read().splitlines()

        with open(api_response_path, "r", encoding="utf-8") as f:
            api_lines = f.read().splitlines()

        with open(ocr_text_path, "r", encoding="utf-8") as f:
            ocr_text = f.read()

        # === Generate final netlist ===
        final_netlist = generate_named_netlist(raw_netlist, api_lines, ocr_text)

        # === Print final output ===
        print(f"\nFinal named netlist for {image_file}:\n" + "=" * 40)
        for line in final_netlist:
            print(line)

        # === Save final netlist to output_files_path/final_results/image_name.txt ===
        output_file_path = os.path.join(test_results_path, f"{image_basename}.txt")
        print(f"Saving final netlist to {output_file_path}")
        with open(output_file_path, "w", encoding="utf-8") as f:
            for line in final_netlist:
                f.write(line + "\n")
