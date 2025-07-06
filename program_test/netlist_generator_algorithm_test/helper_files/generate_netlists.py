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
    "- A LABEL:VALUE list, extracted by an API vision model. It may include some parenthetical descriptions before each label — these should be ignored.\n"
    "- OCR results that show the surrounding detected text for each component ID, with confidence scores.\n\n"

    "Your job is to:\n"
    "1. Start by parsing the LABEL:VALUE list. For each line, extract the component label and its value (e.g., `R1: 10kΩ`). Ignore any parenthetical explanation that comes before the label.\n\n"
    "2. Next, for each component in the RAW NETLIST:\n"
    "   - Match the internal ID (like `Resistor_3`) to a real-world label (like `R1`, `Qp`, `V1`) using OCR. Only accept OCR labels with confidence ≥ 0.9.\n"
    "   - Then match the component to the correct VALUE:\n"
    "     - Use the API’s value for that label as the default, if available.\n"
    "     - If the OCR includes pieces of the value (like '10' and 'V') near the component, consider that confirmation.\n"
    "     - If OCR contradicts the API, you may omit or correct the value — only if clearly wrong.\n\n"
    "3. Construct the final line for each component in this format:\n"
    "   - `<LABEL> <NODE1> <NODE2> [<VALUE>]`\n"
    "   - Or `<LABEL> <NODE1> <NODE2> <NODE3> [<VALUE>]`\n"
    "   - Include the value only if it was confidently determined.\n"
    "   - Do NOT include any reasoning or explanation in your output.\n\n"
    "4. Do NOT invent any labels or values. Only use what is supported by the API or OCR.\n"
    "5. OCR is about 90% accurate. Its purpose is to assist and confirm API values and labels, not replace them.\n\n"

    "=== RAW NETLIST ===\n"
    f"{raw_netlist_text}\n\n"
    "=== LABEL:VALUE MAPPING ===\n"
    f"{component_label_text}\n\n"
    "=== OCR RESULTS ===\n"
    f"{ocr_text}\n\n"
    "Now only output the final netlist, one line per component. No comments, no reasoning, no extra text."
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
        "temperature": 0.3
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
