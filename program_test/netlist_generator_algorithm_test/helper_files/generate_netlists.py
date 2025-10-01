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
    "      - Consider OCR text near that component ID as helpful guidance when available\n"
    "      - OCR provides valuable hints but is not always accurate (~90%) - weigh OCR evidence alongside other factors\n"
    "      - When OCR strongly suggests a specific label (high confidence, clear match), give it significant consideration\n"
    "      - When OCR is unclear or conflicts with logical assignment patterns, rely more on API reasoning\n"
    "      - Use case-insensitive comparison and partial matching for OCR hints\n"
    "      - Example: Transistor_BJT_1 has nearby OCR 'Q1' (95% confidence) → this strongly suggests label Q1\n"
    "      - Example: Transistor_BJT_2 has nearby OCR 'T_' (low confidence) → use as weak hint, rely more on API logic\n\n"
    
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
    "   - Balance API reasoning with OCR guidance - neither should be ignored\n"
    "   - OCR provides valuable hints (~90% accuracy) but should be weighed against logical patterns\n"
    "   - Strong OCR evidence (high confidence, clear text) should influence assignments significantly\n"
    "   - Weak OCR evidence (low confidence, unclear text) should be treated as minor hints\n"
    "   - Consider OCR confidence scores when available to gauge reliability\n"
    "   - For components without useful OCR hints, rely on API reasoning and assignment patterns\n"
    "   - Do NOT invent any labels or values not supported by the data\n"
    "   - Do NOT include any reasoning, comments, or explanations in your output\n"
    "   - Always validate that values make sense for the component type\n\n"

    "5. FINAL REVIEW STEP:\n"
    "   - Before submitting your answer, review your complete netlist for consistency\n"
    "   - Check that no component labels are duplicated (e.g., two different components both labeled Q1)\n"
    "   - Ensure all available labels of each type are used before any duplicates occur\n"
    "   - Verify that component types match their assigned labels (transistors get Q labels, resistors get R labels, etc.)\n"
    "   - If you find duplicates, reassign one of them to an unused label of the same type\n\n"

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

def identify_crossing_nodes(crossing_data, node_labels, region_to_node_mapping=None):
    """
    Identifies which final node IDs contain crossing points.
    Returns a set of final node IDs that have crossings.

    Args:
        crossing_data: JSON data from crossing detection
        node_labels: The node labels array (regions)
        region_to_node_mapping: Mapping from regions to final node IDs
    """
    crossing_regions = set()
    crossing_nodes = set()

    if not crossing_data or not crossing_data.get('crossings'):
        return crossing_nodes

    h, w = node_labels.shape

    # First, identify which regions contain crossings
    for crossing in crossing_data['crossings']:
        x, y = crossing['crossing_point']

        # Check bounds and find regions at crossing point
        regions_at_crossing = set()

        # Check the exact point
        if 0 <= y < h and 0 <= x < w:
            region_id = node_labels[y, x]
            if region_id > 0:
                regions_at_crossing.add(region_id)

        # Also check nearby pixels in case the crossing point is slightly off
        for dx in [-3, -2, -1, 0, 1, 2, 3]:
            for dy in [-3, -2, -1, 0, 1, 2, 3]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    region_id = node_labels[ny, nx]
                    if region_id > 0:
                        regions_at_crossing.add(region_id)

        crossing_regions.update(regions_at_crossing)
        print(f"Crossing at ({x}, {y}) found in regions: {regions_at_crossing}")

    # Map regions to final node IDs if mapping is provided
    if region_to_node_mapping:
        for region in crossing_regions:
            if region in region_to_node_mapping:
                final_node_id = region_to_node_mapping[region]
                crossing_nodes.add(final_node_id)
                print(f"Region {region} -> Final node {final_node_id} (has crossing)")
    else:
        # If no mapping provided, use region IDs directly
        crossing_nodes = crossing_regions

    return crossing_nodes

def create_crossing_enhanced_netlist(raw_netlist_lines, crossing_nodes, region_to_node_mapping=None):
    """
    Creates an enhanced netlist with crossing markers (c prefix) for nodes that contain crossings.

    Args:
        raw_netlist_lines: List of netlist lines (e.g., ["Resistor_1 1 2", "Capacitor_1 2 3"])
        crossing_nodes: Set of node IDs that have crossings
        region_to_node_mapping: Optional mapping from regions to final node IDs

    Returns:
        List of enhanced netlist lines with crossing markers
    """
    enhanced_lines = []

    for line in raw_netlist_lines:
        if not line.strip():
            enhanced_lines.append(line)
            continue

        parts = line.strip().split()
        if len(parts) < 2:
            enhanced_lines.append(line)
            continue

        component = parts[0]
        nodes = parts[1:]

        # Apply crossing markers to nodes
        enhanced_nodes = []
        for node_str in nodes:
            try:
                node_id = int(node_str)
                if node_id in crossing_nodes:
                    enhanced_nodes.append(f"c{node_id}")
                else:
                    enhanced_nodes.append(node_str)
            except ValueError:
                # Non-numeric node (shouldn't happen in our case, but handle gracefully)
                enhanced_nodes.append(node_str)

        enhanced_line = f"{component} {' '.join(enhanced_nodes)}"
        enhanced_lines.append(enhanced_line)

    return enhanced_lines

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

    # Save region-to-node mapping for crossing detection use
    results_folder = os.path.join(test_results_path, os.path.splitext(image_file)[0])
    os.makedirs(results_folder, exist_ok=True)
    mapping_file = os.path.join(results_folder, 'region_to_node_mapping.json')
    with open(mapping_file, 'w') as f:
        # Convert numpy int32 keys to regular int for JSON serialization
        mapping_for_json = {int(k): int(v) for k, v in new_region_to_node.items()}
        json.dump(mapping_for_json, f, indent=4)

    # Save node connection results
    results_file = os.path.join(results_folder, 'netlist_structure.txt')
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

def generate_netlist_for_image(image_file, output_files_path, test_results_path, create_test_netlist=True):
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

    # Generate standard netlist
    overlay_and_find_nodes_with_connected_regions(node_labels, components, test_results_path, image_file)

    # Create test_netlist with crossing markers if crossing data exists and is requested
    if create_test_netlist:
        crossing_data_path = os.path.join(image_output_folder, "crossing_data.json")
        if os.path.exists(crossing_data_path):
            with open(crossing_data_path, 'r') as f:
                crossing_data = json.load(f)

            # Load region-to-node mapping
            mapping_file = os.path.join(test_results_path, image_name, 'region_to_node_mapping.json')
            region_to_node_mapping = {}
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    region_to_node_mapping = json.load(f)
                    # Convert string keys back to integers
                    region_to_node_mapping = {int(k): v for k, v in region_to_node_mapping.items()}

            # Identify nodes that contain crossings
            crossing_nodes = identify_crossing_nodes(crossing_data, node_labels, region_to_node_mapping)

            if crossing_nodes:
                print(f"Found crossings in final nodes: {sorted(crossing_nodes)}")

                # Create test_netlist directory
                test_netlist_path = os.path.join(test_results_path, "test_netlist")
                os.makedirs(test_netlist_path, exist_ok=True)

                # Read the standard netlist structure
                standard_netlist_path = os.path.join(test_results_path, image_name, 'netlist_structure.txt')
                if os.path.exists(standard_netlist_path):
                    with open(standard_netlist_path, 'r') as f:
                        raw_netlist = f.read().splitlines()

                    # Create enhanced netlist with crossing markers
                    enhanced_netlist = create_crossing_enhanced_netlist(raw_netlist, crossing_nodes)

                    # Save enhanced netlist
                    test_netlist_file = os.path.join(test_netlist_path, f"{image_name}_with_crossings.txt")
                    with open(test_netlist_file, 'w') as f:
                        for line in enhanced_netlist:
                            f.write(line + "\n")

                    print(f"Created test netlist with crossing markers: {test_netlist_file}")

                else:
                    print(f"Warning: Standard netlist not found at {standard_netlist_path}")
            else:
                print("No crossings detected or no crossing nodes identified.")
        else:
            print(f"No crossing data found at {crossing_data_path}")

def generate_all_netlists(approved_images_path, output_files_path, test_results_path, create_test_netlist=True):
    os.makedirs(test_results_path, exist_ok=True)
    image_files = [f for f in os.listdir(approved_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_basename = os.path.splitext(image_file)[0]

        generate_netlist_for_image(image_file, output_files_path, test_results_path, create_test_netlist)

        # === Define input paths ===
        raw_netlist_path = os.path.join(test_results_path, image_basename, "netlist_structure.txt")
        api_response_path = os.path.join(output_files_path, image_basename, "API_first_response.txt")
        ocr_text_path = os.path.join(output_files_path, image_basename, "ocr_results", "circuit_components_with_text.txt")

        # === Check for crossing disambiguation netlist ===
        crossing_netlist_path = os.path.join(output_files_path, image_basename, "crossing_disambiguation", "crossing_1_line1_connected_netlist.txt")

        # === Read input files ===
        if os.path.exists(crossing_netlist_path):
            print(f"Using crossing disambiguation netlist for {image_basename}")
            raw_netlist = clean_crossing_netlist(crossing_netlist_path)
        else:
            print(f"Using normal netlist structure for {image_basename}")
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

        # === Create enhanced final netlist with crossing markers for test_netlist ===
        if create_test_netlist:
            crossing_data_path = os.path.join(output_files_path, image_basename, "crossing_data.json")
            if os.path.exists(crossing_data_path):
                with open(crossing_data_path, 'r') as f:
                    crossing_data = json.load(f)

                # Load node_labels and mapping to identify crossing nodes
                node_labels_path = os.path.join(output_files_path, image_basename, "node_labels.npy")
                mapping_file = os.path.join(test_results_path, image_basename, 'region_to_node_mapping.json')

                if os.path.exists(node_labels_path) and os.path.exists(mapping_file):
                    node_labels = np.load(node_labels_path)

                    with open(mapping_file, 'r') as f:
                        region_to_node_mapping = json.load(f)
                        region_to_node_mapping = {int(k): v for k, v in region_to_node_mapping.items()}

                    crossing_nodes = identify_crossing_nodes(crossing_data, node_labels, region_to_node_mapping)

                    if crossing_nodes:
                        # Create enhanced final netlist with crossing markers
                        enhanced_final_netlist = create_crossing_enhanced_netlist(final_netlist, crossing_nodes)

                        # Save enhanced final netlist
                        test_netlist_path = os.path.join(test_results_path, "test_netlist")
                        os.makedirs(test_netlist_path, exist_ok=True)
                        enhanced_output_path = os.path.join(test_netlist_path, f"{image_basename}_final_with_crossings.txt")

                        with open(enhanced_output_path, "w", encoding="utf-8") as f:
                            for line in enhanced_final_netlist:
                                f.write(line + "\n")

                        print(f"Created enhanced final netlist with crossing markers: {enhanced_output_path}")

                        # === Print enhanced final output ===
                        print(f"\nEnhanced final netlist with crossings for {image_file}:\n" + "=" * 50)
                        for line in enhanced_final_netlist:
                            print(line)

                        # === Generate crossing disambiguation images and netlists ===
                        generate_crossing_disambiguation_images_and_netlists(
                            image_file, output_files_path, test_results_path, crossing_data, image_basename
                        )


def generate_crossing_disambiguation_images_and_netlists(image_file, output_files_path, test_results_path, crossing_data, image_basename):
    """
    Generate modified images with crossings removed and specific lines connected,
    then run the full pipeline on each to get actual netlists.
    """

    if not crossing_data or not crossing_data.get('crossings'):
        return

    print(f"\n" + "=" * 80)
    print(f"GENERATING CROSSING DISAMBIGUATION IMAGES AND NETLISTS")
    print(f"Original image: {image_file}")
    print(f"Found {len(crossing_data['crossings'])} crossings to process")
    print(f"=" * 80)

    # Load the original input image
    # Try to find the original image in the inputs folder
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_image_path = os.path.join(script_dir, 'inputs', 'test_images', image_file)

    if not os.path.exists(original_image_path):
        print(f"Warning: Could not find original input image at {original_image_path}")
        return

    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"Warning: Could not load original image")
        return

    print(f"Loaded original image from: {original_image_path}")

    # Create disambiguation folder
    disambiguation_folder = os.path.join(output_files_path, image_basename, "crossing_disambiguation")
    os.makedirs(disambiguation_folder, exist_ok=True)

    # Process each crossing
    for crossing_idx, crossing in enumerate(crossing_data['crossings']):
        crossing_id = crossing['id']
        crossing_point = crossing['crossing_point']
        line1 = crossing['line1']
        line2 = crossing['line2']

        print(f"\nProcessing Crossing {crossing_id} at {crossing_point}")
        print(f"Line 1 ({line1['type']}): {line1['start']} -> {line1['end']}")
        print(f"Line 2 ({line2['type']}): {line2['start']} -> {line2['end']}")

        # Create line1-only image (connect line1, remove crossing)
        line1_image = create_single_line_image(original_image, crossing_point, line1, line2, connect_line1=True)
        line1_image_path = os.path.join(disambiguation_folder, f"crossing_{crossing_id}_line1_connected.png")
        cv2.imwrite(line1_image_path, line1_image)
        print(f"Created line1-only image: {line1_image_path}")

        # Create line2-only image (connect line2, remove crossing)
        line2_image = create_single_line_image(original_image, crossing_point, line1, line2, connect_line1=False)
        line2_image_path = os.path.join(disambiguation_folder, f"crossing_{crossing_id}_line2_connected.png")
        cv2.imwrite(line2_image_path, line2_image)
        print(f"Created line2-only image: {line2_image_path}")

        # Run full pipeline on each modified image
        print(f"Running pipeline on line1-connected image...")
        run_pipeline_on_disambiguation_image(line1_image_path, f"crossing_{crossing_id}_line1_connected", disambiguation_folder)

        print(f"Running pipeline on line2-connected image...")
        run_pipeline_on_disambiguation_image(line2_image_path, f"crossing_{crossing_id}_line2_connected", disambiguation_folder)


def create_single_line_image(original_image, crossing_point, line1, line2, connect_line1=True):
    """
    Create a modified image where the crossing is removed and only one line is connected.

    Args:
        original_image: The original image
        crossing_point: [x, y] coordinates of the crossing
        line1: Line 1 data with start/end points
        line2: Line 2 data with start/end points
        connect_line1: If True, connect line1. If False, connect line2.
    """

    # Create a copy of the original image
    modified_image = original_image.copy()

    crossing_x, crossing_y = crossing_point

    # Step 1: Get the color at the crossing point before removing it
    crossing_color = modified_image[crossing_y, crossing_x].copy()
    print(f"  Crossing color at ({crossing_x}, {crossing_y}): {crossing_color}")

    # Step 2: Remove the crossing point by drawing a circle (background color)
    # Determine background color by looking at corners of image
    background_color = modified_image[0, 0].copy()  # Use top-left corner as background
    cv2.circle(modified_image, (crossing_x, crossing_y), 6, background_color.tolist(), -1)
    print(f"  Removed crossing point with background color: {background_color}")

    # Step 3: Draw the selected line using the original crossing color
    if connect_line1:
        start1 = tuple(line1['start'])
        end1 = tuple(line1['end'])
        cv2.line(modified_image, start1, end1, crossing_color.tolist(), 2)
        print(f"  Connected line1: {start1} -> {end1} with color {crossing_color}")
    else:
        start2 = tuple(line2['start'])
        end2 = tuple(line2['end'])
        cv2.line(modified_image, start2, end2, crossing_color.tolist(), 2)
        print(f"  Connected line2: {start2} -> {end2} with color {crossing_color}")

    return modified_image


def run_pipeline_on_disambiguation_image(image_path, image_name, disambiguation_folder):
    """
    Run the full netlist generation pipeline on a disambiguation image.
    """
    import sys
    import os
    from helper_files.process_images import process_circuit_image

    print(f"  Running full pipeline on {image_name}...")

    try:
        # Get model path (same as main pipeline)
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        PROJECT_PATH = os.path.dirname(os.path.dirname(script_dir))
        model_folder = os.path.join(PROJECT_PATH, 'current_trained_model')
        train_folders = [folder for folder in os.listdir(model_folder) if folder.startswith('train')]

        if not train_folders:
            print(f"  No model found")
            return

        def extract_suffix(folder_name):
            if folder_name == "train":
                return 0
            else:
                return int(folder_name[5:])

        latest_train_folder = max(train_folders, key=extract_suffix)
        model_path = os.path.join(model_folder, latest_train_folder, 'weights', 'last.pt')

        # Create temporary output folder for this disambiguation image
        temp_output_folder = os.path.join(disambiguation_folder, f"{image_name}_processing")
        os.makedirs(temp_output_folder, exist_ok=True)

        # Load crossing data for bridge component injection
        crossing_data = None
        disambiguation_context = None

        # Determine disambiguation context from image name and load crossing data
        if "line1_connected" in image_name:
            disambiguation_context = "line1_connected"
        elif "line2_connected" in image_name:
            disambiguation_context = "line2_connected"

        if disambiguation_context:
            # Try to find crossing_data.json in the parent debugging folder
            debug_folder = os.path.dirname(disambiguation_folder)
            crossing_data_path = os.path.join(debug_folder, 'crossing_data.json')
            if os.path.exists(crossing_data_path):
                try:
                    with open(crossing_data_path, 'r') as f:
                        crossing_data = json.load(f)
                    print(f"  Loaded crossing data for bridge injection: {disambiguation_context}")
                except Exception as e:
                    print(f"  Failed to load crossing data: {e}")

        # Process the disambiguation image with bridge component injection
        print(f"  Processing image with YOLO and bridge components...")
        process_circuit_image(image_path, model_path, temp_output_folder, use_ocr=False, use_gpu=True, detect_crossings=False, crossing_data=crossing_data, disambiguation_context=disambiguation_context)

        # Generate netlist from the processed components
        print(f"  Generating netlist...")
        circuit_info_path = os.path.join(temp_output_folder, 'circuit_info.json')
        node_labels_path = os.path.join(temp_output_folder, 'node_labels.npy')

        if os.path.exists(circuit_info_path) and os.path.exists(node_labels_path):
            # Generate netlist using same logic as main pipeline
            with open(circuit_info_path, 'r') as f:
                components = json.load(f)

            node_labels = np.load(node_labels_path)

            # Generate connections (simplified version)
            netlist_lines = generate_disambiguation_netlist(components, node_labels)

            # Save netlist
            netlist_path = os.path.join(disambiguation_folder, f"{image_name}_netlist.txt")
            with open(netlist_path, 'w') as f:
                f.write(f"# Disambiguation netlist for {image_name}\n")
                f.write(f"# Generated from: {image_path}\n")
                f.write(f"# This shows actual connections when crossing is resolved\n\n")
                for line in netlist_lines:
                    f.write(line + "\n")

            print(f"  Generated netlist: {netlist_path}")

            # Print netlist to terminal
            print(f"\nNETLIST FOR {image_name.upper()}:")
            print("=" * 60)
            for line in netlist_lines:
                print(line)
            print("=" * 60)

        else:
            print(f"  Failed to generate circuit info or node labels")

    except Exception as e:
        print(f"  Error running pipeline: {e}")
        import traceback
        traceback.print_exc()


def generate_disambiguation_netlist(components, node_labels):
    """
    Generate a simplified netlist from components and node labels.
    """
    from scipy.spatial import distance

    netlist_lines = []
    region_to_node = {}
    current_node_id = 1
    height, width = node_labels.shape

    # Map regions to nodes (simplified version of main logic)
    for component in components:
        comp_id = component["id"]
        connected_nodes = []

        for point in component["connection_points"]:
            px, py = point
            if py >= height or px >= width:
                continue

            # Find nearest edge point
            edge_points = np.argwhere(node_labels > 0)
            if edge_points.size > 0:
                distances = distance.cdist([(py, px)], edge_points)
                nearest_index = np.argmin(distances)
                nearest_point = edge_points[nearest_index]
                region = node_labels[nearest_point[0], nearest_point[1]]

                if region > 0:
                    if region not in region_to_node:
                        region_to_node[region] = current_node_id
                        current_node_id += 1

                    node_id = region_to_node[region]
                    if node_id not in connected_nodes:
                        connected_nodes.append(node_id)

        if connected_nodes:
            # Add 'c' suffix to nodes connected to bridge components
            if comp_id.startswith('Bridge_'):
                node_strings = [f"{node}c" for node in connected_nodes]
            else:
                node_strings = [str(node) for node in connected_nodes]
            netlist_lines.append(f"{comp_id} {' '.join(node_strings)}")

    # Merge nodes connected by bridge components and remove bridge entries
    netlist_lines = merge_bridge_connected_nodes(netlist_lines)

    return netlist_lines


def merge_bridge_connected_nodes(netlist_lines):
    """
    Merge nodes that are connected by bridge components.
    Bridge components indicate that two nodes should actually be the same node.
    """
    # Find bridge components and the nodes they connect
    bridge_connections = {}  # {bridge_id: [node1, node2]}
    node_merge_map = {}  # {old_node: new_node}

    # Extract bridge connections
    for line in netlist_lines:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].startswith('Bridge_'):
            bridge_id = parts[0]
            # Extract nodes (remove 'c' suffix)
            nodes = []
            for node_str in parts[1:]:
                if node_str.endswith('c'):
                    node_num = int(node_str[:-1])  # Remove 'c' suffix
                    nodes.append(node_num)

            if len(nodes) >= 1:
                bridge_connections[bridge_id] = nodes

    # Collect all nodes that are connected by bridge components
    # All bridge nodes should be merged together (they represent the same cut wire)
    all_bridge_nodes = []
    for bridge_id, nodes in bridge_connections.items():
        all_bridge_nodes.extend(nodes)

    if len(all_bridge_nodes) >= 2:
        # Merge all bridge nodes to the lowest numbered node
        min_node = min(all_bridge_nodes)
        for node in all_bridge_nodes:
            if node != min_node:
                node_merge_map[node] = min_node
                print(f"Merged node {node} -> {min_node} (bridge connection)")

    # Apply merging to all netlist lines (except bridge components)
    merged_lines = []
    for line in netlist_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            comp_id = parts[0]

            # Skip bridge components in final netlist
            if comp_id.startswith('Bridge_'):
                continue

            # Merge nodes for regular components
            new_nodes = []
            for node_str in parts[1:]:
                node_num = int(node_str)
                # Apply merge mapping
                merged_node = node_merge_map.get(node_num, node_num)
                new_nodes.append(str(merged_node))

            merged_lines.append(f"{comp_id} {' '.join(new_nodes)}")

    # Final cleanup: renumber nodes to fill gaps in sequence
    merged_lines = renumber_nodes_sequentially(merged_lines)

    return merged_lines


def renumber_nodes_sequentially(netlist_lines):
    """
    Renumber nodes to eliminate gaps in the numbering sequence.
    For example: if nodes 1,2,3,4,6 are used, replace all 6 with 5.
    """
    # Extract all unique node numbers from the netlist
    used_nodes = set()
    for line in netlist_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            for node_str in parts[1:]:
                try:
                    node_num = int(node_str)
                    used_nodes.add(node_num)
                except ValueError:
                    continue  # Skip non-numeric values

    if not used_nodes:
        return netlist_lines

    # Find gaps in the sequence and create renumbering map
    max_node = max(used_nodes)
    renumber_map = {}

    # Create sequential numbering starting from 1
    sorted_nodes = sorted(used_nodes)
    for i, node in enumerate(sorted_nodes):
        new_node = i + 1
        if node != new_node:
            renumber_map[node] = new_node

    # Apply renumbering if needed
    if renumber_map:
        print(f"Renumbering nodes: {renumber_map}")
        renumbered_lines = []
        for line in netlist_lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                comp_id = parts[0]
                new_nodes = []
                for node_str in parts[1:]:
                    try:
                        node_num = int(node_str)
                        new_node = renumber_map.get(node_num, node_num)
                        new_nodes.append(str(new_node))
                    except ValueError:
                        new_nodes.append(node_str)  # Keep non-numeric as is

                renumbered_lines.append(f"{comp_id} {' '.join(new_nodes)}")
            else:
                renumbered_lines.append(line)

        return renumbered_lines

    return netlist_lines


def clean_crossing_netlist(crossing_netlist_path):
    """
    Clean the crossing disambiguation netlist for use as raw netlist structure.
    Removes comments, GND components, and empty lines.
    """
    try:
        with open(crossing_netlist_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip comment lines, GND components, and empty lines
            if (line and
                not line.startswith('#') and
                not line.startswith('GND_')):
                cleaned_lines.append(line)

        print(f"Cleaned crossing netlist: {len(cleaned_lines)} lines")
        return cleaned_lines

    except Exception as e:
        print(f"Error reading crossing netlist: {e}")
        return []




