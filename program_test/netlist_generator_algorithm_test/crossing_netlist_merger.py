#!/usr/bin/env python3
"""
Crossing Netlist Merger

This module merges the two disambiguation netlists (line1_connected and line2_connected)
into a final netlist that properly handles wire crossings using bridge components as merge points.
"""

import re
import os
from collections import defaultdict


class NetlistMerger:
    """
    Merges disambiguation netlists using bridge components as connection markers.
    """

    def __init__(self):
        self.components = {}  # component_id -> nodes
        self.bridge_mappings = {}  # bridge_id -> node_id for each netlist

    def parse_netlist_line(self, line):
        """
        Parse a single netlist line into component_id and connected nodes.

        Args:
            line (str): Netlist line like "Resistor_1 3 4"

        Returns:
            tuple: (component_id, nodes_list) or None if invalid line
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return None

        parts = line.split()
        if len(parts) < 2:
            return None

        component_id = parts[0]
        nodes = [int(node) for node in parts[1:]]
        return component_id, nodes

    def load_disambiguation_netlist(self, netlist_path, context):
        """
        Load and parse a disambiguation netlist file.

        Args:
            netlist_path (str): Path to the netlist file
            context (str): "line1_connected" or "line2_connected"

        Returns:
            dict: component_id -> nodes mapping
        """
        components = {}
        bridges = {}

        if not os.path.exists(netlist_path):
            print(f"Warning: Netlist file not found: {netlist_path}")
            return components, bridges

        with open(netlist_path, 'r') as f:
            for line in f:
                parsed = self.parse_netlist_line(line)
                if parsed:
                    component_id, nodes = parsed
                    components[component_id] = nodes

                    # Track bridge components separately
                    if component_id.startswith('Bridge_'):
                        bridges[component_id] = nodes

        print(f"Loaded {len(components)} components from {context} netlist")
        print(f"Found {len(bridges)} bridge components")

        return components, bridges

    def identify_bridge_pairs(self, bridges1, bridges2):
        """
        Identify bridge pairs that represent the same wire across both netlists.

        Args:
            bridges1 (dict): Bridge components from line1_connected netlist
            bridges2 (dict): Bridge components from line2_connected netlist

        Returns:
            dict: mapping of bridge pairs and their connection info
        """
        bridge_pairs = {}

        # Extract crossing and line information from bridge IDs
        # Format: Bridge_{crossing_id}_{line_name}_{endpoint}
        pattern = r'Bridge_(\d+)_(line\d+)_(start|end)'

        bridges1_parsed = {}
        bridges2_parsed = {}

        # Parse bridge IDs from both netlists
        for bridge_id, nodes in bridges1.items():
            match = re.match(pattern, bridge_id)
            if match:
                crossing_id, line_name, endpoint = match.groups()
                key = (crossing_id, line_name)
                if key not in bridges1_parsed:
                    bridges1_parsed[key] = {}
                bridges1_parsed[key][endpoint] = {'bridge_id': bridge_id, 'nodes': nodes}

        for bridge_id, nodes in bridges2.items():
            match = re.match(pattern, bridge_id)
            if match:
                crossing_id, line_name, endpoint = match.groups()
                key = (crossing_id, line_name)
                if key not in bridges2_parsed:
                    bridges2_parsed[key] = {}
                bridges2_parsed[key][endpoint] = {'bridge_id': bridge_id, 'nodes': nodes}

        # Identify wire pairs that need to be connected across netlists
        for crossing_id in set(k[0] for k in bridges1_parsed.keys()) | set(k[0] for k in bridges2_parsed.keys()):
            # For each crossing, we should have:
            # - line1 bridges in line2_connected netlist (line1 was cut)
            # - line2 bridges in line1_connected netlist (line2 was cut)

            line1_key = (crossing_id, 'line1')
            line2_key = (crossing_id, 'line2')

            line1_bridges = bridges2_parsed.get(line1_key, {})  # line1 bridges in line2_connected
            line2_bridges = bridges1_parsed.get(line2_key, {})  # line2 bridges in line1_connected

            if line1_bridges and line2_bridges:
                bridge_pairs[f'crossing_{crossing_id}'] = {
                    'line1_bridges': line1_bridges,
                    'line2_bridges': line2_bridges
                }

        return bridge_pairs

    def merge_netlists(self, netlist1_path, netlist2_path):
        """
        Merge two disambiguation netlists into a final netlist.

        Args:
            netlist1_path (str): Path to line1_connected netlist
            netlist2_path (str): Path to line2_connected netlist

        Returns:
            list: Merged netlist lines
        """
        # Load both netlists
        components1, bridges1 = self.load_disambiguation_netlist(netlist1_path, "line1_connected")
        components2, bridges2 = self.load_disambiguation_netlist(netlist2_path, "line2_connected")

        # Identify bridge pairs for wire connections
        bridge_pairs = self.identify_bridge_pairs(bridges1, bridges2)

        print(f"Found {len(bridge_pairs)} crossing wire pairs to merge")

        # Start with components from line1_connected (arbitrary choice)
        merged_components = {}
        for comp_id, nodes in components1.items():
            if not comp_id.startswith('Bridge_'):
                merged_components[comp_id] = nodes.copy()

        # Create node mapping for merging across netlists
        # We'll need to map nodes from netlist2 to equivalent nodes in the merged result
        node_mapping = self.create_node_mapping(components1, components2, bridge_pairs)

        # Add non-bridge components from line2_connected, applying node mapping
        for comp_id, nodes in components2.items():
            if not comp_id.startswith('Bridge_') and comp_id not in merged_components:
                mapped_nodes = [node_mapping.get(node, node) for node in nodes]
                merged_components[comp_id] = mapped_nodes

        # Apply bridge connections to merge wires across crossings
        for crossing_name, pair_info in bridge_pairs.items():
            line1_bridges = pair_info['line1_bridges']
            line2_bridges = pair_info['line2_bridges']

            # Connect the nodes that bridge endpoints point to
            if 'start' in line1_bridges and 'end' in line1_bridges:
                if 'start' in line2_bridges and 'end' in line2_bridges:
                    self.apply_bridge_connections(merged_components, line1_bridges, line2_bridges, node_mapping)

        # Convert to netlist format
        netlist_lines = []
        for comp_id, nodes in merged_components.items():
            nodes_str = ' '.join(map(str, nodes))
            netlist_lines.append(f"{comp_id} {nodes_str}")

        return netlist_lines

    def create_node_mapping(self, components1, components2, bridge_pairs):
        """
        Create mapping between node IDs in both netlists based on bridge connections.

        Args:
            components1 (dict): Components from line1_connected netlist
            components2 (dict): Components from line2_connected netlist
            bridge_pairs (dict): Bridge pair information

        Returns:
            dict: node_id mapping from netlist2 to merged netlist
        """
        node_mapping = {}
        next_available_node = max(
            max(max(nodes) for nodes in components1.values() if nodes),
            max(max(nodes) for nodes in components2.values() if nodes)
        ) + 1

        # For now, return identity mapping - more sophisticated logic can be added
        # to handle node conflicts and bridge-based connections
        return node_mapping

    def apply_bridge_connections(self, merged_components, line1_bridges, line2_bridges, node_mapping):
        """
        Apply wire connections across the crossing using bridge information.

        Args:
            merged_components (dict): The merged component dictionary to modify
            line1_bridges (dict): Bridge information for line1
            line2_bridges (dict): Bridge information for line2
            node_mapping (dict): Node mapping from netlist2 to merged netlist
        """
        print(f"Connecting line1 bridges: {line1_bridges}")
        print(f"Connecting line2 bridges: {line2_bridges}")

        # Extract connection points
        if 'start' in line1_bridges and 'end' in line1_bridges:
            line1_start_node = line1_bridges['start']['nodes'][0]
            line1_end_node = line1_bridges['end']['nodes'][0]

        if 'start' in line2_bridges and 'end' in line2_bridges:
            line2_start_node = line2_bridges['start']['nodes'][0]
            line2_end_node = line2_bridges['end']['nodes'][0]

        # Create virtual wire components to represent the crossing connections
        # Line1 connection: connects line1_start_node to line1_end_node
        # Line2 connection: connects line2_start_node to line2_end_node

        wire_component_1 = f"CrossingWire_1"
        wire_component_2 = f"CrossingWire_2"

        merged_components[wire_component_1] = [line1_start_node, line1_end_node]
        merged_components[wire_component_2] = [line2_start_node, line2_end_node]

        print(f"Added crossing wire connections:")
        print(f"  {wire_component_1}: {line1_start_node} <-> {line1_end_node}")
        print(f"  {wire_component_2}: {line2_start_node} <-> {line2_end_node}")


def merge_crossing_netlists(output_folder, image_basename):
    """
    Merge disambiguation netlists for an image with crossings.

    Args:
        output_folder (str): Output debugging folder path
        image_basename (str): Base name of the image (e.g., "131")

    Returns:
        str: Path to the merged netlist file
    """
    disambiguation_folder = os.path.join(output_folder, image_basename, "crossing_disambiguation")

    # Find the disambiguation netlist files
    line1_netlist = None
    line2_netlist = None

    for filename in os.listdir(disambiguation_folder):
        if filename.endswith('_line1_connected_netlist.txt'):
            line1_netlist = os.path.join(disambiguation_folder, filename)
        elif filename.endswith('_line2_connected_netlist.txt'):
            line2_netlist = os.path.join(disambiguation_folder, filename)

    if not line1_netlist or not line2_netlist:
        print(f"Could not find both disambiguation netlists in {disambiguation_folder}")
        return None

    print(f"\nMERGING CROSSING DISAMBIGUATION NETLISTS")
    print(f"Line1 connected: {line1_netlist}")
    print(f"Line2 connected: {line2_netlist}")

    # Create merger and merge the netlists
    merger = NetlistMerger()
    merged_lines = merger.merge_netlists(line1_netlist, line2_netlist)

    # Save the merged netlist
    merged_netlist_path = os.path.join(disambiguation_folder, f"{image_basename}_merged_netlist.txt")
    with open(merged_netlist_path, 'w') as f:
        f.write(f"# Merged netlist for {image_basename} with resolved crossings\n")
        f.write(f"# Merged from {os.path.basename(line1_netlist)} and {os.path.basename(line2_netlist)}\n\n")
        for line in merged_lines:
            f.write(line + "\n")

    print(f"Merged netlist saved: {merged_netlist_path}")

    # Print the merged netlist
    print(f"\nMERGED NETLIST:")
    print("=" * 60)
    for line in merged_lines:
        print(line)

    return merged_netlist_path


if __name__ == "__main__":
    # Test the merger with existing disambiguation results
    test_output_folder = "outputs/debugging"
    test_image_basename = "131"
    merge_crossing_netlists(test_output_folder, test_image_basename)