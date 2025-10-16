import numpy as np
import matplotlib
matplotlib.use('Agg') 

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

