import os
from xml.dom import minidom

# Directory to save the output files
out_dir = './out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Parse the XML file
file = minidom.parse('annotations.xml')

# Fixed label mapping for both methods
label_mapping = {"Rectangle": 0}  # Support both
label_counter = len(label_mapping) - 1

# Processing each image
images = file.getElementsByTagName('image')

for image in images:
    width = int(image.getAttribute('width'))
    height = int(image.getAttribute('height'))
    name = image.getAttribute('name')
    points_elements = image.getElementsByTagName('points')
    bboxes = image.getElementsByTagName('box')

    # Create a new label file for each image
    label_file_path = os.path.join(out_dir, name[:-4] + '.txt')
    with open(label_file_path, 'w') as label_file:

        # Handle bounding box method (your method)
        for bbox in bboxes:
            bbox_label = bbox.getAttribute("label")
            bbox_group = bbox.getAttribute("group_id")
            if bbox_group == "":
                bbox_group = None

            if bbox_label not in label_mapping:
                label_mapping[bbox_label] = label_counter
                label_counter += 1

            label_id = label_mapping[bbox_label]

            xtl = float(bbox.getAttribute('xtl'))
            ytl = float(bbox.getAttribute('ytl'))
            xbr = float(bbox.getAttribute('xbr'))
            ybr = float(bbox.getAttribute('ybr'))
            w = xbr - xtl
            h = ybr - ytl

            bbox_center_x = (xtl + (w / 2)) / width
            bbox_center_y = (ytl + (h / 2)) / height
            bbox_norm_w = w / width
            bbox_norm_h = h / height

            label_file.write(f"{label_id} {bbox_center_x} {bbox_center_y} {bbox_norm_w} {bbox_norm_h} ")

            matched_points = []
            for points in points_elements:
                points_label = points.getAttribute("label")
                points_group = points.getAttribute("group_id")

                if points_label == "Pin" and points_group == bbox_group:
                    points_data = points.getAttribute('points')
                    points_list = points_data.split(';')

                    for point in points_list:
                        p1, p2 = map(float, point.split(','))
                        norm_p1 = p1 / width
                        norm_p2 = p2 / height
                        matched_points.append((norm_p1, norm_p2, 1))

            while len(matched_points) < 5:
                matched_points.append((0.0, 0.0, 0))  

            for i, (norm_p1, norm_p2, visibility) in enumerate(matched_points[:5]):
                label_file.write(f"{norm_p1} {norm_p2} {visibility}")
                if i < len(matched_points[:5]) - 1:
                    label_file.write(" ")
                else:
                    label_file.write("\n")  

        # Handle keypoints forming a square (partner's method)
        for points in points_elements:
            points_label = points.getAttribute("label")

            if points_label == "Keypoint_Square":  # Identifies a shape purely using keypoints
                keypoint_data = points.getAttribute('points')
                keypoint_list = keypoint_data.split(';')

                normalized_keypoints = []
                for kp in keypoint_list:
                    kp_x, kp_y = map(float, kp.split(','))
                    norm_x = kp_x / width
                    norm_y = kp_y / height
                    normalized_keypoints.append((norm_x, norm_y, 1))  

                label_id = label_mapping["Keypoint_Square"]
                label_file.write(f"{label_id} ")

                while len(normalized_keypoints) < 5:
                    normalized_keypoints.append((0.0, 0.0, 0))  

                for i, (norm_x, norm_y, visibility) in enumerate(normalized_keypoints[:5]):
                    label_file.write(f"{norm_x} {norm_y} {visibility}")
                    if i < len(normalized_keypoints[:5]) - 1:
                        label_file.write(" ")
                    else:
                        label_file.write("\n")  

print("Output files created successfully in:", out_dir)
