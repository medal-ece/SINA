
import cv2 
import numpy as np
from utils.recognizer import detect
from utils.node_detector import node_detector
from utils.mapping import mapping, mid_point
from utils.text_removal import TextRemover
from hough.htlcnn.demo import predict_lines
from utils.line_merger import cluster_grouping,radial_grouping_new,apply_linelabels
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from PIL import Image
import os
import sys
from utils.yolov8 import comp_detection


def save_image_from_array(array: np.ndarray, image_path: str):
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(array)
    
    # Save the image to the specified path
    image.save(image_path)


def main(image_output_dir,img_original,img_name,use_clustering=False):
    """main function where all algorithms are called

    Args:
        img (numpy array): input image

    Returns:
        result (numpy array): final rebuilt image
        boxes1 (numpy array): bounding boxes on given input image
        main_img1 (numpy array): nodes and terminals on given input image
        comp_list (List): list of all the components detected 
        jns_list (List): list of all the junctions detected 
        conn_list (List): list of connections traced 
    """

    sample_dict = {}
    

    ann_img,dim_matrix = comp_detection(os.path.join(image_output_dir, 'scanned_circuit.png'))
    h, w = ann_img.shape[:2]
    img_original = cv2.resize(img_original, (w, h))
   
    # converting images to grayscale
    img_og = cv2.cvtColor(img_original,cv2.COLOR_RGB2BGR)
    main_img_og = np.copy(img_og)
    gray_og = cv2.cvtColor(img_og,cv2.COLOR_BGR2GRAY)

    #### Removing components from original image
    comp_removed = np.copy(img_og)
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    names = ['AC_Source', 'BJT', 'Battery', 'Capacitor', 'Current_Source', 'DC_Source', 'Diode', 'Ground', 'Inductor', 'MOSFET', 'Resistor', 'Voltage_Source']
    color_map = {
    'AC_Source': (0, 0, 255),         # Red
    'BJT': (0, 255, 0),               # Green
    'Battery': (255, 0, 0),           # Blue
    'Capacitor': (255, 165, 0),       # Orange
    'Current_Source': (255, 255, 0),  # Yellow
    'DC_Source': (0, 255, 255),       # Cyan
    'Diode': (255, 0, 255),           # Magenta
    'Ground': (128, 0, 128),          # Purple
    'Inductor': (0, 128, 128),        # Teal
    'MOSFET': (128, 128, 0),          # Olive
    'Resistor': (0, 128, 0),          # Dark Green
    'Voltage_Source': (128, 0, 0),    # Maroon
    }
    default_color = (255, 255, 255)
    # making a copy of the original image and text removed, plotting bboxes on image and removing detected bounding boxes
    boxes = np.zeros_like(gray_og)
    boxes1 = np.zeros_like(main_img_og)
    main_img1 = cv2.cvtColor(main_img_og,cv2.COLOR_BGR2RGB)
    ratio = 0.85
    component_dict = {}
    for i in range(dim_matrix.shape[0]):
        dim = dim_matrix[i]
        start = (int(dim[0]), int(dim[1]))
        end = (int(dim[2]), int(dim[3]))
        
        # Calculate width and height
        width = end[0] - start[0]
        height = end[1] - start[1]
        
        # Calculate reduction
        reduction_w = int(width * (1 - ratio) / 2)
        reduction_h = int(height * (1 - ratio) / 2)
        
        # Adjust start and end points
        start = (start[0] + reduction_w, start[1] + reduction_h)
        end = (end[0] - reduction_w, end[1] - reduction_h)
        
        # Get the label and determine the color
        label = names[dim[5]]
        if(label in component_dict):
            component_dict[label]+=1
        else:
            component_dict[label] = 1
        color = color_map.get(label, default_color)  # Use the color mapping or default color if label not found
        
        # Draw the bounding boxes with the specified color
        boxes = cv2.rectangle(boxes, start, end, 255, 1) 
        boxes1 = cv2.rectangle(main_img1, start, end, color, 2)
        
        # Optionally draw the label text on the image
        text = f'{label}'
        # boxes1 = cv2.putText(main_img1, text, (start[0] - 30, start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        comp_removed[int(round(dim[1])):int(round(dim[3])), int(round(dim[0])):int(round(dim[2]))] = 255
    sample_dict['Component Description'] = component_dict
    
    # save comp_removed image
    save_path = os.path.join(image_output_dir, 'component_removed_circuit.png')
    save_image_from_array(comp_removed,save_path)

    # remove text from component removed image
    remover = TextRemover()
    img_text_removed = remover.remove_text(save_path)
    img_text_removed = cv2.cvtColor(img_text_removed, cv2.COLOR_RGB2BGR)
    img_text_removed = cv2.cvtColor(img_text_removed,cv2.COLOR_BGR2GRAY)
    save_path = os.path.join(image_output_dir, 'text_and_comp_removed_circuit.png')
    save_image_from_array(img_text_removed,save_path)
    
    ##### Detecting line segments using Deep Hough Transformer and grouping line segments using either radial/clustering mechanism
    line_marked_img, processed_lines = predict_lines(config_file="./hough/htlcnn/config/wireframe.yaml", checkpoint_path="./hough/checkpoint.pth", image=img_text_removed, devices="0", threshold=0.99)

    sample_dict['No. of Nets'] = len(processed_lines)
    
    if(use_clustering):
        clustered_img,cluster_mids = cluster_grouping(img_text_removed,processed_lines,comp_removed.shape)
    else:
        clustered_img,annotation_labels = radial_grouping_new(img_text_removed,processed_lines,comp_removed.shape)

    # Apply line annotation labels on original image and component detected image
    og_image_with_line_labels = apply_linelabels(gray_og,annotation_labels)
    comdetected_image_with_line_labels = apply_linelabels(boxes1,annotation_labels)
    
    ##### Detecting nodes on the components removed image
    
    nodes = node_detector(img_text_removed)

    sample_dict['No. of Nodes'] = len(nodes)

    ##### Detecting terminals using bboxes image and thresholded input image
    img = cv2.GaussianBlur(gray_og,(9,9),0)

    
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    boxes = boxes == 255
    th = th == 255
    comp_pos1 = np.logical_not(np.logical_not(boxes)+th)
    comp_pos1 = comp_pos1.astype(np.uint8)
    comp_pos = comp_pos1*255
    comp_dim_tmp = []
    ##### Using contour detection to find the exact centers of the terminals
    contours, hierarchy = cv2.findContours(comp_pos,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    for i,cntr in enumerate(contours):
        M = cv2.moments(cntr)
        length = cv2.arcLength(cntr,True)
        (cx,cy), r = cv2.minEnclosingCircle(cntr)
        comp_dim_tmp.append([cy,cx,length])
    h = len(comp_dim_tmp) - 2*dim_matrix.shape[0]
    comp_dim_tmp = sorted(comp_dim_tmp, key = lambda x: x[2])#[h:]    
    comp_dim = []
    for dim in comp_dim_tmp:
        comp_dim.append([dim[0],dim[1]])
    nodes = np.array(nodes)
    comp_dim = np.array(comp_dim)

    
    ##### Drawing terminals and nodes on the input image, to verify
    main_img1 = cv2.cvtColor(main_img_og,cv2.COLOR_BGR2RGB)
    for y,x in comp_dim:
        cv2.circle(main_img1,(int(x),int(y)), 1, (0,255,0), 5)
    for x,y in nodes:
        cv2.circle(main_img1,(int(y),int(x)), 1, (250,0,0), 5)

    #### Mapping nodes, terminals and components
    
    maps,node_comp_map,node_node_map = mapping(dim_matrix,comp_dim,nodes)

    
    #### Generating .txt file with components and connections
    f = open(os.path.join(image_output_dir,"output.txt"), "w")

    result = cv2.cvtColor(main_img_og,cv2.COLOR_BGR2RGB)

    # writes all the components present and draws them on a plain image
    f.write("Components in the circuit are: \n")
    count_ind = [0]*len(classes)
    comp_list = []
    for i in range(dim_matrix.shape[0]):
        cl = int(dim_matrix[i][5])
        dim = dim_matrix[i]
        start = (int(dim[0]),int(dim[1]))
        end = (int(dim[2]),int(dim[3]))
        cv2.rectangle(result, start, end, (255,0,0), 2)
        midx,midy = mid_point(dim[1],dim[0],dim[3],dim[2])
        cv2.putText(result,classes[cl]+str(count_ind[cl]+1), (int(midy),int(midx)),cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 0), 1, cv2.LINE_AA)
        f.write(names[cl]+" "+classes[cl]+str(count_ind[cl]+1)+"\n")
        comp_list.append(names[cl]+" "+classes[cl]+str(count_ind[cl]+1))
        count_ind[cl] = count_ind[cl] + 1
    
    # writes all the nodes/junctions present and draws them on a plain image
    f.write("Junctions in the circuit are: \n")
    jns_list = []
    for i in range(nodes.shape[0]):
        f.write("Node N"+str(i+1)+"\n")
        jns_list.append("Junction N"+str(i+1))
        x,y= nodes[i]
        cv2.circle(result,(int(y),int(x)), 1, (0,0,255), 6)
        cv2.putText(result, str(i), (int(y),int(x)),cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 0), 1, cv2.LINE_AA) 
    
    # writes all the connections present and draws them on a plain image
    f.write("Connections in the circuit are: \n")
    conn_list = []
    count_ind = [0]*len(classes)
    for i,_ in enumerate(maps):
        cl = int(dim_matrix[i][5])
        n1 =  node_comp_map[2*i]
        n2 = node_comp_map[2*i+1]
        start1 = (int(round(nodes[n1][1])),int(round(nodes[n1][0])))
        end2 = (int(round(nodes[n2][1])),int(round(nodes[n2][0])))
        end1 = (int(round(comp_dim[int(maps[i][0])][1])),int(round(comp_dim[int(maps[i][0])][0])))
        start2 = (int(round(comp_dim[int(maps[i][1])][1])),int(round(comp_dim[int(maps[i][1])][0])))
        f.write(classes[cl]+str(count_ind[cl]+1)+" is between Node"+str(n1)+" and Node"+str(n2)+"\n")
        conn_list.append(classes[cl]+str(count_ind[cl]+1)+" is between Node"+str(n1)+" and Node"+str(n2))
        count_ind[cl] = count_ind[cl] + 1
    
    count_node_ind = [0]*len(node_node_map)
    for i,_ in enumerate(node_node_map):
        n1 = node_node_map[i][0]
        n2 = node_node_map[i][1]
        count = 0
        for j,_ in enumerate(node_node_map):
            if j != i:
                n11 = node_node_map[j][0]
                n21 = node_node_map[j][1]
                if n1 == n21 and n2 == n11:
                    count = count+1+count_node_ind[j]
        count_node_ind[i] = count
    for i ,_ in enumerate(node_node_map):
        n1 = node_node_map[i][0]
        n2 = node_node_map[i][1]             
        if count_node_ind[i] < 2:
            f.write("Node"+str(n1)+" and "+"Node"+str(n2)+" are connected"+"\n")
            conn_list.append("Node"+str(n1)+" and "+"Node"+str(n2)+" are connected")
            start = (int(round(nodes[n1][1])),int(round(nodes[n1][0])))
            end = (int(round(nodes[n2][1])),int(round(nodes[n2][0])))
    f.close()

    return result, boxes1, main_img1,comp_removed, og_image_with_line_labels,comdetected_image_with_line_labels,comp_list, jns_list, conn_list,sample_dict
