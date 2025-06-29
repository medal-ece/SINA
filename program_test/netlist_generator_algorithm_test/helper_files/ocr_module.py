# import cv2
# import os
# from helper_files.proximity_analysis import assign_text_to_nearest_component, get_component_center
# import re
# import easyocr
# import pytesseract
# import matplotlib
# matplotlib.use('Agg') 

# # ===== OCR CLASS DEFINITION =====
# class CircuitOCR:
#     def __init__(self, use_gpu=True):
#         """Initialize OCR engines"""
#         print("Initializing OCR engines...")
        
#         # Initialize EasyOCR (this works with your current setup)
#         self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
         
#         # No filtering - capture all text
#         self.component_patterns = []  # Empty - we'll capture everything
#         self.combined_pattern = None  # No pattern matching
        
#         # Common OCR misreads
#         self.ocr_corrections = {
#             'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8',
#             'G': '6', 'l': '1', 'o': '0', 'i': '1', 's': '5'
#         }

#     def advanced_preprocessing(self, image):
#         """Create multiple preprocessing versions optimized for OCR"""
#         versions = {}
        
#         # Original
#         versions['original'] = image.copy()
        
#         # Grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         versions['gray'] = gray
        
#         # High contrast binary (Otsu's method)
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         versions['binary_otsu'] = binary
        
#         # Adaptive threshold (good for varying lighting)
#         adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY, 21, 10)
#         versions['adaptive'] = adaptive
        
#         # Inverted versions (for white text on dark background)
#         versions['inverted_gray'] = cv2.bitwise_not(gray)
#         versions['inverted_binary'] = cv2.bitwise_not(binary)
#         versions['inverted_adaptive'] = cv2.bitwise_not(adaptive)
        
#         # Enhanced contrast using CLAHE
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
#         versions['enhanced_contrast'] = enhanced
        
#         # Denoised version
#         denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#         versions['denoised'] = denoised
        
#         # Edge-preserved smoothing
#         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
#         versions['bilateral'] = bilateral
        
#         # Morphological operations to clean text
#         kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
#         versions['morph_close'] = morph_close
        
#         # Dilated version (makes text thicker)
#         kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#         dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
#         versions['dilated'] = dilated
        
#         # Super resolution attempt (resize)
#         height, width = gray.shape
#         if width < 2000:
#             scale = 2000 / width
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#             versions['upscaled'] = resized
        
#         return versions

#     def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7):
#         """Extract text using EasyOCR with confidence scores and bounding boxes"""
#         try:
#             # Run EasyOCR with specific settings
#             results = self.easy_reader.readtext(
#                 image, 
#                 detail=detail,
#                 width_ths=width_ths,
#                 height_ths=height_ths,
#                 paragraph=False,
#                 text_threshold=0.3,
#                 low_text=0.3,
#                 mag_ratio=1.5
#             )
            
#             if detail == 1:
#                 # Return tuples of (text, confidence, bbox)
#                 text_with_conf_bbox = []
#                 for result in results:
#                     if result[2] > 0.2:  # confidence threshold
#                         bbox_points = result[0]
#                         # Calculate center point of bbox
#                         x_coords = [p[0] for p in bbox_points]
#                         y_coords = [p[1] for p in bbox_points]
#                         center_x = sum(x_coords) / len(x_coords)
#                         center_y = sum(y_coords) / len(y_coords)
                        
#                         text_with_conf_bbox.append({
#                             'text': result[1].upper(),
#                             'confidence': result[2],
#                             'bbox': bbox_points,
#                             'center': (center_x, center_y)
#                         })
#             else:
#                 # When detail=0, no bbox available
#                 text_with_conf_bbox = [{'text': text.upper(), 'confidence': 0.0, 'bbox': None, 'center': None} 
#                                       for text in results]
            
#             return text_with_conf_bbox
#         except Exception as e:
#             print(f"    EasyOCR error: {e}")
#             return []

#     def extract_text_with_tesseract(self, image):
#         """Extract text using Tesseract with confidence scores"""
#         text_with_conf = []
        
#         # Try getting individual words with confidence scores
#         try:
#             data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#             for i, word in enumerate(data['text']):
#                 if word.strip():
#                     confidence = float(data['conf'][i])
#                     if confidence > 0:
#                         text_with_conf.append({
#                             'text': word.upper(),
#                             'confidence': confidence / 100.0,
#                             'bbox': None,
#                             'center': None
#                         })
#         except:
#             pass
        
#         # Other configurations without confidence
#         configs = [
#             r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             r'--psm 8',
#             r'--psm 12',
#             r'--psm 13',
#         ]
        
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(image, config=config)
#                 if text.strip():
#                     words = text.upper().split()
#                     for word in words:
#                         if word.strip():
#                             text_with_conf.append({
#                                 'text': word,
#                                 'confidence': 0.5,
#                                 'bbox': None,
#                                 'center': None
#                             })
#             except:
#                 pass
            
#         return text_with_conf


#     def correct_ocr_errors(self, text):
#         """Correct common OCR misreads"""
#         corrected = text
#         for wrong, right in self.ocr_corrections.items():
#             corrected = corrected.replace(wrong, right)
#         return corrected

#     def extract_components_from_texts(self, text_with_conf_bbox_list):
#         """Extract ALL text with confidence scores and locations (no filtering)"""
#         # Combine all results
#         all_items = {}  # Use dict to track best result for each text
        
#         for item in text_with_conf_bbox_list:
#             text = item.get('text', '')
#             conf = item.get('confidence', 0.0)
#             bbox = item.get('bbox', None)
#             center = item.get('center', None)
            
#             # Split by common separators and clean
#             items = re.split(r'[\s,;:\n\r\t]+', text)
#             for sub_item in items:
#                 sub_item = sub_item.strip()
#                 # Apply OCR corrections
#                 corrected = self.correct_ocr_errors(sub_item)
#                 if corrected and len(corrected) > 0:
#                     # Keep the highest confidence version with location info
#                     if corrected not in all_items or conf > all_items[corrected]['confidence']:
#                         all_items[corrected] = {
#                             'text': corrected,
#                             'confidence': conf,
#                             'bbox': bbox,
#                             'center': center
#                         }
        
#         # Convert to list sorted by confidence
#         return sorted(all_items.values(), key=lambda x: x['confidence'], reverse=True)

#     def process_image(self, image_path):
#         """Main processing function that returns text with confidence scores"""
#         # Read image
#         image = cv2.imread(image_path)
        
#         # Get all preprocessing versions
#         versions = self.advanced_preprocessing(image)
        
#         all_text_with_conf = []  # List of (text, confidence) tuples
        
#         print("  Processing with multiple OCR engines and preprocessing...")
        
#         # Process each version
#         for version_name, processed_img in versions.items():
#             # Convert grayscale to BGR for EasyOCR if needed
#             if len(processed_img.shape) == 2:
#                 img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 img_for_ocr = processed_img
            
#             # EasyOCR
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr)
#             all_text_with_conf.extend(text_conf_list)
            
#             # EasyOCR with different parameters
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
#             all_text_with_conf.extend(text_conf_list)
            
#             # Tesseract
#             text_conf_list = self.extract_text_with_tesseract(processed_img)
#             all_text_with_conf.extend(text_conf_list)
             
#         # Try cropping and processing regions
#         height, width = image.shape[:2]
#         quadrants = [
#             (0, 0, width//2, height//2),
#             (width//2, 0, width, height//2),
#             (0, height//2, width//2, height),
#             (width//2, height//2, width, height)
#         ]
        
#         for x1, y1, x2, y2 in quadrants:
#             cropped = image[y1:y2, x1:x2]
#             text_conf_list = self.extract_text_with_easyocr(cropped)
#             all_text_with_conf.extend(text_conf_list)
        
#         # Process and deduplicate
#         final_results = self.extract_components_from_texts(all_text_with_conf)
        
#         return final_results, versions

#     def visualize_results(self, image, all_text, output_path):
#         """Create visualization of detected text"""
#         try:
#             # Set matplotlib to use non-interactive backend
#             import matplotlib
#             matplotlib.use('Agg')  # Use non-GUI backend
#             import matplotlib.pyplot as plt
#             import matplotlib.patches as patches
            
#             # Run EasyOCR to get bounding boxes
#             results = self.easy_reader.readtext(image, detail=1)
            
#             fig, ax = plt.subplots(1, figsize=(15, 10))
#             ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
#             # Draw bounding boxes for all detected text
#             for i, (bbox, text, prob) in enumerate(results):
#                 if prob > 0.2:  # Show all text with reasonable confidence
#                     # Draw bounding box
#                     points = bbox
#                     rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=2)
#                     ax.add_patch(rect)
                    
#                     # Add label
#                     ax.text(points[0][0], points[0][1]-5, text, 
#                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
#                            fontsize=8)
            
#             ax.set_title(f'Circuit OCR Results - Found {len(all_text)} text items', 
#                         fontsize=16, fontweight='bold')
#             ax.axis('off')
#             plt.tight_layout()
#             plt.savefig(output_path, dpi=200, bbox_inches='tight')
#             plt.close()
#         except Exception as e:
#             print(f"    Warning: Could not create visualization: {e}")
#             # Create a simple text visualization instead
#             with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
#                 f.write(f"Found {len(all_text)} text items:\n")
#                 f.write('\n'.join(all_text))

# # ===== OCR PROCESSING FUNCTION =====
# def process_circuit_ocr(image_path, output_folder, ocr_engine, components=None):
#     """Process OCR for circuit image and extract ALL text with confidence scores and proximity to components"""
#     print("\n  Running OCR text extraction...")
    
#     # Create OCR subfolder
#     ocr_output_folder = os.path.join(output_folder, "ocr_results")
#     os.makedirs(ocr_output_folder, exist_ok=True)
    
#     # Read image
#     image = cv2.imread(image_path)
    
#     # Extract all text with confidence and location
#     text_with_info, versions = ocr_engine.process_image(image_path)
    
#     # Save some preprocessing versions for debugging
#     for i, (name, img) in enumerate(list(versions.items())[:4]):
#         cv2.imwrite(os.path.join(ocr_output_folder, f"ocr_preprocess_{name}.png"), img)
    
#     # Visualize (pass just the text for visualization)
#     all_text = [item['text'] for item in text_with_info]
#     if all_text:
#         ocr_engine.visualize_results(image, all_text, 
#                                    os.path.join(ocr_output_folder, "ocr_detected_text.png"))
    
#     # Save results with confidence scores
#     with open(os.path.join(ocr_output_folder, "extracted_text_with_confidence.txt"), 'w', encoding='utf-8') as f:
#         if text_with_info:
#             for item in text_with_info:
#                 f.write(f"{item['text']}\t{item['confidence']:.3f}\n")
#             print(f"    Found {len(text_with_info)} text items")
#             # Show top 10 with confidence
#             print("    Top OCR detections:")
#             for item in text_with_info[:10]:
#                 print(f"      {item['text']} (confidence: {item['confidence']:.3f})")
#             if len(text_with_info) > 10:
#                 print(f"      ... and {len(text_with_info) - 10} more")
#         else:
#             f.write("No text detected")
#             print("    No text detected")
    
#     # Save just the text (without confidence) for compatibility
#     with open(os.path.join(ocr_output_folder, "extracted_text.txt"), 'w', encoding='utf-8') as f:
#         if all_text:
#             f.write('\n'.join(all_text))
    
#     # If components are provided, create proximity-based mapping
#     if components and text_with_info:
#         print("    Analyzing text proximity to components...")
#         component_text_mapping = assign_text_to_nearest_component(text_with_info, components)
        
#         # Save proximity-based summary
#         with open(os.path.join(ocr_output_folder, "text_by_component_proximity.txt"), 'w', encoding='utf-8') as f:
#             f.write("OCR Text Assigned to Nearest Components\n")
#             f.write("="*60 + "\n")
#             f.write("Note: Each text appears only under its closest component\n")
#             f.write("="*60 + "\n\n")
            
#             for comp_id, text_items in component_text_mapping.items():
#                 # Find component info
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"Component: {comp_id} (Type: {comp_info['type']})\n")
#                     f.write(f"Bounding Box: {comp_info['bbox']}\n")
#                     f.write("-"*40 + "\n")
                    
#                     if text_items:
#                         f.write(f"Assigned Text ({len(text_items)} items, sorted by distance):\n")
#                         for i, item in enumerate(text_items):
#                             f.write(f"  {i+1}. {item['text']:<15} (conf: {item['confidence']:.3f}, dist: {item['distance']:.1f}px)\n")
#                     else:
#                         f.write("  No text assigned to this component\n")
#                     f.write("\n")
        
#         # Create visual proximity map showing assignments
#         proximity_vis = image.copy()
        
#         # Color map for components
#         colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
#         # Draw components and their assigned texts
#         for idx, (comp_id, text_items) in enumerate(component_text_mapping.items()):
#             comp_info = next((c for c in components if c['id'] == comp_id), None)
#             if comp_info:
#                 x_min, y_min, x_max, y_max = comp_info['bbox']
#                 color = colors[idx % len(colors)]
                
#                 # Draw component box
#                 cv2.rectangle(proximity_vis, (x_min, y_min), (x_max, y_max), color, 2)
#                 cv2.putText(proximity_vis, comp_id, (x_min, y_min - 5),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
#                 # Draw lines to all assigned texts
#                 comp_center = get_component_center(comp_info['bbox'])
#                 for text_item in text_items:
#                     if text_item['center']:
#                         # Draw line from component to text
#                         cv2.line(proximity_vis, 
#                                 (int(comp_center[0]), int(comp_center[1])), 
#                                 (int(text_item['center'][0]), int(text_item['center'][1])), 
#                                 color, 1)
#                         # Draw text with same color as component
#                         cv2.putText(proximity_vis, text_item['text'], 
#                                    (int(text_item['center'][0]), int(text_item['center'][1]) - 5),
#                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(os.path.join(ocr_output_folder, "text_component_assignments.png"), proximity_vis)
        
#         # Also create a summary statistics file
#         with open(os.path.join(ocr_output_folder, "assignment_statistics.txt"), 'w', encoding='utf-8') as f:
#             f.write("Text Assignment Statistics\n")
#             f.write("="*40 + "\n\n")
            
#             total_texts = sum(len(items) for items in component_text_mapping.values())
#             f.write(f"Total texts detected: {len(text_with_info)}\n")
#             f.write(f"Total texts with locations: {total_texts}\n")
#             f.write(f"Total components: {len(components)}\n\n")
            
#             f.write("Texts per component:\n")
#             for comp_id, text_items in component_text_mapping.items():
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"  {comp_id} ({comp_info['type']}): {len(text_items)} texts\n")
    
#     # Save detailed summary
#     with open(os.path.join(ocr_output_folder, "ocr_summary.txt"), 'w', encoding='utf-8') as f:
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total text items found: {len(text_with_info)}\n\n")
#         f.write("All extracted text with confidence scores:\n")
#         f.write("-" * 50 + "\n")
#         f.write("Text\t\t\tConfidence\n")
#         f.write("-" * 50 + "\n")
#         for item in text_with_info:
#             f.write(f"{item['text']:<20}\t{item['confidence']:.3f}\n")
    
#     return text_with_info



# import cv2
# import os
# from helper_files.proximity_analysis import assign_text_to_nearest_component, get_component_center
# import re
# import easyocr
# import pytesseract
# import matplotlib
# matplotlib.use('Agg') 
# import numpy as np

# # ===== OCR CLASS DEFINITION =====
# class CircuitOCR:
#     def __init__(self, use_gpu=True):
#         """Initialize OCR engines"""
#         print("Initializing OCR engines...")
        
#         # Initialize EasyOCR (this works with your current setup)
#         self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
         
#         # No filtering - capture all text
#         self.component_patterns = []  # Empty - we'll capture everything
#         self.combined_pattern = None  # No pattern matching
        
#         # Common OCR misreads - expanded list
#         self.ocr_corrections = {
#             'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8',
#             'G': '6', 'l': '1', 'o': '0', 'i': '1', 's': '5',
#             'D': '0', 'Q': '0', 'U': '0'  # Additional common misreads
#         }

#     def advanced_preprocessing(self, image):
#         """Create multiple preprocessing versions optimized for OCR"""
#         versions = {}
        
#         # Original
#         versions['original'] = image.copy()
        
#         # Grayscale
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.copy()
#         versions['gray'] = gray
        
#         # High contrast binary (Otsu's method)
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         versions['binary_otsu'] = binary
        
#         # Adaptive threshold (good for varying lighting)
#         adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY, 21, 10)
#         versions['adaptive'] = adaptive
        
#         # Inverted versions (for white text on dark background)
#         versions['inverted_gray'] = cv2.bitwise_not(gray)
#         versions['inverted_binary'] = cv2.bitwise_not(binary)
#         versions['inverted_adaptive'] = cv2.bitwise_not(adaptive)
        
#         # Enhanced contrast using CLAHE
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
#         versions['enhanced_contrast'] = enhanced
        
#         # Denoised version
#         denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#         versions['denoised'] = denoised
        
#         # Edge-preserved smoothing
#         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
#         versions['bilateral'] = bilateral
        
#         # Morphological operations to clean text
#         kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
#         versions['morph_close'] = morph_close
        
#         # Dilated version (makes text thicker)
#         kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#         dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
#         versions['dilated'] = dilated
        
#         # Super resolution attempt (resize)
#         height, width = gray.shape
#         if width < 2000:
#             scale = 2000 / width
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#             versions['upscaled'] = resized
        
#         return versions

#     def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7, offset_x=0, offset_y=0):
#         """Extract text using EasyOCR with confidence scores and bounding boxes"""
#         try:
#             # Run EasyOCR with specific settings
#             results = self.easy_reader.readtext(
#                 image, 
#                 detail=detail,
#                 width_ths=width_ths,
#                 height_ths=height_ths,
#                 paragraph=False,
#                 text_threshold=0.3,
#                 low_text=0.3,
#                 mag_ratio=1.5
#             )
            
#             text_with_conf_bbox = []
            
#             if detail == 1:
#                 # Return tuples of (text, confidence, bbox)
#                 for result in results:
#                     if len(result) >= 3 and result[2] > 0.2:  # confidence threshold
#                         bbox_points = result[0]
#                         # Calculate center point of bbox
#                         x_coords = [p[0] + offset_x for p in bbox_points]
#                         y_coords = [p[1] + offset_y for p in bbox_points]
#                         center_x = sum(x_coords) / len(x_coords)
#                         center_y = sum(y_coords) / len(y_coords)
                        
#                         # Adjust bbox points with offset
#                         adjusted_bbox = [[p[0] + offset_x, p[1] + offset_y] for p in bbox_points]
                        
#                         text_with_conf_bbox.append({
#                             'text': str(result[1]).upper(),
#                             'confidence': float(result[2]),
#                             'bbox': adjusted_bbox,
#                             'center': (center_x, center_y)
#                         })
#             else:
#                 # When detail=0, no bbox available
#                 text_with_conf_bbox = [{'text': str(text).upper(), 'confidence': 0.5, 'bbox': None, 'center': None} 
#                                       for text in results if text]
            
#             return text_with_conf_bbox
#         except Exception as e:
#             print(f"    EasyOCR error: {e}")
#             return []

#     def extract_text_with_tesseract(self, image):
#         """Extract text using Tesseract with confidence scores"""
#         text_with_conf = []
        
#         # Try getting individual words with confidence scores
#         try:
#             data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#             n_boxes = len(data['text'])
#             for i in range(n_boxes):
#                 if data['text'][i].strip():
#                     confidence = float(data['conf'][i])
#                     if confidence > 0:
#                         # Get bounding box
#                         x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#                         center_x = x + w/2
#                         center_y = y + h/2
#                         bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        
#                         text_with_conf.append({
#                             'text': data['text'][i].upper(),
#                             'confidence': confidence / 100.0,
#                             'bbox': bbox,
#                             'center': (center_x, center_y)
#                         })
#         except Exception as e:
#             print(f"    Tesseract data extraction error: {e}")
        
#         # Other configurations without confidence
#         configs = [
#             r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             r'--psm 8',
#             r'--psm 12',
#             r'--psm 13',
#         ]
        
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(image, config=config)
#                 if text.strip():
#                     words = text.upper().split()
#                     for word in words:
#                         word = word.strip()
#                         if word and not any(item['text'] == word for item in text_with_conf):
#                             text_with_conf.append({
#                                 'text': word,
#                                 'confidence': 0.5,
#                                 'bbox': None,
#                                 'center': None
#                             })
#             except:
#                 pass
            
#         return text_with_conf

#     def correct_ocr_errors(self, text):
#         """Correct common OCR misreads"""
#         corrected = text
        
#         # Only apply corrections if the text looks like a component identifier
#         if re.match(r'^[A-Z]+\d*$', text) or re.match(r'^\d+[A-Z]*$', text):
#             for wrong, right in self.ocr_corrections.items():
#                 corrected = corrected.replace(wrong, right)
        
#         return corrected

#     def extract_components_from_texts(self, text_with_conf_bbox_list):
#         """Extract ALL text with confidence scores and locations (no filtering)"""
#         # Combine all results
#         all_items = {}  # Use dict to track best result for each text
        
#         for item in text_with_conf_bbox_list:
#             if not isinstance(item, dict):
#                 continue
                
#             text = item.get('text', '')
#             conf = item.get('confidence', 0.0)
#             bbox = item.get('bbox', None)
#             center = item.get('center', None)
            
#             # Split by common separators and clean
#             items = re.split(r'[\s,;:\n\r\t]+', text)
#             for sub_item in items:
#                 sub_item = sub_item.strip()
#                 if not sub_item:
#                     continue
                    
#                 # Apply OCR corrections
#                 corrected = self.correct_ocr_errors(sub_item)
#                 if corrected and len(corrected) > 0:
#                     # Keep the highest confidence version with location info
#                     if corrected not in all_items or conf > all_items[corrected]['confidence']:
#                         all_items[corrected] = {
#                             'text': corrected,
#                             'confidence': conf,
#                             'bbox': bbox,
#                             'center': center
#                         }
        
#         # Convert to list sorted by confidence
#         return sorted(all_items.values(), key=lambda x: x['confidence'], reverse=True)

#     def process_image(self, image_path):
#         """Main processing function that returns text with confidence scores"""
#         # Read image
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"    Error: Could not read image {image_path}")
#             return [], {}
        
#         # Get all preprocessing versions
#         versions = self.advanced_preprocessing(image)
        
#         all_text_with_conf = []  # List of text info dicts
        
#         print("  Processing with multiple OCR engines and preprocessing...")
        
#         # Process each version
#         for version_name, processed_img in versions.items():
#             if processed_img is None:
#                 continue
                
#             # Convert grayscale to BGR for EasyOCR if needed
#             if len(processed_img.shape) == 2:
#                 img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 img_for_ocr = processed_img
            
#             # EasyOCR
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr)
#             all_text_with_conf.extend(text_conf_list)
            
#             # EasyOCR with different parameters
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
#             all_text_with_conf.extend(text_conf_list)
            
#             # Tesseract
#             text_conf_list = self.extract_text_with_tesseract(processed_img)
#             all_text_with_conf.extend(text_conf_list)
        
#         # Try cropping and processing regions
#         height, width = image.shape[:2]
#         quadrants = [
#             (0, 0, width//2, height//2),
#             (width//2, 0, width, height//2),
#             (0, height//2, width//2, height),
#             (width//2, height//2, width, height)
#         ]
        
#         for x1, y1, x2, y2 in quadrants:
#             cropped = image[y1:y2, x1:x2]
#             # Pass offset to adjust coordinates back to full image
#             text_conf_list = self.extract_text_with_easyocr(cropped, offset_x=x1, offset_y=y1)
#             all_text_with_conf.extend(text_conf_list)
        
#         # Process and deduplicate
#         final_results = self.extract_components_from_texts(all_text_with_conf)
        
#         return final_results, versions

#     def visualize_results(self, image, text_with_info, output_path):
#         """Create visualization of detected text"""
#         try:
#             # Set matplotlib to use non-interactive backend
#             import matplotlib
#             matplotlib.use('Agg')  # Use non-GUI backend
#             import matplotlib.pyplot as plt
#             import matplotlib.patches as patches
            
#             fig, ax = plt.subplots(1, figsize=(15, 10))
#             ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
#             # Draw bounding boxes for all detected text
#             for i, item in enumerate(text_with_info):
#                 if item.get('bbox') and item.get('confidence', 0) > 0.2:
#                     # Draw bounding box
#                     points = item['bbox']
#                     rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=2)
#                     ax.add_patch(rect)
                    
#                     # Add label
#                     ax.text(points[0][0], points[0][1]-5, 
#                            f"{item['text']} ({item['confidence']:.2f})", 
#                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
#                            fontsize=8)
            
#             ax.set_title(f'Circuit OCR Results - Found {len(text_with_info)} text items', 
#                         fontsize=16, fontweight='bold')
#             ax.axis('off')
#             plt.tight_layout()
#             plt.savefig(output_path, dpi=200, bbox_inches='tight')
#             plt.close()
#         except Exception as e:
#             print(f"    Warning: Could not create visualization: {e}")
#             # Create a simple text visualization instead
#             with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
#                 f.write(f"Found {len(text_with_info)} text items:\n")
#                 for item in text_with_info:
#                     f.write(f"{item['text']} (conf: {item['confidence']:.3f})\n")

# # ===== OCR PROCESSING FUNCTION =====
# def process_circuit_ocr(image_path, output_folder, ocr_engine, components=None):
#     """Process OCR for circuit image and extract ALL text with confidence scores and proximity to components"""
#     print("\n  Running OCR text extraction...")
    
#     # Create OCR subfolder
#     ocr_output_folder = os.path.join(output_folder, "ocr_results")
#     os.makedirs(ocr_output_folder, exist_ok=True)
    
#     # Read image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"    Error: Could not read image {image_path}")
#         return []
    
#     # Extract all text with confidence and location
#     text_with_info, versions = ocr_engine.process_image(image_path)
    
#     # Save some preprocessing versions for debugging
#     for i, (name, img) in enumerate(list(versions.items())[:4]):
#         if img is not None:
#             cv2.imwrite(os.path.join(ocr_output_folder, f"ocr_preprocess_{name}.png"), img)
    
#     # Visualize
#     if text_with_info:
#         ocr_engine.visualize_results(image, text_with_info, 
#                                    os.path.join(ocr_output_folder, "ocr_detected_text.png"))
    
#     # Save results with confidence scores
#     with open(os.path.join(ocr_output_folder, "extracted_text_with_confidence.txt"), 'w', encoding='utf-8') as f:
#         if text_with_info:
#             for item in text_with_info:
#                 f.write(f"{item['text']}\t{item['confidence']:.3f}\n")
#             print(f"    Found {len(text_with_info)} text items")
#             # Show top 10 with confidence
#             print("    Top OCR detections:")
#             for item in text_with_info[:10]:
#                 print(f"      {item['text']} (confidence: {item['confidence']:.3f})")
#             if len(text_with_info) > 10:
#                 print(f"      ... and {len(text_with_info) - 10} more")
#         else:
#             f.write("No text detected")
#             print("    No text detected")
    
#     # Save just the text (without confidence) for compatibility
#     all_text = [item['text'] for item in text_with_info]
#     with open(os.path.join(ocr_output_folder, "extracted_text.txt"), 'w', encoding='utf-8') as f:
#         if all_text:
#             f.write('\n'.join(all_text))
    
#     # If components are provided, create proximity-based mapping
#     if components and text_with_info:
#         print("    Analyzing text proximity to components...")
        
#         # Filter text_with_info to only include items with center coordinates
#         text_with_locations = [item for item in text_with_info if item.get('center') is not None]
        
#         if not text_with_locations:
#             print("    Warning: No text items have location information for proximity analysis")
#             return text_with_info
        
#         component_text_mapping = assign_text_to_nearest_component(text_with_locations, components)
        
#         # Save proximity-based summary
#         with open(os.path.join(ocr_output_folder, "text_by_component_proximity.txt"), 'w', encoding='utf-8') as f:
#             f.write("OCR Text Assigned to Nearest Components\n")
#             f.write("="*60 + "\n")
#             f.write("Note: Each text appears only under its closest component\n")
#             f.write("="*60 + "\n\n")
            
#             for comp_id, text_items in component_text_mapping.items():
#                 # Find component info
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"Component: {comp_id} (Type: {comp_info['type']})\n")
#                     f.write(f"Bounding Box: {comp_info['bbox']}\n")
#                     f.write("-"*40 + "\n")
                    
#                     if text_items:
#                         f.write(f"Assigned Text ({len(text_items)} items, sorted by distance):\n")
#                         for i, item in enumerate(text_items):
#                             f.write(f"  {i+1}. {item['text']:<15} (conf: {item['confidence']:.3f}, dist: {item['distance']:.1f}px)\n")
#                     else:
#                         f.write("  No text assigned to this component\n")
#                     f.write("\n")
        
#         # Create visual proximity map showing assignments
#         proximity_vis = image.copy()
        
#         # Color map for components
#         colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
#         # Draw components and their assigned texts
#         for idx, (comp_id, text_items) in enumerate(component_text_mapping.items()):
#             comp_info = next((c for c in components if c['id'] == comp_id), None)
#             if comp_info and 'bbox' in comp_info:
#                 bbox = comp_info['bbox']
#                 if len(bbox) == 4:
#                     x_min, y_min, x_max, y_max = bbox
#                     color = colors[idx % len(colors)]
                    
#                     # Draw component box
#                     cv2.rectangle(proximity_vis, (x_min, y_min), (x_max, y_max), color, 2)
#                     cv2.putText(proximity_vis, comp_id, (x_min, y_min - 5),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
#                     # Draw lines to all assigned texts
#                     comp_center = get_component_center(comp_info['bbox'])
#                     for text_item in text_items:
#                         if text_item.get('center'):
#                             # Draw line from component to text
#                             cv2.line(proximity_vis, 
#                                     (int(comp_center[0]), int(comp_center[1])), 
#                                     (int(text_item['center'][0]), int(text_item['center'][1])), 
#                                     color, 1)
#                             # Draw text with same color as component
#                             cv2.putText(proximity_vis, text_item['text'], 
#                                        (int(text_item['center'][0]), int(text_item['center'][1]) - 5),
#                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(os.path.join(ocr_output_folder, "text_component_assignments.png"), proximity_vis)
        
#         # Also create a summary statistics file
#         with open(os.path.join(ocr_output_folder, "assignment_statistics.txt"), 'w', encoding='utf-8') as f:
#             f.write("Text Assignment Statistics\n")
#             f.write("="*40 + "\n\n")
            
#             total_assigned = sum(len(items) for items in component_text_mapping.values())
#             f.write(f"Total texts detected: {len(text_with_info)}\n")
#             f.write(f"Total texts with locations: {len(text_with_locations)}\n")
#             f.write(f"Total texts assigned to components: {total_assigned}\n")
#             f.write(f"Total components: {len(components)}\n\n")
            
#             f.write("Texts per component:\n")
#             for comp_id, text_items in component_text_mapping.items():
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"  {comp_id} ({comp_info.get('type', 'unknown')}): {len(text_items)} texts\n")
    
#     # Save detailed summary
#     with open(os.path.join(ocr_output_folder, "ocr_summary.txt"), 'w', encoding='utf-8') as f:
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total text items found: {len(text_with_info)}\n\n")
#         f.write("All extracted text with confidence scores:\n")
#         f.write("-" * 50 + "\n")
#         f.write("Text\t\t\tConfidence\n")
#         f.write("-" * 50 + "\n")
#         for item in text_with_info:
#             f.write(f"{item['text']:<20}\t{item['confidence']:.3f}\n")
    
#     return text_with_info



# import cv2
# import os
# from helper_files.proximity_analysis import assign_text_to_nearest_component, get_component_center
# import re
# import easyocr
# import pytesseract
# import matplotlib
# matplotlib.use('Agg') 
# import numpy as np

# # ===== OCR CLASS DEFINITION =====
# class CircuitOCR:
#     def __init__(self, use_gpu=True):
#         """Initialize OCR engines"""
#         print("Initializing OCR engines...")
        
#         # Initialize EasyOCR (this works with your current setup)
#         self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
         
#         # No filtering - capture all text
#         self.component_patterns = []  # Empty - we'll capture everything
#         self.combined_pattern = None  # No pattern matching
        
#         # Common OCR misreads - expanded list
#         self.ocr_corrections = {
#             'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8',
#             'G': '6', 'l': '1', 'o': '0', 'i': '1', 's': '5',
#             'D': '0', 'Q': '0', 'U': '0'  # Additional common misreads
#         }

#     def advanced_preprocessing(self, image):
#         """Create multiple preprocessing versions optimized for OCR"""
#         versions = {}
        
#         # Original
#         versions['original'] = image.copy()
        
#         # Grayscale
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.copy()
#         versions['gray'] = gray
        
#         # High contrast binary (Otsu's method)
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         versions['binary_otsu'] = binary
        
#         # Adaptive threshold (good for varying lighting)
#         adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY, 21, 10)
#         versions['adaptive'] = adaptive
        
#         # Inverted versions (for white text on dark background)
#         versions['inverted_gray'] = cv2.bitwise_not(gray)
#         versions['inverted_binary'] = cv2.bitwise_not(binary)
#         versions['inverted_adaptive'] = cv2.bitwise_not(adaptive)
        
#         # Enhanced contrast using CLAHE
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
#         versions['enhanced_contrast'] = enhanced
        
#         # Denoised version
#         denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#         versions['denoised'] = denoised
        
#         # Edge-preserved smoothing
#         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
#         versions['bilateral'] = bilateral
        
#         # Morphological operations to clean text
#         kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
#         versions['morph_close'] = morph_close
        
#         # Dilated version (makes text thicker)
#         kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#         dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
#         versions['dilated'] = dilated
        
#         # Super resolution attempt (resize)
#         height, width = gray.shape
#         if width < 2000:
#             scale = 2000 / width
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#             versions['upscaled'] = resized
        
#         return versions

#     def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7, offset_x=0, offset_y=0):
#         """Extract text using EasyOCR with confidence scores and bounding boxes"""
#         try:
#             # Get image dimensions for validation
#             if len(image.shape) == 3:
#                 img_height, img_width, _ = image.shape
#             else:
#                 img_height, img_width = image.shape
            
#             # Run EasyOCR with specific settings
#             results = self.easy_reader.readtext(
#                 image, 
#                 detail=detail,
#                 width_ths=width_ths,
#                 height_ths=height_ths,
#                 paragraph=False,
#                 text_threshold=0.3,
#                 low_text=0.3,
#                 mag_ratio=1.5
#             )
            
#             text_with_conf_bbox = []
            
#             if detail == 1:
#                 # Return tuples of (text, confidence, bbox)
#                 for result in results:
#                     if len(result) >= 3 and result[2] > 0.3:  # Increased confidence threshold
#                         bbox_points = result[0]
#                         text = str(result[1]).strip()
                        
#                         # Skip empty or very short text
#                         if not text or len(text) < 1:
#                             continue
                        
#                         # Calculate center point of bbox
#                         x_coords = [p[0] + offset_x for p in bbox_points]
#                         y_coords = [p[1] + offset_y for p in bbox_points]
                        
#                         # Validate coordinates are within image bounds
#                         if offset_x == 0 and offset_y == 0:  # Only validate for non-cropped images
#                             if any(x < 0 or x > img_width for x in x_coords):
#                                 continue
#                             if any(y < 0 or y > img_height for y in y_coords):
#                                 continue
                        
#                         center_x = sum(x_coords) / len(x_coords)
#                         center_y = sum(y_coords) / len(y_coords)
                        
#                         # Adjust bbox points with offset
#                         adjusted_bbox = [[p[0] + offset_x, p[1] + offset_y] for p in bbox_points]
                        
#                         text_with_conf_bbox.append({
#                             'text': text.upper(),
#                             'confidence': float(result[2]),
#                             'bbox': adjusted_bbox,
#                             'center': (center_x, center_y)
#                         })
#             else:
#                 # When detail=0, no bbox available
#                 text_with_conf_bbox = [{'text': str(text).upper(), 'confidence': 0.5, 'bbox': None, 'center': None} 
#                                       for text in results if text and str(text).strip()]
            
#             return text_with_conf_bbox
#         except Exception as e:
#             print(f"    EasyOCR error: {e}")
#             return []

#     def extract_text_with_tesseract(self, image):
#         """Extract text using Tesseract with confidence scores"""
#         text_with_conf = []
        
#         # Try getting individual words with confidence scores
#         try:
#             data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#             n_boxes = len(data['text'])
#             for i in range(n_boxes):
#                 if data['text'][i].strip():
#                     confidence = float(data['conf'][i])
#                     if confidence > 0:
#                         # Get bounding box
#                         x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#                         center_x = x + w/2
#                         center_y = y + h/2
#                         bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        
#                         text_with_conf.append({
#                             'text': data['text'][i].upper(),
#                             'confidence': confidence / 100.0,
#                             'bbox': bbox,
#                             'center': (center_x, center_y)
#                         })
#         except Exception as e:
#             print(f"    Tesseract data extraction error: {e}")
        
#         # Other configurations without confidence
#         configs = [
#             r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             r'--psm 8',
#             r'--psm 12',
#             r'--psm 13',
#         ]
        
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(image, config=config)
#                 if text.strip():
#                     words = text.upper().split()
#                     for word in words:
#                         word = word.strip()
#                         if word and not any(item['text'] == word for item in text_with_conf):
#                             text_with_conf.append({
#                                 'text': word,
#                                 'confidence': 0.5,
#                                 'bbox': None,
#                                 'center': None
#                             })
#             except:
#                 pass
            
#         return text_with_conf

#     def correct_ocr_errors(self, text):
#         """Correct common OCR misreads"""
#         corrected = text
        
#         # Only apply corrections if the text looks like a component identifier
#         if re.match(r'^[A-Z]+\d*$', text) or re.match(r'^\d+[A-Z]*$', text):
#             for wrong, right in self.ocr_corrections.items():
#                 corrected = corrected.replace(wrong, right)
        
#         return corrected

#     def extract_components_from_texts(self, text_with_conf_bbox_list):
#         """Extract ALL text with confidence scores and locations (no filtering)"""
#         # Combine all results
#         all_items = {}  # Use dict to track best result for each text
        
#         for item in text_with_conf_bbox_list:
#             if not isinstance(item, dict):
#                 continue
                
#             text = item.get('text', '')
#             conf = item.get('confidence', 0.0)
#             bbox = item.get('bbox', None)
#             center = item.get('center', None)
            
#             # Split by common separators and clean
#             items = re.split(r'[\s,;:\n\r\t]+', text)
#             for sub_item in items:
#                 sub_item = sub_item.strip()
#                 if not sub_item:
#                     continue
                    
#                 # Apply OCR corrections
#                 corrected = self.correct_ocr_errors(sub_item)
#                 if corrected and len(corrected) > 0:
#                     # Keep the highest confidence version with location info
#                     if corrected not in all_items or conf > all_items[corrected]['confidence']:
#                         all_items[corrected] = {
#                             'text': corrected,
#                             'confidence': conf,
#                             'bbox': bbox,
#                             'center': center
#                         }
        
#         # Convert to list sorted by confidence
#         return sorted(all_items.values(), key=lambda x: x['confidence'], reverse=True)

#     def process_image(self, image_path):
#         """Main processing function that returns text with confidence scores"""
#         # Read image
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"    Error: Could not read image {image_path}")
#             return [], {}
        
#         # Get original image dimensions
#         orig_height, orig_width = image.shape[:2]
        
#         # Get all preprocessing versions
#         versions = self.advanced_preprocessing(image)
        
#         all_text_with_conf = []  # List of text info dicts
        
#         print("  Processing with multiple OCR engines and preprocessing...")
        
#         # Process each version
#         for version_name, processed_img in versions.items():
#             if processed_img is None:
#                 continue
                
#             # Skip upscaled version for initial processing to avoid duplicate detections
#             if version_name == 'upscaled':
#                 continue
                
#             # Convert grayscale to BGR for EasyOCR if needed
#             if len(processed_img.shape) == 2:
#                 img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 img_for_ocr = processed_img
            
#             # Ensure we're working with original dimensions
#             if img_for_ocr.shape[:2] != (orig_height, orig_width):
#                 continue
            
#             # EasyOCR
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr)
#             all_text_with_conf.extend(text_conf_list)
            
#             # EasyOCR with different parameters for smaller text
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
#             all_text_with_conf.extend(text_conf_list)
            
#             # Tesseract (only on select versions to reduce false positives)
#             if version_name in ['binary_otsu', 'adaptive', 'enhanced_contrast']:
#                 text_conf_list = self.extract_text_with_tesseract(processed_img)
#                 all_text_with_conf.extend(text_conf_list)
        
#         # Try cropping and processing regions (but validate results more strictly)
#         height, width = image.shape[:2]
#         quadrants = [
#             (0, 0, width//2, height//2),
#             (width//2, 0, width, height//2),
#             (0, height//2, width//2, height),
#             (width//2, height//2, width, height)
#         ]
        
#         for x1, y1, x2, y2 in quadrants:
#             cropped = image[y1:y2, x1:x2]
#             # Only process if the region has sufficient content
#             gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#             if cv2.countNonZero(gray_crop) > 0.1 * gray_crop.size:  # At least 10% non-zero pixels
#                 text_conf_list = self.extract_text_with_easyocr(cropped, offset_x=x1, offset_y=y1)
#                 all_text_with_conf.extend(text_conf_list)
        
#         # Filter out detections that are likely false positives
#         filtered_text_with_conf = []
#         for item in all_text_with_conf:
#             if not isinstance(item, dict):
#                 continue
            
#             # Check if the detection has valid bounds
#             if item.get('bbox') and item.get('center'):
#                 bbox = item['bbox']
#                 if len(bbox) >= 4:
#                     x_coords = [p[0] for p in bbox]
#                     y_coords = [p[1] for p in bbox]
                    
#                     # Validate bounds
#                     if (all(0 <= x <= orig_width for x in x_coords) and 
#                         all(0 <= y <= orig_height for y in y_coords)):
#                         filtered_text_with_conf.append(item)
#             else:
#                 # Keep items without bbox info (from Tesseract simple mode)
#                 filtered_text_with_conf.append(item)
        
#         # Process and deduplicate
#         final_results = self.extract_components_from_texts(filtered_text_with_conf)
        
#         return final_results, versions

#     def visualize_results(self, image, text_with_info, output_path):
#         """Create visualization of detected text"""
#         try:
#             # Set matplotlib to use non-interactive backend
#             import matplotlib
#             matplotlib.use('Agg')  # Use non-GUI backend
#             import matplotlib.pyplot as plt
#             import matplotlib.patches as patches
            
#             # Create figure with proper size matching the image aspect ratio
#             height, width = image.shape[:2]
#             fig_width = 15
#             fig_height = fig_width * height / width
            
#             fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
            
#             # Display the image
#             ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
#             # Set the axis limits to match image dimensions exactly
#             ax.set_xlim(0, width)
#             ax.set_ylim(height, 0)  # Inverted for image coordinates
            
#             # Filter out invalid detections
#             valid_detections = []
#             for item in text_with_info:
#                 if item.get('bbox') and item.get('confidence', 0) > 0.3:  # Increased threshold
#                     # Check if bbox is within image bounds
#                     bbox = item['bbox']
#                     if bbox and len(bbox) >= 4:
#                         x_coords = [p[0] for p in bbox]
#                         y_coords = [p[1] for p in bbox]
                        
#                         # Check if coordinates are valid
#                         if (all(0 <= x <= width for x in x_coords) and 
#                             all(0 <= y <= height for y in y_coords) and
#                             max(x_coords) - min(x_coords) > 5 and  # Minimum width
#                             max(y_coords) - min(y_coords) > 5):     # Minimum height
#                             valid_detections.append(item)
            
#             # Draw bounding boxes for valid detections only
#             for i, item in enumerate(valid_detections):
#                 points = item['bbox']
#                 rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=2)
#                 ax.add_patch(rect)
                
#                 # Add label
#                 ax.text(points[0][0], points[0][1]-5, 
#                        f"{item['text']} ({item['confidence']:.2f})", 
#                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
#                        fontsize=8)
            
#             ax.set_title(f'Circuit OCR Results - Found {len(valid_detections)} valid text items', 
#                         fontsize=16, fontweight='bold')
#             ax.axis('off')
            
#             # Save with proper DPI and bounds
#             plt.tight_layout()
#             plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
#             plt.close()
            
#             print(f"    Visualization saved with {len(valid_detections)} valid detections out of {len(text_with_info)} total")
            
#         except Exception as e:
#             print(f"    Warning: Could not create visualization: {e}")
#             # Create a simple text visualization instead
#             with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
#                 f.write(f"Found {len(text_with_info)} text items:\n")
#                 for item in text_with_info:
#                     f.write(f"{item['text']} (conf: {item['confidence']:.3f})\n")

# # ===== OCR PROCESSING FUNCTION =====
# def process_circuit_ocr(image_path, output_folder, ocr_engine, components=None):
#     """Process OCR for circuit image and extract ALL text with confidence scores and proximity to components"""
#     print("\n  Running OCR text extraction...")
    
#     # Create OCR subfolder
#     ocr_output_folder = os.path.join(output_folder, "ocr_results")
#     os.makedirs(ocr_output_folder, exist_ok=True)
    
#     # Read image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"    Error: Could not read image {image_path}")
#         return []
    
#     # Extract all text with confidence and location
#     text_with_info, versions = ocr_engine.process_image(image_path)
    
#     # Save some preprocessing versions for debugging
#     for i, (name, img) in enumerate(list(versions.items())[:4]):
#         if img is not None:
#             cv2.imwrite(os.path.join(ocr_output_folder, f"ocr_preprocess_{name}.png"), img)
    
#     # Visualize
#     if text_with_info:
#         ocr_engine.visualize_results(image, text_with_info, 
#                                    os.path.join(ocr_output_folder, "ocr_detected_text.png"))
    
#     # Save results with confidence scores
#     with open(os.path.join(ocr_output_folder, "extracted_text_with_confidence.txt"), 'w', encoding='utf-8') as f:
#         if text_with_info:
#             for item in text_with_info:
#                 f.write(f"{item['text']}\t{item['confidence']:.3f}\n")
#             print(f"    Found {len(text_with_info)} text items")
#             # Show top 10 with confidence
#             print("    Top OCR detections:")
#             for item in text_with_info[:10]:
#                 print(f"      {item['text']} (confidence: {item['confidence']:.3f})")
#             if len(text_with_info) > 10:
#                 print(f"      ... and {len(text_with_info) - 10} more")
#         else:
#             f.write("No text detected")
#             print("    No text detected")
    
#     # Save just the text (without confidence) for compatibility
#     all_text = [item['text'] for item in text_with_info]
#     with open(os.path.join(ocr_output_folder, "extracted_text.txt"), 'w', encoding='utf-8') as f:
#         if all_text:
#             f.write('\n'.join(all_text))
    
#     # If components are provided, create proximity-based mapping
#     if components and text_with_info:
#         print("    Analyzing text proximity to components...")
        
#         # Filter text_with_info to only include items with center coordinates
#         text_with_locations = [item for item in text_with_info if item.get('center') is not None]
        
#         if not text_with_locations:
#             print("    Warning: No text items have location information for proximity analysis")
#             return text_with_info
        
#         component_text_mapping = assign_text_to_nearest_component(text_with_locations, components)
        
#         # Save proximity-based summary
#         with open(os.path.join(ocr_output_folder, "text_by_component_proximity.txt"), 'w', encoding='utf-8') as f:
#             f.write("OCR Text Assigned to Nearest Components\n")
#             f.write("="*60 + "\n")
#             f.write("Note: Each text appears only under its closest component\n")
#             f.write("="*60 + "\n\n")
            
#             for comp_id, text_items in component_text_mapping.items():
#                 # Find component info
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"Component: {comp_id} (Type: {comp_info['type']})\n")
#                     f.write(f"Bounding Box: {comp_info['bbox']}\n")
#                     f.write("-"*40 + "\n")
                    
#                     if text_items:
#                         f.write(f"Assigned Text ({len(text_items)} items, sorted by distance):\n")
#                         for i, item in enumerate(text_items):
#                             f.write(f"  {i+1}. {item['text']:<15} (conf: {item['confidence']:.3f}, dist: {item['distance']:.1f}px)\n")
#                     else:
#                         f.write("  No text assigned to this component\n")
#                     f.write("\n")
        
#         # Create visual proximity map showing assignments
#         proximity_vis = image.copy()
        
#         # Color map for components
#         colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
#         # Draw components and their assigned texts
#         for idx, (comp_id, text_items) in enumerate(component_text_mapping.items()):
#             comp_info = next((c for c in components if c['id'] == comp_id), None)
#             if comp_info and 'bbox' in comp_info:
#                 bbox = comp_info['bbox']
#                 if len(bbox) == 4:
#                     x_min, y_min, x_max, y_max = bbox
#                     color = colors[idx % len(colors)]
                    
#                     # Draw component box
#                     cv2.rectangle(proximity_vis, (x_min, y_min), (x_max, y_max), color, 2)
#                     cv2.putText(proximity_vis, comp_id, (x_min, y_min - 5),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
#                     # Draw lines to all assigned texts
#                     comp_center = get_component_center(comp_info['bbox'])
#                     for text_item in text_items:
#                         if text_item.get('center'):
#                             # Draw line from component to text
#                             cv2.line(proximity_vis, 
#                                     (int(comp_center[0]), int(comp_center[1])), 
#                                     (int(text_item['center'][0]), int(text_item['center'][1])), 
#                                     color, 1)
#                             # Draw text with same color as component
#                             cv2.putText(proximity_vis, text_item['text'], 
#                                        (int(text_item['center'][0]), int(text_item['center'][1]) - 5),
#                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(os.path.join(ocr_output_folder, "text_component_assignments.png"), proximity_vis)
        
#         # Also create a summary statistics file
#         with open(os.path.join(ocr_output_folder, "assignment_statistics.txt"), 'w', encoding='utf-8') as f:
#             f.write("Text Assignment Statistics\n")
#             f.write("="*40 + "\n\n")
            
#             total_assigned = sum(len(items) for items in component_text_mapping.values())
#             f.write(f"Total texts detected: {len(text_with_info)}\n")
#             f.write(f"Total texts with locations: {len(text_with_locations)}\n")
#             f.write(f"Total texts assigned to components: {total_assigned}\n")
#             f.write(f"Total components: {len(components)}\n\n")
            
#             f.write("Texts per component:\n")
#             for comp_id, text_items in component_text_mapping.items():
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"  {comp_id} ({comp_info.get('type', 'unknown')}): {len(text_items)} texts\n")
    
#     # Save detailed summary
#     with open(os.path.join(ocr_output_folder, "ocr_summary.txt"), 'w', encoding='utf-8') as f:
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total text items found: {len(text_with_info)}\n\n")
#         f.write("All extracted text with confidence scores:\n")
#         f.write("-" * 50 + "\n")
#         f.write("Text\t\t\tConfidence\n")
#         f.write("-" * 50 + "\n")
#         for item in text_with_info:
#             f.write(f"{item['text']:<20}\t{item['confidence']:.3f}\n")
    
#     return text_with_info






# # // versionn 4
# import cv2
# import os
# from helper_files.proximity_analysis import assign_text_to_nearest_component, get_component_center
# import re
# import easyocr
# import pytesseract
# import matplotlib
# matplotlib.use('Agg') 
# import numpy as np

# # ===== OCR CLASS DEFINITION =====
# class CircuitOCR:
#     def __init__(self, use_gpu=True):
#         """Initialize OCR engines"""
#         print("Initializing OCR engines...")
        
#         # Initialize EasyOCR (this works with your current setup)
#         self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
         
#         # No filtering - capture all text
#         self.component_patterns = []  # Empty - we'll capture everything
#         self.combined_pattern = None  # No pattern matching
        
#         # Common OCR misreads - expanded list
#         self.ocr_corrections = {
#             'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8',
#             'G': '6', 'l': '1', 'o': '0', 'i': '1', 's': '5',
#             'D': '0', 'Q': '0', 'U': '0'  # Additional common misreads
#         }

#     def advanced_preprocessing(self, image):
#         """Create multiple preprocessing versions optimized for OCR"""
#         versions = {}
        
#         # Original
#         versions['original'] = image.copy()
        
#         # Grayscale
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.copy()
#         versions['gray'] = gray
        
#         # High contrast binary (Otsu's method)
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         versions['binary_otsu'] = binary
        
#         # Adaptive threshold (good for varying lighting)
#         adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY, 21, 10)
#         versions['adaptive'] = adaptive
        
#         # Inverted versions (for white text on dark background)
#         versions['inverted_gray'] = cv2.bitwise_not(gray)
#         versions['inverted_binary'] = cv2.bitwise_not(binary)
#         versions['inverted_adaptive'] = cv2.bitwise_not(adaptive)
        
#         # Enhanced contrast using CLAHE
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
#         versions['enhanced_contrast'] = enhanced
        
#         # Denoised version
#         denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#         versions['denoised'] = denoised
        
#         # Edge-preserved smoothing
#         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
#         versions['bilateral'] = bilateral
        
#         # Morphological operations to clean text
#         kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
#         versions['morph_close'] = morph_close
        
#         # Dilated version (makes text thicker)
#         kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#         dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
#         versions['dilated'] = dilated
        
#         # Super resolution attempt (resize)
#         height, width = gray.shape
#         if width < 2000:
#             scale = 2000 / width
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#             versions['upscaled'] = resized
        
#         return versions

#     def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7, offset_x=0, offset_y=0):
#         """Extract text using EasyOCR with confidence scores and bounding boxes"""
#         try:
#             # Get image dimensions for validation
#             if len(image.shape) == 3:
#                 img_height, img_width, _ = image.shape
#             else:
#                 img_height, img_width = image.shape
            
#             # Run EasyOCR with specific settings
#             results = self.easy_reader.readtext(
#                 image, 
#                 detail=detail,
#                 width_ths=width_ths,
#                 height_ths=height_ths,
#                 paragraph=False,
#                 text_threshold=0.3,
#                 low_text=0.3,
#                 mag_ratio=1.5
#             )
            
#             text_with_conf_bbox = []
            
#             if detail == 1:
#                 # Return tuples of (text, confidence, bbox)
#                 for result in results:
#                     if len(result) >= 3 and result[2] > 0.3:  # Increased confidence threshold
#                         bbox_points = result[0]
#                         text = str(result[1]).strip()
                        
#                         # Skip empty or very short text
#                         if not text or len(text) < 1:
#                             continue
                        
#                         # Calculate center point of bbox
#                         x_coords = [p[0] + offset_x for p in bbox_points]
#                         y_coords = [p[1] + offset_y for p in bbox_points]
                        
#                         # Validate coordinates are within image bounds
#                         if offset_x == 0 and offset_y == 0:  # Only validate for non-cropped images
#                             if any(x < 0 or x > img_width for x in x_coords):
#                                 continue
#                             if any(y < 0 or y > img_height for y in y_coords):
#                                 continue
                        
#                         center_x = sum(x_coords) / len(x_coords)
#                         center_y = sum(y_coords) / len(y_coords)
                        
#                         # Adjust bbox points with offset
#                         adjusted_bbox = [[p[0] + offset_x, p[1] + offset_y] for p in bbox_points]
                        
#                         text_with_conf_bbox.append({
#                             'text': text.upper(),
#                             'confidence': float(result[2]),
#                             'bbox': adjusted_bbox,
#                             'center': (center_x, center_y)
#                         })
#             else:
#                 # When detail=0, no bbox available
#                 text_with_conf_bbox = [{'text': str(text).upper(), 'confidence': 0.5, 'bbox': None, 'center': None} 
#                                       for text in results if text and str(text).strip()]
            
#             return text_with_conf_bbox
#         except Exception as e:
#             print(f"    EasyOCR error: {e}")
#             return []

#     def extract_text_with_tesseract(self, image):
#         """Extract text using Tesseract with confidence scores"""
#         text_with_conf = []
        
#         # Try getting individual words with confidence scores
#         try:
#             data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#             n_boxes = len(data['text'])
#             for i in range(n_boxes):
#                 if data['text'][i].strip():
#                     confidence = float(data['conf'][i])
#                     if confidence > 0:
#                         # Get bounding box
#                         x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#                         center_x = x + w/2
#                         center_y = y + h/2
#                         bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        
#                         text_with_conf.append({
#                             'text': data['text'][i].upper(),
#                             'confidence': confidence / 100.0,
#                             'bbox': bbox,
#                             'center': (center_x, center_y)
#                         })
#         except Exception as e:
#             print(f"    Tesseract data extraction error: {e}")
        
#         # Other configurations without confidence
#         configs = [
#             r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             r'--psm 8',
#             r'--psm 12',
#             r'--psm 13',
#         ]
        
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(image, config=config)
#                 if text.strip():
#                     words = text.upper().split()
#                     for word in words:
#                         word = word.strip()
#                         if word and not any(item['text'] == word for item in text_with_conf):
#                             text_with_conf.append({
#                                 'text': word,
#                                 'confidence': 0.5,
#                                 'bbox': None,
#                                 'center': None
#                             })
#             except:
#                 pass
            
#         return text_with_conf

#     def correct_ocr_errors(self, text):
#         """Correct common OCR misreads"""
#         corrected = text
        
#         # Only apply corrections if the text looks like a component identifier
#         if re.match(r'^[A-Z]+\d*$', text) or re.match(r'^\d+[A-Z]*$', text):
#             for wrong, right in self.ocr_corrections.items():
#                 corrected = corrected.replace(wrong, right)
        
#         return corrected

#     def extract_components_from_texts(self, text_with_conf_bbox_list):
#         """Extract ALL text with confidence scores and locations (no filtering)"""
#         # Combine all results
#         all_items = {}  # Use dict to track best result for each text
        
#         for item in text_with_conf_bbox_list:
#             if not isinstance(item, dict):
#                 continue
                
#             text = item.get('text', '')
#             conf = item.get('confidence', 0.0)
#             bbox = item.get('bbox', None)
#             center = item.get('center', None)
            
#             # Split by common separators and clean
#             items = re.split(r'[\s,;:\n\r\t]+', text)
#             for sub_item in items:
#                 sub_item = sub_item.strip()
#                 if not sub_item:
#                     continue
                    
#                 # Apply OCR corrections
#                 corrected = self.correct_ocr_errors(sub_item)
#                 if corrected and len(corrected) > 0:
#                     # Keep the highest confidence version with location info
#                     if corrected not in all_items or conf > all_items[corrected]['confidence']:
#                         all_items[corrected] = {
#                             'text': corrected,
#                             'confidence': conf,
#                             'bbox': bbox,
#                             'center': center
#                         }
        
#         # Convert to list sorted by confidence
#         return sorted(all_items.values(), key=lambda x: x['confidence'], reverse=True)

#     def process_image(self, image_path):
#         """Main processing function that returns text with confidence scores"""
#         # Read image
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"    Error: Could not read image {image_path}")
#             return [], {}
        
#         # Get original image dimensions
#         orig_height, orig_width = image.shape[:2]
        
#         # Get all preprocessing versions
#         versions = self.advanced_preprocessing(image)
        
#         all_text_with_conf = []  # List of text info dicts
        
#         print("  Processing with multiple OCR engines and preprocessing...")
        
#         # Process each version
#         for version_name, processed_img in versions.items():
#             if processed_img is None:
#                 continue
                
#             # Skip upscaled version for initial processing to avoid duplicate detections
#             if version_name == 'upscaled':
#                 continue
                
#             # Convert grayscale to BGR for EasyOCR if needed
#             if len(processed_img.shape) == 2:
#                 img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 img_for_ocr = processed_img
            
#             # Ensure we're working with original dimensions
#             if img_for_ocr.shape[:2] != (orig_height, orig_width):
#                 continue
            
#             # EasyOCR
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr)
#             all_text_with_conf.extend(text_conf_list)
            
#             # EasyOCR with different parameters for smaller text
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
#             all_text_with_conf.extend(text_conf_list)
            
#             # Tesseract (only on select versions to reduce false positives)
#             if version_name in ['binary_otsu', 'adaptive', 'enhanced_contrast']:
#                 text_conf_list = self.extract_text_with_tesseract(processed_img)
#                 all_text_with_conf.extend(text_conf_list)
        
#         # Try cropping and processing regions (but validate results more strictly)
#         height, width = image.shape[:2]
#         quadrants = [
#             (0, 0, width//2, height//2),
#             (width//2, 0, width, height//2),
#             (0, height//2, width//2, height),
#             (width//2, height//2, width, height)
#         ]
        
#         for x1, y1, x2, y2 in quadrants:
#             cropped = image[y1:y2, x1:x2]
#             # Only process if the region has sufficient content
#             gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#             if cv2.countNonZero(gray_crop) > 0.1 * gray_crop.size:  # At least 10% non-zero pixels
#                 text_conf_list = self.extract_text_with_easyocr(cropped, offset_x=x1, offset_y=y1)
#                 all_text_with_conf.extend(text_conf_list)
        
#         # Filter out detections that are likely false positives
#         filtered_text_with_conf = []
#         for item in all_text_with_conf:
#             if not isinstance(item, dict):
#                 continue
            
#             # Check if the detection has valid bounds
#             if item.get('bbox') and item.get('center'):
#                 bbox = item['bbox']
#                 if len(bbox) >= 4:
#                     x_coords = [p[0] for p in bbox]
#                     y_coords = [p[1] for p in bbox]
                    
#                     # Validate bounds
#                     if (all(0 <= x <= orig_width for x in x_coords) and 
#                         all(0 <= y <= orig_height for y in y_coords)):
#                         filtered_text_with_conf.append(item)
#             else:
#                 # Keep items without bbox info (from Tesseract simple mode)
#                 filtered_text_with_conf.append(item)
        
#         # Process and deduplicate
#         final_results = self.extract_components_from_texts(filtered_text_with_conf)
        
#         return final_results, versions

#     def visualize_results(self, image, text_with_info, output_path):
#         """Create visualization of detected text"""
#         try:
#             # Set matplotlib to use non-interactive backend
#             import matplotlib
#             matplotlib.use('Agg')  # Use non-GUI backend
#             import matplotlib.pyplot as plt
#             import matplotlib.patches as patches
            
#             # Create figure with proper size matching the image aspect ratio
#             height, width = image.shape[:2]
#             fig_width = 15
#             fig_height = fig_width * height / width
            
#             fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
            
#             # Display the image
#             ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
#             # Set the axis limits to match image dimensions exactly
#             ax.set_xlim(0, width)
#             ax.set_ylim(height, 0)  # Inverted for image coordinates
            
#             # Filter out invalid detections
#             valid_detections = []
#             for item in text_with_info:
#                 if item.get('bbox') and item.get('confidence', 0) > 0.3:  # Increased threshold
#                     # Check if bbox is within image bounds
#                     bbox = item['bbox']
#                     if bbox and len(bbox) >= 4:
#                         x_coords = [p[0] for p in bbox]
#                         y_coords = [p[1] for p in bbox]
                        
#                         # Check if coordinates are valid
#                         if (all(0 <= x <= width for x in x_coords) and 
#                             all(0 <= y <= height for y in y_coords) and
#                             max(x_coords) - min(x_coords) > 5 and  # Minimum width
#                             max(y_coords) - min(y_coords) > 5):     # Minimum height
#                             valid_detections.append(item)
            
#             # Draw bounding boxes for valid detections only
#             for i, item in enumerate(valid_detections):
#                 points = item['bbox']
#                 rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=2)
#                 ax.add_patch(rect)
                
#                 # Add label
#                 ax.text(points[0][0], points[0][1]-5, 
#                        f"{item['text']} ({item['confidence']:.2f})", 
#                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
#                        fontsize=8)
            
#             ax.set_title(f'Circuit OCR Results - Found {len(valid_detections)} valid text items', 
#                         fontsize=16, fontweight='bold')
#             ax.axis('off')
            
#             # Save with proper DPI and bounds
#             plt.tight_layout()
#             plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
#             plt.close()
            
#             print(f"    Visualization saved with {len(valid_detections)} valid detections out of {len(text_with_info)} total")
            
#         except Exception as e:
#             print(f"    Warning: Could not create visualization: {e}")
#             # Create a simple text visualization instead
#             with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
#                 f.write(f"Found {len(text_with_info)} text items:\n")
#                 for item in text_with_info:
#                     f.write(f"{item['text']} (conf: {item['confidence']:.3f})\n")

# # ===== OCR PROCESSING FUNCTION =====
# def process_circuit_ocr(image_path, output_folder, ocr_engine, components=None):
#     """Process OCR for circuit image and extract ALL text with confidence scores and proximity to components"""
#     print("\n  Running OCR text extraction...")
    
#     # Create OCR subfolder
#     ocr_output_folder = os.path.join(output_folder, "ocr_results")
#     os.makedirs(ocr_output_folder, exist_ok=True)
    
#     # Read image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"    Error: Could not read image {image_path}")
#         return []
    
#     # Extract all text with confidence and location
#     text_with_info, versions = ocr_engine.process_image(image_path)
    
#     # Save some preprocessing versions for debugging
#     for i, (name, img) in enumerate(list(versions.items())[:4]):
#         if img is not None:
#             cv2.imwrite(os.path.join(ocr_output_folder, f"ocr_preprocess_{name}.png"), img)
    
#     # Visualize
#     if text_with_info:
#         ocr_engine.visualize_results(image, text_with_info, 
#                                    os.path.join(ocr_output_folder, "ocr_detected_text.png"))
    
#     # Save results with confidence scores
#     with open(os.path.join(ocr_output_folder, "extracted_text_with_confidence.txt"), 'w', encoding='utf-8') as f:
#         if text_with_info:
#             for item in text_with_info:
#                 f.write(f"{item['text']}\t{item['confidence']:.3f}\n")
#             print(f"    Found {len(text_with_info)} text items")
#             # Show top 10 with confidence
#             print("    Top OCR detections:")
#             for item in text_with_info[:10]:
#                 print(f"      {item['text']} (confidence: {item['confidence']:.3f})")
#             if len(text_with_info) > 10:
#                 print(f"      ... and {len(text_with_info) - 10} more")
#         else:
#             f.write("No text detected")
#             print("    No text detected")
    
#     # Save just the text (without confidence) for compatibility
#     all_text = [item['text'] for item in text_with_info]
#     with open(os.path.join(ocr_output_folder, "extracted_text.txt"), 'w', encoding='utf-8') as f:
#         if all_text:
#             f.write('\n'.join(all_text))
    
#     # If components are provided, create proximity-based mapping
#     if components and text_with_info:
#         print("    Analyzing text proximity to components...")
        
#         # Filter text_with_info to only include items with center coordinates
#         text_with_locations = [item for item in text_with_info if item.get('center') is not None]
        
#         if not text_with_locations:
#             print("    Warning: No text items have location information for proximity analysis")
#             return text_with_info
        
#         component_text_mapping = assign_text_to_nearest_component(text_with_locations, components)
        
#         # Save proximity-based summary
#         with open(os.path.join(ocr_output_folder, "text_by_component_proximity.txt"), 'w', encoding='utf-8') as f:
#             f.write("OCR Text Assigned to Nearest Components\n")
#             f.write("="*60 + "\n")
#             f.write("Note: Each text appears only under its closest component\n")
#             f.write("="*60 + "\n\n")
            
#             for comp_id, text_items in component_text_mapping.items():
#                 # Find component info
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"Component: {comp_id} (Type: {comp_info['type']})\n")
#                     f.write(f"Bounding Box: {comp_info['bbox']}\n")
#                     f.write("-"*40 + "\n")
                    
#                     if text_items:
#                         f.write(f"Assigned Text ({len(text_items)} items, sorted by distance):\n")
#                         for i, item in enumerate(text_items):
#                             f.write(f"  {i+1}. {item['text']:<15} (conf: {item['confidence']:.3f}, dist: {item['distance']:.1f}px)\n")
#                     else:
#                         f.write("  No text assigned to this component\n")
#                     f.write("\n")
        
#         # Create visual proximity map showing assignments
#         proximity_vis = image.copy()
        
#         # Color map for components
#         colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
#         # Draw components and their assigned texts
#         for idx, (comp_id, text_items) in enumerate(component_text_mapping.items()):
#             comp_info = next((c for c in components if c['id'] == comp_id), None)
#             if comp_info and 'bbox' in comp_info:
#                 bbox = comp_info['bbox']
#                 if len(bbox) == 4:
#                     x_min, y_min, x_max, y_max = bbox
#                     color = colors[idx % len(colors)]
                    
#                     # Draw component box
#                     cv2.rectangle(proximity_vis, (x_min, y_min), (x_max, y_max), color, 2)
#                     cv2.putText(proximity_vis, comp_id, (x_min, y_min - 5),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
#                     # Draw lines to all assigned texts
#                     comp_center = get_component_center(comp_info['bbox'])
#                     for text_item in text_items:
#                         if text_item.get('center'):
#                             # Draw line from component to text
#                             cv2.line(proximity_vis, 
#                                     (int(comp_center[0]), int(comp_center[1])), 
#                                     (int(text_item['center'][0]), int(text_item['center'][1])), 
#                                     color, 1)
#                             # Draw text with same color as component
#                             cv2.putText(proximity_vis, text_item['text'], 
#                                        (int(text_item['center'][0]), int(text_item['center'][1]) - 5),
#                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(os.path.join(ocr_output_folder, "text_component_assignments.png"), proximity_vis)
        
#         # Also create a summary statistics file
#         with open(os.path.join(ocr_output_folder, "assignment_statistics.txt"), 'w', encoding='utf-8') as f:
#             f.write("Text Assignment Statistics\n")
#             f.write("="*40 + "\n\n")
            
#             total_assigned = sum(len(items) for items in component_text_mapping.values())
#             f.write(f"Total texts detected: {len(text_with_info)}\n")
#             f.write(f"Total texts with locations: {len(text_with_locations)}\n")
#             f.write(f"Total texts assigned to components: {total_assigned}\n")
#             f.write(f"Total components: {len(components)}\n\n")
            
#             f.write("Texts per component:\n")
#             for comp_id, text_items in component_text_mapping.items():
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"  {comp_id} ({comp_info.get('type', 'unknown')}): {len(text_items)} texts\n")
    
#     # Save detailed summary
#     with open(os.path.join(ocr_output_folder, "ocr_summary.txt"), 'w', encoding='utf-8') as f:
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total text items found: {len(text_with_info)}\n\n")
#         f.write("All extracted text with confidence scores:\n")
#         f.write("-" * 50 + "\n")
#         f.write("Text\t\t\tConfidence\n")
#         f.write("-" * 50 + "\n")
#         for item in text_with_info:
#             f.write(f"{item['text']:<20}\t{item['confidence']:.3f}\n")
    
#     return text_with_info



# # version 3
# import cv2
# import os
# from helper_files.proximity_analysis import assign_text_to_nearest_component, get_component_center
# import re
# import easyocr
# import pytesseract
# import matplotlib
# matplotlib.use('Agg') 
# import numpy as np

# # ===== OCR CLASS DEFINITION =====
# class CircuitOCR:
#     def __init__(self, use_gpu=True):
#         """Initialize OCR engines"""
#         print("Initializing OCR engines...")
        
#         # Initialize EasyOCR (this works with your current setup)
#         self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
         
#         # No filtering - capture all text
#         self.component_patterns = []  # Empty - we'll capture everything
#         self.combined_pattern = None  # No pattern matching
        
#         # Common OCR misreads - expanded list
#         self.ocr_corrections = {
#             'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8',
#             'G': '6', 'l': '1', 'o': '0', 'i': '1', 's': '5',
#             'D': '0', 'Q': '0', 'U': '0'  # Additional common misreads
#         }

#     def advanced_preprocessing(self, image):
#         """Create multiple preprocessing versions optimized for OCR"""
#         versions = {}
        
#         # Original
#         versions['original'] = image.copy()
        
#         # Grayscale
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.copy()
#         versions['gray'] = gray
        
#         # High contrast binary (Otsu's method)
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         versions['binary_otsu'] = binary
        
#         # Adaptive threshold (good for varying lighting)
#         adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY, 21, 10)
#         versions['adaptive'] = adaptive
        
#         # Inverted versions (for white text on dark background)
#         versions['inverted_gray'] = cv2.bitwise_not(gray)
#         versions['inverted_binary'] = cv2.bitwise_not(binary)
#         versions['inverted_adaptive'] = cv2.bitwise_not(adaptive)
        
#         # Enhanced contrast using CLAHE
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
#         versions['enhanced_contrast'] = enhanced
        
#         # Denoised version
#         denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#         versions['denoised'] = denoised
        
#         # Edge-preserved smoothing
#         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
#         versions['bilateral'] = bilateral
        
#         # Morphological operations to clean text
#         kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
#         versions['morph_close'] = morph_close
        
#         # Dilated version (makes text thicker)
#         kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#         dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
#         versions['dilated'] = dilated
        
#         # Super resolution attempt (resize)
#         height, width = gray.shape
#         if width < 2000:
#             scale = 2000 / width
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#             versions['upscaled'] = resized
        
#         return versions

#     def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7, offset_x=0, offset_y=0):
#         """Extract text using EasyOCR with confidence scores and bounding boxes"""
#         try:
#             # Get image dimensions for validation
#             if len(image.shape) == 3:
#                 img_height, img_width, _ = image.shape
#             else:
#                 img_height, img_width = image.shape
            
#             # Run EasyOCR with specific settings
#             results = self.easy_reader.readtext(
#                 image, 
#                 detail=detail,
#                 width_ths=width_ths,
#                 height_ths=height_ths,
#                 paragraph=False,
#                 text_threshold=0.3,
#                 low_text=0.3,
#                 mag_ratio=1.5
#             )
            
#             text_with_conf_bbox = []
            
#             if detail == 1:
#                 # Return tuples of (text, confidence, bbox)
#                 for result in results:
#                     if len(result) >= 3 and result[2] > 0.3:  # Increased confidence threshold
#                         bbox_points = result[0]
#                         text = str(result[1]).strip()
                        
#                         # Skip empty or very short text
#                         if not text or len(text) < 1:
#                             continue
                        
#                         # Calculate center point of bbox
#                         x_coords = [p[0] + offset_x for p in bbox_points]
#                         y_coords = [p[1] + offset_y for p in bbox_points]
                        
#                         # Validate coordinates are within image bounds
#                         if offset_x == 0 and offset_y == 0:  # Only validate for non-cropped images
#                             if any(x < 0 or x > img_width for x in x_coords):
#                                 continue
#                             if any(y < 0 or y > img_height for y in y_coords):
#                                 continue
                        
#                         center_x = sum(x_coords) / len(x_coords)
#                         center_y = sum(y_coords) / len(y_coords)
                        
#                         # Adjust bbox points with offset
#                         adjusted_bbox = [[p[0] + offset_x, p[1] + offset_y] for p in bbox_points]
                        
#                         text_with_conf_bbox.append({
#                             'text': text.upper(),
#                             'confidence': float(result[2]),
#                             'bbox': adjusted_bbox,
#                             'center': (center_x, center_y)
#                         })
#             else:
#                 # When detail=0, no bbox available
#                 text_with_conf_bbox = [{'text': str(text).upper(), 'confidence': 0.5, 'bbox': None, 'center': None} 
#                                       for text in results if text and str(text).strip()]
            
#             return text_with_conf_bbox
#         except Exception as e:
#             print(f"    EasyOCR error: {e}")
#             return []

#     def extract_text_with_tesseract(self, image):
#         """Extract text using Tesseract with confidence scores"""
#         text_with_conf = []
        
#         # Try getting individual words with confidence scores
#         try:
#             data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#             n_boxes = len(data['text'])
#             for i in range(n_boxes):
#                 if data['text'][i].strip():
#                     confidence = float(data['conf'][i])
#                     if confidence > 0:
#                         # Get bounding box
#                         x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#                         center_x = x + w/2
#                         center_y = y + h/2
#                         bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        
#                         text_with_conf.append({
#                             'text': data['text'][i].upper(),
#                             'confidence': confidence / 100.0,
#                             'bbox': bbox,
#                             'center': (center_x, center_y)
#                         })
#         except Exception as e:
#             print(f"    Tesseract data extraction error: {e}")
        
#         # Other configurations without confidence
#         configs = [
#             r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             r'--psm 8',
#             r'--psm 12',
#             r'--psm 13',
#         ]
        
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(image, config=config)
#                 if text.strip():
#                     words = text.upper().split()
#                     for word in words:
#                         word = word.strip()
#                         if word and not any(item['text'] == word for item in text_with_conf):
#                             text_with_conf.append({
#                                 'text': word,
#                                 'confidence': 0.5,
#                                 'bbox': None,
#                                 'center': None
#                             })
#             except:
#                 pass
            
#         return text_with_conf

#     def correct_ocr_errors(self, text):
#         """Correct common OCR misreads"""
#         corrected = text
        
#         # Only apply corrections if the text looks like a component identifier
#         if re.match(r'^[A-Z]+\d*$', text) or re.match(r'^\d+[A-Z]*$', text):
#             for wrong, right in self.ocr_corrections.items():
#                 corrected = corrected.replace(wrong, right)
        
#         return corrected

#     def extract_components_from_texts(self, text_with_conf_bbox_list):
#         """Extract ALL text with confidence scores and locations (no filtering)"""
#         # Combine all results
#         all_items = {}  # Use dict to track best result for each text
        
#         for item in text_with_conf_bbox_list:
#             if not isinstance(item, dict):
#                 continue
                
#             text = item.get('text', '')
#             conf = item.get('confidence', 0.0)
#             bbox = item.get('bbox', None)
#             center = item.get('center', None)
            
#             # Split by common separators and clean
#             items = re.split(r'[\s,;:\n\r\t]+', text)
#             for sub_item in items:
#                 sub_item = sub_item.strip()
#                 if not sub_item:
#                     continue
                    
#                 # Apply OCR corrections
#                 corrected = self.correct_ocr_errors(sub_item)
#                 if corrected and len(corrected) > 0:
#                     # Keep the highest confidence version with location info
#                     if corrected not in all_items or conf > all_items[corrected]['confidence']:
#                         all_items[corrected] = {
#                             'text': corrected,
#                             'confidence': conf,
#                             'bbox': bbox,
#                             'center': center
#                         }
        
#         # Convert to list sorted by confidence
#         return sorted(all_items.values(), key=lambda x: x['confidence'], reverse=True)

#     def process_image(self, image_path):
#         """Main processing function that returns text with confidence scores"""
#         # Read image
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"    Error: Could not read image {image_path}")
#             return [], {}
        
#         # Get all preprocessing versions
#         versions = self.advanced_preprocessing(image)
        
#         all_text_with_conf = []  # List of text info dicts
        
#         print("  Processing with multiple OCR engines and preprocessing...")
        
#         # Process each version
#         for version_name, processed_img in versions.items():
#             if processed_img is None:
#                 continue
                
#             # Convert grayscale to BGR for EasyOCR if needed
#             if len(processed_img.shape) == 2:
#                 img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 img_for_ocr = processed_img
            
#             # EasyOCR
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr)
#             all_text_with_conf.extend(text_conf_list)
            
#             # EasyOCR with different parameters
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
#             all_text_with_conf.extend(text_conf_list)
            
#             # Tesseract
#             text_conf_list = self.extract_text_with_tesseract(processed_img)
#             all_text_with_conf.extend(text_conf_list)
        
#         # Try cropping and processing regions
#         height, width = image.shape[:2]
#         quadrants = [
#             (0, 0, width//2, height//2),
#             (width//2, 0, width, height//2),
#             (0, height//2, width//2, height),
#             (width//2, height//2, width, height)
#         ]
        
#         for x1, y1, x2, y2 in quadrants:
#             cropped = image[y1:y2, x1:x2]
#             # Pass offset to adjust coordinates back to full image
#             text_conf_list = self.extract_text_with_easyocr(cropped, offset_x=x1, offset_y=y1)
#             all_text_with_conf.extend(text_conf_list)
        
#         # Process and deduplicate
#         final_results = self.extract_components_from_texts(all_text_with_conf)
        
#         return final_results, versions

#     def visualize_results(self, image, text_with_info, output_path):
#         """Create visualization of detected text"""
#         try:
#             # Set matplotlib to use non-interactive backend
#             import matplotlib
#             matplotlib.use('Agg')  # Use non-GUI backend
#             import matplotlib.pyplot as plt
#             import matplotlib.patches as patches
            
#             # Create figure with proper size matching the image aspect ratio
#             height, width = image.shape[:2]
#             fig_width = 15
#             fig_height = fig_width * height / width
            
#             fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
            
#             # Display the image
#             ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
#             # Set the axis limits to match image dimensions exactly
#             ax.set_xlim(0, width)
#             ax.set_ylim(height, 0)  # Inverted for image coordinates
            
#             # Filter out invalid detections
#             valid_detections = []
#             for item in text_with_info:
#                 if item.get('bbox') and item.get('confidence', 0) > 0.3:  # Increased threshold
#                     # Check if bbox is within image bounds
#                     bbox = item['bbox']
#                     if bbox and len(bbox) >= 4:
#                         x_coords = [p[0] for p in bbox]
#                         y_coords = [p[1] for p in bbox]
                        
#                         # Check if coordinates are valid
#                         if (all(0 <= x <= width for x in x_coords) and 
#                             all(0 <= y <= height for y in y_coords) and
#                             max(x_coords) - min(x_coords) > 5 and  # Minimum width
#                             max(y_coords) - min(y_coords) > 5):     # Minimum height
#                             valid_detections.append(item)
            
#             # Draw bounding boxes for valid detections only
#             for i, item in enumerate(valid_detections):
#                 points = item['bbox']
#                 rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=2)
#                 ax.add_patch(rect)
                
#                 # Add label
#                 ax.text(points[0][0], points[0][1]-5, 
#                        f"{item['text']} ({item['confidence']:.2f})", 
#                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
#                        fontsize=8)
            
#             ax.set_title(f'Circuit OCR Results - Found {len(valid_detections)} valid text items', 
#                         fontsize=16, fontweight='bold')
#             ax.axis('off')
            
#             # Save with proper DPI and bounds
#             plt.tight_layout()
#             plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
#             plt.close()
            
#             print(f"    Visualization saved with {len(valid_detections)} valid detections out of {len(text_with_info)} total")
            
#         except Exception as e:
#             print(f"    Warning: Could not create visualization: {e}")
#             # Create a simple text visualization instead
#             with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
#                 f.write(f"Found {len(text_with_info)} text items:\n")
#                 for item in text_with_info:
#                     f.write(f"{item['text']} (conf: {item['confidence']:.3f})\n")

# # ===== OCR PROCESSING FUNCTION =====
# def process_circuit_ocr(image_path, output_folder, ocr_engine, components=None):
#     """Process OCR for circuit image and extract ALL text with confidence scores and proximity to components"""
#     print("\n  Running OCR text extraction...")
    
#     # Create OCR subfolder
#     ocr_output_folder = os.path.join(output_folder, "ocr_results")
#     os.makedirs(ocr_output_folder, exist_ok=True)
    
#     # Read image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"    Error: Could not read image {image_path}")
#         return []
    
#     # Extract all text with confidence and location
#     text_with_info, versions = ocr_engine.process_image(image_path)
    
#     # Save some preprocessing versions for debugging
#     for i, (name, img) in enumerate(list(versions.items())[:4]):
#         if img is not None:
#             cv2.imwrite(os.path.join(ocr_output_folder, f"ocr_preprocess_{name}.png"), img)
    
#     # Visualize
#     if text_with_info:
#         ocr_engine.visualize_results(image, text_with_info, 
#                                    os.path.join(ocr_output_folder, "ocr_detected_text.png"))
    
#     # Save results with confidence scores
#     with open(os.path.join(ocr_output_folder, "extracted_text_with_confidence.txt"), 'w', encoding='utf-8') as f:
#         if text_with_info:
#             for item in text_with_info:
#                 f.write(f"{item['text']}\t{item['confidence']:.3f}\n")
#             print(f"    Found {len(text_with_info)} text items")
#             # Show top 10 with confidence
#             print("    Top OCR detections:")
#             for item in text_with_info[:10]:
#                 print(f"      {item['text']} (confidence: {item['confidence']:.3f})")
#             if len(text_with_info) > 10:
#                 print(f"      ... and {len(text_with_info) - 10} more")
#         else:
#             f.write("No text detected")
#             print("    No text detected")
    
#     # Save just the text (without confidence) for compatibility
#     all_text = [item['text'] for item in text_with_info]
#     with open(os.path.join(ocr_output_folder, "extracted_text.txt"), 'w', encoding='utf-8') as f:
#         if all_text:
#             f.write('\n'.join(all_text))
    
#     # If components are provided, create proximity-based mapping
#     if components and text_with_info:
#         print("    Analyzing text proximity to components...")
        
#         # Filter text_with_info to only include items with center coordinates
#         text_with_locations = [item for item in text_with_info if item.get('center') is not None]
        
#         if not text_with_locations:
#             print("    Warning: No text items have location information for proximity analysis")
#             return text_with_info
        
#         component_text_mapping = assign_text_to_nearest_component(text_with_locations, components)
        
#         # Save proximity-based summary
#         with open(os.path.join(ocr_output_folder, "text_by_component_proximity.txt"), 'w', encoding='utf-8') as f:
#             f.write("OCR Text Assigned to Nearest Components\n")
#             f.write("="*60 + "\n")
#             f.write("Note: Each text appears only under its closest component\n")
#             f.write("="*60 + "\n\n")
            
#             for comp_id, text_items in component_text_mapping.items():
#                 # Find component info
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"Component: {comp_id} (Type: {comp_info['type']})\n")
#                     f.write(f"Bounding Box: {comp_info['bbox']}\n")
#                     f.write("-"*40 + "\n")
                    
#                     if text_items:
#                         f.write(f"Assigned Text ({len(text_items)} items, sorted by distance):\n")
#                         for i, item in enumerate(text_items):
#                             f.write(f"  {i+1}. {item['text']:<15} (conf: {item['confidence']:.3f}, dist: {item['distance']:.1f}px)\n")
#                     else:
#                         f.write("  No text assigned to this component\n")
#                     f.write("\n")
        
#         # Create visual proximity map showing assignments
#         proximity_vis = image.copy()
        
#         # Color map for components
#         colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
#         # Draw components and their assigned texts
#         for idx, (comp_id, text_items) in enumerate(component_text_mapping.items()):
#             comp_info = next((c for c in components if c['id'] == comp_id), None)
#             if comp_info and 'bbox' in comp_info:
#                 bbox = comp_info['bbox']
#                 if len(bbox) == 4:
#                     x_min, y_min, x_max, y_max = bbox
#                     color = colors[idx % len(colors)]
                    
#                     # Draw component box
#                     cv2.rectangle(proximity_vis, (x_min, y_min), (x_max, y_max), color, 2)
#                     cv2.putText(proximity_vis, comp_id, (x_min, y_min - 5),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
#                     # Draw lines to all assigned texts
#                     comp_center = get_component_center(comp_info['bbox'])
#                     for text_item in text_items:
#                         if text_item.get('center'):
#                             # Draw line from component to text
#                             cv2.line(proximity_vis, 
#                                     (int(comp_center[0]), int(comp_center[1])), 
#                                     (int(text_item['center'][0]), int(text_item['center'][1])), 
#                                     color, 1)
#                             # Draw text with same color as component
#                             cv2.putText(proximity_vis, text_item['text'], 
#                                        (int(text_item['center'][0]), int(text_item['center'][1]) - 5),
#                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(os.path.join(ocr_output_folder, "text_component_assignments.png"), proximity_vis)
        
#         # Also create a summary statistics file
#         with open(os.path.join(ocr_output_folder, "assignment_statistics.txt"), 'w', encoding='utf-8') as f:
#             f.write("Text Assignment Statistics\n")
#             f.write("="*40 + "\n\n")
            
#             total_assigned = sum(len(items) for items in component_text_mapping.values())
#             f.write(f"Total texts detected: {len(text_with_info)}\n")
#             f.write(f"Total texts with locations: {len(text_with_locations)}\n")
#             f.write(f"Total texts assigned to components: {total_assigned}\n")
#             f.write(f"Total components: {len(components)}\n\n")
            
#             f.write("Texts per component:\n")
#             for comp_id, text_items in component_text_mapping.items():
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"  {comp_id} ({comp_info.get('type', 'unknown')}): {len(text_items)} texts\n")
    
#     # Save detailed summary
#     with open(os.path.join(ocr_output_folder, "ocr_summary.txt"), 'w', encoding='utf-8') as f:
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total text items found: {len(text_with_info)}\n\n")
#         f.write("All extracted text with confidence scores:\n")
#         f.write("-" * 50 + "\n")
#         f.write("Text\t\t\tConfidence\n")
#         f.write("-" * 50 + "\n")
#         for item in text_with_info:
#             f.write(f"{item['text']:<20}\t{item['confidence']:.3f}\n")
    
#     return text_with_info



# # version 2:


# import cv2
# import os
# from helper_files.proximity_analysis import assign_text_to_nearest_component, get_component_center
# import re
# import easyocr
# import pytesseract
# import matplotlib
# matplotlib.use('Agg') 
# import numpy as np

# # ===== OCR CLASS DEFINITION =====
# class CircuitOCR:
#     def __init__(self, use_gpu=True):
#         """Initialize OCR engines"""
#         print("Initializing OCR engines...")
        
#         # Initialize EasyOCR (this works with your current setup)
#         self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
         
#         # No filtering - capture all text
#         self.component_patterns = []  # Empty - we'll capture everything
#         self.combined_pattern = None  # No pattern matching
        
#         # Common OCR misreads - expanded list
#         self.ocr_corrections = {
#             'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8',
#             'G': '6', 'l': '1', 'o': '0', 'i': '1', 's': '5',
#             'D': '0', 'Q': '0', 'U': '0'  # Additional common misreads
#         }

#     def advanced_preprocessing(self, image):
#         """Create multiple preprocessing versions optimized for OCR"""
#         versions = {}
        
#         # Original
#         versions['original'] = image.copy()
        
#         # Grayscale
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.copy()
#         versions['gray'] = gray
        
#         # High contrast binary (Otsu's method)
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         versions['binary_otsu'] = binary
        
#         # Adaptive threshold (good for varying lighting)
#         adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY, 21, 10)
#         versions['adaptive'] = adaptive
        
#         # Inverted versions (for white text on dark background)
#         versions['inverted_gray'] = cv2.bitwise_not(gray)
#         versions['inverted_binary'] = cv2.bitwise_not(binary)
#         versions['inverted_adaptive'] = cv2.bitwise_not(adaptive)
        
#         # Enhanced contrast using CLAHE
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
#         versions['enhanced_contrast'] = enhanced
        
#         # Denoised version
#         denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#         versions['denoised'] = denoised
        
#         # Edge-preserved smoothing
#         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
#         versions['bilateral'] = bilateral
        
#         # Morphological operations to clean text
#         kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
#         versions['morph_close'] = morph_close
        
#         # Dilated version (makes text thicker)
#         kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#         dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
#         versions['dilated'] = dilated
        
#         # Super resolution attempt (resize)
#         height, width = gray.shape
#         if width < 2000:
#             scale = 2000 / width
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#             versions['upscaled'] = resized
        
#         return versions

#     def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7, offset_x=0, offset_y=0):
#         """Extract text using EasyOCR with confidence scores and bounding boxes"""
#         try:
#             # Run EasyOCR with specific settings
#             results = self.easy_reader.readtext(
#                 image, 
#                 detail=detail,
#                 width_ths=width_ths,
#                 height_ths=height_ths,
#                 paragraph=False,
#                 text_threshold=0.3,
#                 low_text=0.3,
#                 mag_ratio=1.5
#             )
            
#             text_with_conf_bbox = []
            
#             if detail == 1:
#                 # Return tuples of (text, confidence, bbox)
#                 for result in results:
#                     if len(result) >= 3 and result[2] > 0.2:  # confidence threshold
#                         bbox_points = result[0]
#                         # Calculate center point of bbox
#                         x_coords = [p[0] + offset_x for p in bbox_points]
#                         y_coords = [p[1] + offset_y for p in bbox_points]
#                         center_x = sum(x_coords) / len(x_coords)
#                         center_y = sum(y_coords) / len(y_coords)
                        
#                         # Adjust bbox points with offset
#                         adjusted_bbox = [[p[0] + offset_x, p[1] + offset_y] for p in bbox_points]
                        
#                         text_with_conf_bbox.append({
#                             'text': str(result[1]).upper(),
#                             'confidence': float(result[2]),
#                             'bbox': adjusted_bbox,
#                             'center': (center_x, center_y)
#                         })
#             else:
#                 # When detail=0, no bbox available
#                 text_with_conf_bbox = [{'text': str(text).upper(), 'confidence': 0.5, 'bbox': None, 'center': None} 
#                                       for text in results if text]
            
#             return text_with_conf_bbox
#         except Exception as e:
#             print(f"    EasyOCR error: {e}")
#             return []

#     def extract_text_with_tesseract(self, image):
#         """Extract text using Tesseract with confidence scores"""
#         text_with_conf = []
        
#         # Try getting individual words with confidence scores
#         try:
#             data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#             n_boxes = len(data['text'])
#             for i in range(n_boxes):
#                 if data['text'][i].strip():
#                     confidence = float(data['conf'][i])
#                     if confidence > 0:
#                         # Get bounding box
#                         x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#                         center_x = x + w/2
#                         center_y = y + h/2
#                         bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        
#                         text_with_conf.append({
#                             'text': data['text'][i].upper(),
#                             'confidence': confidence / 100.0,
#                             'bbox': bbox,
#                             'center': (center_x, center_y)
#                         })
#         except Exception as e:
#             print(f"    Tesseract data extraction error: {e}")
        
#         # Other configurations without confidence
#         configs = [
#             r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             r'--psm 8',
#             r'--psm 12',
#             r'--psm 13',
#         ]
        
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(image, config=config)
#                 if text.strip():
#                     words = text.upper().split()
#                     for word in words:
#                         word = word.strip()
#                         if word and not any(item['text'] == word for item in text_with_conf):
#                             text_with_conf.append({
#                                 'text': word,
#                                 'confidence': 0.5,
#                                 'bbox': None,
#                                 'center': None
#                             })
#             except:
#                 pass
            
#         return text_with_conf

#     def correct_ocr_errors(self, text):
#         """Correct common OCR misreads"""
#         corrected = text
        
#         # Only apply corrections if the text looks like a component identifier
#         if re.match(r'^[A-Z]+\d*$', text) or re.match(r'^\d+[A-Z]*$', text):
#             for wrong, right in self.ocr_corrections.items():
#                 corrected = corrected.replace(wrong, right)
        
#         return corrected

#     def extract_components_from_texts(self, text_with_conf_bbox_list):
#         """Extract ALL text with confidence scores and locations (no filtering)"""
#         # Combine all results
#         all_items = {}  # Use dict to track best result for each text
        
#         for item in text_with_conf_bbox_list:
#             if not isinstance(item, dict):
#                 continue
                
#             text = item.get('text', '')
#             conf = item.get('confidence', 0.0)
#             bbox = item.get('bbox', None)
#             center = item.get('center', None)
            
#             # Split by common separators and clean
#             items = re.split(r'[\s,;:\n\r\t]+', text)
#             for sub_item in items:
#                 sub_item = sub_item.strip()
#                 if not sub_item:
#                     continue
                    
#                 # Apply OCR corrections
#                 corrected = self.correct_ocr_errors(sub_item)
#                 if corrected and len(corrected) > 0:
#                     # Keep the highest confidence version with location info
#                     if corrected not in all_items or conf > all_items[corrected]['confidence']:
#                         all_items[corrected] = {
#                             'text': corrected,
#                             'confidence': conf,
#                             'bbox': bbox,
#                             'center': center
#                         }
        
#         # Convert to list sorted by confidence
#         return sorted(all_items.values(), key=lambda x: x['confidence'], reverse=True)

#     def process_image(self, image_path):
#         """Main processing function that returns text with confidence scores"""
#         # Read image
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"    Error: Could not read image {image_path}")
#             return [], {}
        
#         # Get all preprocessing versions
#         versions = self.advanced_preprocessing(image)
        
#         all_text_with_conf = []  # List of text info dicts
        
#         print("  Processing with multiple OCR engines and preprocessing...")
        
#         # Process each version
#         for version_name, processed_img in versions.items():
#             if processed_img is None:
#                 continue
                
#             # Convert grayscale to BGR for EasyOCR if needed
#             if len(processed_img.shape) == 2:
#                 img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 img_for_ocr = processed_img
            
#             # EasyOCR
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr)
#             all_text_with_conf.extend(text_conf_list)
            
#             # EasyOCR with different parameters
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
#             all_text_with_conf.extend(text_conf_list)
            
#             # Tesseract
#             text_conf_list = self.extract_text_with_tesseract(processed_img)
#             all_text_with_conf.extend(text_conf_list)
        
#         # Try cropping and processing regions
#         height, width = image.shape[:2]
#         quadrants = [
#             (0, 0, width//2, height//2),
#             (width//2, 0, width, height//2),
#             (0, height//2, width//2, height),
#             (width//2, height//2, width, height)
#         ]
        
#         for x1, y1, x2, y2 in quadrants:
#             cropped = image[y1:y2, x1:x2]
#             # Pass offset to adjust coordinates back to full image
#             text_conf_list = self.extract_text_with_easyocr(cropped, offset_x=x1, offset_y=y1)
#             all_text_with_conf.extend(text_conf_list)
        
#         # Process and deduplicate
#         final_results = self.extract_components_from_texts(all_text_with_conf)
        
#         return final_results, versions

#     def visualize_results(self, image, text_with_info, output_path):
#         """Create visualization of detected text"""
#         try:
#             # Set matplotlib to use non-interactive backend
#             import matplotlib
#             matplotlib.use('Agg')  # Use non-GUI backend
#             import matplotlib.pyplot as plt
#             import matplotlib.patches as patches
            
#             # Create figure with proper size matching the image aspect ratio
#             height, width = image.shape[:2]
#             fig_width = 15
#             fig_height = fig_width * height / width
            
#             fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
            
#             # Display the image
#             ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
#             # Set the axis limits to match image dimensions exactly
#             ax.set_xlim(0, width)
#             ax.set_ylim(height, 0)  # Inverted for image coordinates
            
#             # Filter out invalid detections
#             valid_detections = []
#             for item in text_with_info:
#                 if item.get('bbox') and item.get('confidence', 0) > 0.3:  # Increased threshold
#                     # Check if bbox is within image bounds
#                     bbox = item['bbox']
#                     if bbox and len(bbox) >= 4:
#                         x_coords = [p[0] for p in bbox]
#                         y_coords = [p[1] for p in bbox]
                        
#                         # Check if coordinates are valid
#                         if (all(0 <= x <= width for x in x_coords) and 
#                             all(0 <= y <= height for y in y_coords) and
#                             max(x_coords) - min(x_coords) > 5 and  # Minimum width
#                             max(y_coords) - min(y_coords) > 5):     # Minimum height
#                             valid_detections.append(item)
            
#             # Draw bounding boxes for valid detections only
#             for i, item in enumerate(valid_detections):
#                 points = item['bbox']
#                 rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=2)
#                 ax.add_patch(rect)
                
#                 # Add label
#                 ax.text(points[0][0], points[0][1]-5, 
#                        f"{item['text']} ({item['confidence']:.2f})", 
#                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
#                        fontsize=8)
            
#             ax.set_title(f'Circuit OCR Results - Found {len(valid_detections)} valid text items', 
#                         fontsize=16, fontweight='bold')
#             ax.axis('off')
            
#             # Save with proper DPI and bounds
#             plt.tight_layout()
#             plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
#             plt.close()
            
#             print(f"    Visualization saved with {len(valid_detections)} valid detections out of {len(text_with_info)} total")
            
#         except Exception as e:
#             print(f"    Warning: Could not create visualization: {e}")
#             # Create a simple text visualization instead
#             with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
#                 f.write(f"Found {len(text_with_info)} text items:\n")
#                 for item in text_with_info:
#                     f.write(f"{item['text']} (conf: {item['confidence']:.3f})\n")

# # ===== OCR PROCESSING FUNCTION =====
# def process_circuit_ocr(image_path, output_folder, ocr_engine, components=None):
#     """Process OCR for circuit image and extract ALL text with confidence scores and proximity to components"""
#     print("\n  Running OCR text extraction...")
    
#     # Create OCR subfolder
#     ocr_output_folder = os.path.join(output_folder, "ocr_results")
#     os.makedirs(ocr_output_folder, exist_ok=True)
    
#     # Read image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"    Error: Could not read image {image_path}")
#         return []
    
#     # Extract all text with confidence and location
#     text_with_info, versions = ocr_engine.process_image(image_path)
    
#     # Save some preprocessing versions for debugging
#     for i, (name, img) in enumerate(list(versions.items())[:4]):
#         if img is not None:
#             cv2.imwrite(os.path.join(ocr_output_folder, f"ocr_preprocess_{name}.png"), img)
    
#     # Visualize
#     if text_with_info:
#         ocr_engine.visualize_results(image, text_with_info, 
#                                    os.path.join(ocr_output_folder, "ocr_detected_text.png"))
    
#     # Save results with confidence scores
#     with open(os.path.join(ocr_output_folder, "extracted_text_with_confidence.txt"), 'w', encoding='utf-8') as f:
#         if text_with_info:
#             for item in text_with_info:
#                 f.write(f"{item['text']}\t{item['confidence']:.3f}\n")
#             print(f"    Found {len(text_with_info)} text items")
#             # Show top 10 with confidence
#             print("    Top OCR detections:")
#             for item in text_with_info[:10]:
#                 print(f"      {item['text']} (confidence: {item['confidence']:.3f})")
#             if len(text_with_info) > 10:
#                 print(f"      ... and {len(text_with_info) - 10} more")
#         else:
#             f.write("No text detected")
#             print("    No text detected")
    
#     # Save just the text (without confidence) for compatibility
#     all_text = [item['text'] for item in text_with_info]
#     with open(os.path.join(ocr_output_folder, "extracted_text.txt"), 'w', encoding='utf-8') as f:
#         if all_text:
#             f.write('\n'.join(all_text))
    
#     # If components are provided, create proximity-based mapping
#     if components and text_with_info:
#         print("    Analyzing text proximity to components...")
        
#         # Filter text_with_info to only include items with center coordinates
#         text_with_locations = [item for item in text_with_info if item.get('center') is not None]
        
#         if not text_with_locations:
#             print("    Warning: No text items have location information for proximity analysis")
#             return text_with_info
        
#         component_text_mapping = assign_text_to_nearest_component(text_with_locations, components)
        
#         # Save proximity-based summary
#         with open(os.path.join(ocr_output_folder, "text_by_component_proximity.txt"), 'w', encoding='utf-8') as f:
#             f.write("OCR Text Assigned to Nearest Components\n")
#             f.write("="*60 + "\n")
#             f.write("Note: Each text appears only under its closest component\n")
#             f.write("="*60 + "\n\n")
            
#             for comp_id, text_items in component_text_mapping.items():
#                 # Find component info
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"Component: {comp_id} (Type: {comp_info['type']})\n")
#                     f.write(f"Bounding Box: {comp_info['bbox']}\n")
#                     f.write("-"*40 + "\n")
                    
#                     if text_items:
#                         f.write(f"Assigned Text ({len(text_items)} items, sorted by distance):\n")
#                         for i, item in enumerate(text_items):
#                             f.write(f"  {i+1}. {item['text']:<15} (conf: {item['confidence']:.3f}, dist: {item['distance']:.1f}px)\n")
#                     else:
#                         f.write("  No text assigned to this component\n")
#                     f.write("\n")
        
#         # Create visual proximity map showing assignments
#         proximity_vis = image.copy()
        
#         # Color map for components
#         colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
#         # Draw components and their assigned texts
#         for idx, (comp_id, text_items) in enumerate(component_text_mapping.items()):
#             comp_info = next((c for c in components if c['id'] == comp_id), None)
#             if comp_info and 'bbox' in comp_info:
#                 bbox = comp_info['bbox']
#                 if len(bbox) == 4:
#                     x_min, y_min, x_max, y_max = bbox
#                     color = colors[idx % len(colors)]
                    
#                     # Draw component box
#                     cv2.rectangle(proximity_vis, (x_min, y_min), (x_max, y_max), color, 2)
#                     cv2.putText(proximity_vis, comp_id, (x_min, y_min - 5),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
#                     # Draw lines to all assigned texts
#                     comp_center = get_component_center(comp_info['bbox'])
#                     for text_item in text_items:
#                         if text_item.get('center'):
#                             # Draw line from component to text
#                             cv2.line(proximity_vis, 
#                                     (int(comp_center[0]), int(comp_center[1])), 
#                                     (int(text_item['center'][0]), int(text_item['center'][1])), 
#                                     color, 1)
#                             # Draw text with same color as component
#                             cv2.putText(proximity_vis, text_item['text'], 
#                                        (int(text_item['center'][0]), int(text_item['center'][1]) - 5),
#                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(os.path.join(ocr_output_folder, "text_component_assignments.png"), proximity_vis)
        
#         # Also create a summary statistics file
#         with open(os.path.join(ocr_output_folder, "assignment_statistics.txt"), 'w', encoding='utf-8') as f:
#             f.write("Text Assignment Statistics\n")
#             f.write("="*40 + "\n\n")
            
#             total_assigned = sum(len(items) for items in component_text_mapping.values())
#             f.write(f"Total texts detected: {len(text_with_info)}\n")
#             f.write(f"Total texts with locations: {len(text_with_locations)}\n")
#             f.write(f"Total texts assigned to components: {total_assigned}\n")
#             f.write(f"Total components: {len(components)}\n\n")
            
#             f.write("Texts per component:\n")
#             for comp_id, text_items in component_text_mapping.items():
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"  {comp_id} ({comp_info.get('type', 'unknown')}): {len(text_items)} texts\n")
    
#     # Save detailed summary
#     with open(os.path.join(ocr_output_folder, "ocr_summary.txt"), 'w', encoding='utf-8') as f:
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total text items found: {len(text_with_info)}\n\n")
#         f.write("All extracted text with confidence scores:\n")
#         f.write("-" * 50 + "\n")
#         f.write("Text\t\t\tConfidence\n")
#         f.write("-" * 50 + "\n")
#         for item in text_with_info:
#             f.write(f"{item['text']:<20}\t{item['confidence']:.3f}\n")
    
#     return text_with_info


# #  good version with bug
# import cv2
# import os
# from helper_files.proximity_analysis import assign_text_to_nearest_component, get_component_center
# import re
# import easyocr
# import pytesseract
# import matplotlib
# matplotlib.use('Agg') 
# import numpy as np

# # ===== OCR CLASS DEFINITION =====
# class CircuitOCR:
#     def __init__(self, use_gpu=True):
#         """Initialize OCR engines"""
#         print("Initializing OCR engines...")
        
#         # Initialize EasyOCR (this works with your current setup)
#         self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
         
#         # No filtering - capture all text
#         self.component_patterns = []  # Empty - we'll capture everything
#         self.combined_pattern = None  # No pattern matching
        
#         # Common OCR misreads - expanded list
#         self.ocr_corrections = {
#             'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8',
#             'G': '6', 'l': '1', 'o': '0', 'i': '1', 's': '5',
#             'D': '0', 'Q': '0', 'U': '0'  # Additional common misreads
#         }

#     def advanced_preprocessing(self, image):
#         """Create multiple preprocessing versions optimized for OCR"""
#         versions = {}
        
#         # Original
#         versions['original'] = image.copy()
        
#         # Grayscale
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.copy()
#         versions['gray'] = gray
        
#         # High contrast binary (Otsu's method)
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         versions['binary_otsu'] = binary
        
#         # Adaptive threshold (good for varying lighting)
#         adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY, 21, 10)
#         versions['adaptive'] = adaptive
        
#         # Inverted versions (for white text on dark background)
#         versions['inverted_gray'] = cv2.bitwise_not(gray)
#         versions['inverted_binary'] = cv2.bitwise_not(binary)
#         versions['inverted_adaptive'] = cv2.bitwise_not(adaptive)
        
#         # Enhanced contrast using CLAHE
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
#         versions['enhanced_contrast'] = enhanced
        
#         # Denoised version
#         denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#         versions['denoised'] = denoised
        
#         # Edge-preserved smoothing
#         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
#         versions['bilateral'] = bilateral
        
#         # Morphological operations to clean text
#         kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
#         versions['morph_close'] = morph_close
        
#         # Dilated version (makes text thicker)
#         kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#         dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
#         versions['dilated'] = dilated
        
#         # Super resolution attempt (resize)
#         height, width = gray.shape
#         if width < 2000:
#             scale = 2000 / width
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#             versions['upscaled'] = resized
        
#         return versions

#     def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7, offset_x=0, offset_y=0):
#         """Extract text using EasyOCR with confidence scores and bounding boxes"""
#         try:
#             # Run EasyOCR with specific settings
#             results = self.easy_reader.readtext(
#                 image, 
#                 detail=detail,
#                 width_ths=width_ths,
#                 height_ths=height_ths,
#                 paragraph=False,
#                 text_threshold=0.3,
#                 low_text=0.3,
#                 mag_ratio=1.5
#             )
            
#             text_with_conf_bbox = []
            
#             if detail == 1:
#                 # Return tuples of (text, confidence, bbox)
#                 for result in results:
#                     if len(result) >= 3 and result[2] > 0.2:  # confidence threshold
#                         bbox_points = result[0]
#                         text = str(result[1]).strip()
                        
#                         # Skip empty text
#                         if not text:
#                             continue
                        
#                         # Calculate center point of bbox
#                         x_coords = [p[0] + offset_x for p in bbox_points]
#                         y_coords = [p[1] + offset_y for p in bbox_points]
#                         center_x = sum(x_coords) / len(x_coords)
#                         center_y = sum(y_coords) / len(y_coords)
                        
#                         # Adjust bbox points with offset
#                         adjusted_bbox = [[p[0] + offset_x, p[1] + offset_y] for p in bbox_points]
                        
#                         text_with_conf_bbox.append({
#                             'text': text.upper(),
#                             'confidence': float(result[2]),
#                             'bbox': adjusted_bbox,
#                             'center': (center_x, center_y)
#                         })
#             else:
#                 # When detail=0, no bbox available
#                 text_with_conf_bbox = [{'text': str(text).upper(), 'confidence': 0.5, 'bbox': None, 'center': None} 
#                                       for text in results if text and str(text).strip()]
            
#             return text_with_conf_bbox
#         except Exception as e:
#             print(f"    EasyOCR error: {e}")
#             return []

#     def extract_text_with_tesseract(self, image):
#         """Extract text using Tesseract with confidence scores"""
#         text_with_conf = []
        
#         # Try getting individual words with confidence scores
#         try:
#             data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#             n_boxes = len(data['text'])
#             for i in range(n_boxes):
#                 if data['text'][i].strip():
#                     confidence = float(data['conf'][i])
#                     if confidence > 0:
#                         # Get bounding box
#                         x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#                         center_x = x + w/2
#                         center_y = y + h/2
#                         bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        
#                         text_with_conf.append({
#                             'text': data['text'][i].upper(),
#                             'confidence': confidence / 100.0,
#                             'bbox': bbox,
#                             'center': (center_x, center_y)
#                         })
#         except Exception as e:
#             print(f"    Tesseract data extraction error: {e}")
        
#         # Other configurations without confidence
#         configs = [
#             r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             r'--psm 8',
#             r'--psm 12',
#             r'--psm 13',
#         ]
        
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(image, config=config)
#                 if text.strip():
#                     words = text.upper().split()
#                     for word in words:
#                         word = word.strip()
#                         if word and not any(item['text'] == word for item in text_with_conf):
#                             text_with_conf.append({
#                                 'text': word,
#                                 'confidence': 0.5,
#                                 'bbox': None,
#                                 'center': None
#                             })
#             except:
#                 pass
            
#         return text_with_conf

#     def correct_ocr_errors(self, text):
#         """Correct common OCR misreads"""
#         corrected = text
#         for wrong, right in self.ocr_corrections.items():
#             corrected = corrected.replace(wrong, right)
#         return corrected

#     def extract_components_from_texts(self, text_with_conf_bbox_list):
#         """Extract ALL text with confidence scores and locations (no filtering)"""
#         # Combine all results
#         all_items = {}  # Use dict to track best result for each text
        
#         for item in text_with_conf_bbox_list:
#             if not isinstance(item, dict):
#                 continue
                
#             text = item.get('text', '')
#             conf = item.get('confidence', 0.0)
#             bbox = item.get('bbox', None)
#             center = item.get('center', None)
            
#             # Split by common separators and clean
#             items = re.split(r'[\s,;:\n\r\t]+', text)
#             for sub_item in items:
#                 sub_item = sub_item.strip()
#                 if not sub_item:
#                     continue
                    
#                 # Apply OCR corrections
#                 corrected = self.correct_ocr_errors(sub_item)
#                 if corrected and len(corrected) > 0:
#                     # Keep the highest confidence version with location info
#                     if corrected not in all_items or conf > all_items[corrected]['confidence']:
#                         all_items[corrected] = {
#                             'text': corrected,
#                             'confidence': conf,
#                             'bbox': bbox,
#                             'center': center
#                         }
        
#         # Convert to list sorted by confidence
#         return sorted(all_items.values(), key=lambda x: x['confidence'], reverse=True)

#     def process_image(self, image_path):
#         """Main processing function that returns text with confidence scores"""
#         # Read image
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"    Error: Could not read image {image_path}")
#             return [], {}
        
#         # Get all preprocessing versions
#         versions = self.advanced_preprocessing(image)
        
#         all_text_with_conf = []  # List of text info dicts
        
#         print("  Processing with multiple OCR engines and preprocessing...")
        
#         # Process each version
#         for version_name, processed_img in versions.items():
#             if processed_img is None:
#                 continue
                
#             # Convert grayscale to BGR for EasyOCR if needed
#             if len(processed_img.shape) == 2:
#                 img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 img_for_ocr = processed_img
            
#             # EasyOCR
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr)
#             all_text_with_conf.extend(text_conf_list)
            
#             # EasyOCR with different parameters
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
#             all_text_with_conf.extend(text_conf_list)
            
#             # Tesseract
#             text_conf_list = self.extract_text_with_tesseract(processed_img)
#             all_text_with_conf.extend(text_conf_list)
        
#         # Try cropping and processing regions
#         height, width = image.shape[:2]
#         quadrants = [
#             (0, 0, width//2, height//2),
#             (width//2, 0, width, height//2),
#             (0, height//2, width//2, height),
#             (width//2, height//2, width, height)
#         ]
        
#         for x1, y1, x2, y2 in quadrants:
#             cropped = image[y1:y2, x1:x2]
#             # Pass offset to adjust coordinates back to full image
#             text_conf_list = self.extract_text_with_easyocr(cropped, offset_x=x1, offset_y=y1)
#             all_text_with_conf.extend(text_conf_list)
        
#         # Process and deduplicate
#         final_results = self.extract_components_from_texts(all_text_with_conf)
        
#         return final_results, versions

#     def visualize_results(self, image, text_with_info, output_path):
#         """Create visualization of detected text"""
#         try:
#             # Set matplotlib to use non-interactive backend
#             import matplotlib
#             matplotlib.use('Agg')  # Use non-GUI backend
#             import matplotlib.pyplot as plt
#             import matplotlib.patches as patches
            
#             # Create figure with proper size matching the image aspect ratio
#             height, width = image.shape[:2]
#             fig_width = 15
#             fig_height = fig_width * height / width
            
#             fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
            
#             # Display the image
#             ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
#             # Set the axis limits to match image dimensions exactly
#             ax.set_xlim(0, width)
#             ax.set_ylim(height, 0)  # Inverted for image coordinates
            
#             # Draw bounding boxes for all detected text
#             for i, item in enumerate(text_with_info):
#                 if item.get('bbox') and item.get('confidence', 0) > 0.2:
#                     # Check if bbox is within image bounds
#                     bbox = item['bbox']
#                     if bbox and len(bbox) >= 4:
#                         x_coords = [p[0] for p in bbox]
#                         y_coords = [p[1] for p in bbox]
                        
#                         # Check if coordinates are valid
#                         if (all(0 <= x <= width for x in x_coords) and 
#                             all(0 <= y <= height for y in y_coords)):
#                             # Draw bounding box
#                             points = item['bbox']
#                             rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=2)
#                             ax.add_patch(rect)
                            
#                             # Add label
#                             ax.text(points[0][0], points[0][1]-5, 
#                                    f"{item['text']} ({item['confidence']:.2f})", 
#                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
#                                    fontsize=8)
            
#             ax.set_title(f'Circuit OCR Results - Found {len(text_with_info)} text items', 
#                         fontsize=16, fontweight='bold')
#             ax.axis('off')
            
#             # Save with proper DPI and bounds
#             plt.tight_layout()
#             plt.savefig(output_path, dpi=200, bbox_inches='tight')
#             plt.close()
            
#         except Exception as e:
#             print(f"    Warning: Could not create visualization: {e}")
#             # Create a simple text visualization instead
#             with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
#                 f.write(f"Found {len(text_with_info)} text items:\n")
#                 for item in text_with_info:
#                     f.write(f"{item['text']} (conf: {item['confidence']:.3f})\n")

# # ===== OCR PROCESSING FUNCTION =====
# def process_circuit_ocr(image_path, output_folder, ocr_engine, components=None):
#     """Process OCR for circuit image and extract ALL text with confidence scores and proximity to components"""
#     print("\n  Running OCR text extraction...")
    
#     # Create OCR subfolder
#     ocr_output_folder = os.path.join(output_folder, "ocr_results")
#     os.makedirs(ocr_output_folder, exist_ok=True)
    
#     # Read image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"    Error: Could not read image {image_path}")
#         return []
    
#     # Extract all text with confidence and location
#     text_with_info, versions = ocr_engine.process_image(image_path)
    
#     # Save some preprocessing versions for debugging
#     for i, (name, img) in enumerate(list(versions.items())[:4]):
#         if img is not None:
#             cv2.imwrite(os.path.join(ocr_output_folder, f"ocr_preprocess_{name}.png"), img)
    
#     # Visualize
#     if text_with_info:
#         ocr_engine.visualize_results(image, text_with_info, 
#                                    os.path.join(ocr_output_folder, "ocr_detected_text.png"))
    
#     # Save results with confidence scores
#     with open(os.path.join(ocr_output_folder, "extracted_text_with_confidence.txt"), 'w', encoding='utf-8') as f:
#         if text_with_info:
#             for item in text_with_info:
#                 f.write(f"{item['text']}\t{item['confidence']:.3f}\n")
#             print(f"    Found {len(text_with_info)} text items")
#             # Show top 10 with confidence
#             print("    Top OCR detections:")
#             for item in text_with_info[:10]:
#                 print(f"      {item['text']} (confidence: {item['confidence']:.3f})")
#             if len(text_with_info) > 10:
#                 print(f"      ... and {len(text_with_info) - 10} more")
#         else:
#             f.write("No text detected")
#             print("    No text detected")
    
#     # Save just the text (without confidence) for compatibility
#     all_text = [item['text'] for item in text_with_info]
#     with open(os.path.join(ocr_output_folder, "extracted_text.txt"), 'w', encoding='utf-8') as f:
#         if all_text:
#             f.write('\n'.join(all_text))
    
#     # If components are provided, create proximity-based mapping
#     if components and text_with_info:
#         print("    Analyzing text proximity to components...")
        
#         # Filter text_with_info to only include items with center coordinates
#         text_with_locations = [item for item in text_with_info if item.get('center') is not None]
        
#         if not text_with_locations:
#             print("    Warning: No text items have location information for proximity analysis")
#             return text_with_info
        
#         component_text_mapping = assign_text_to_nearest_component(text_with_locations, components)
        
#         # Save proximity-based summary
#         with open(os.path.join(ocr_output_folder, "text_by_component_proximity.txt"), 'w', encoding='utf-8') as f:
#             f.write("OCR Text Assigned to Nearest Components\n")
#             f.write("="*60 + "\n")
#             f.write("Note: Each text appears only under its closest component\n")
#             f.write("="*60 + "\n\n")
            
#             for comp_id, text_items in component_text_mapping.items():
#                 # Find component info
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"Component: {comp_id} (Type: {comp_info['type']})\n")
#                     f.write(f"Bounding Box: {comp_info['bbox']}\n")
#                     f.write("-"*40 + "\n")
                    
#                     if text_items:
#                         f.write(f"Assigned Text ({len(text_items)} items, sorted by distance):\n")
#                         for i, item in enumerate(text_items):
#                             f.write(f"  {i+1}. {item['text']:<15} (conf: {item['confidence']:.3f}, dist: {item['distance']:.1f}px)\n")
#                     else:
#                         f.write("  No text assigned to this component\n")
#                     f.write("\n")
        
#         # Create visual proximity map showing assignments
#         proximity_vis = image.copy()
        
#         # Color map for components
#         colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
#         # Draw components and their assigned texts
#         for idx, (comp_id, text_items) in enumerate(component_text_mapping.items()):
#             comp_info = next((c for c in components if c['id'] == comp_id), None)
#             if comp_info and 'bbox' in comp_info:
#                 bbox = comp_info['bbox']
#                 if len(bbox) == 4:
#                     x_min, y_min, x_max, y_max = bbox
#                     color = colors[idx % len(colors)]
                    
#                     # Draw component box
#                     cv2.rectangle(proximity_vis, (x_min, y_min), (x_max, y_max), color, 2)
#                     cv2.putText(proximity_vis, comp_id, (x_min, y_min - 5),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
#                     # Draw lines to all assigned texts
#                     comp_center = get_component_center(comp_info['bbox'])
#                     for text_item in text_items:
#                         if text_item.get('center'):
#                             # Draw line from component to text
#                             cv2.line(proximity_vis, 
#                                     (int(comp_center[0]), int(comp_center[1])), 
#                                     (int(text_item['center'][0]), int(text_item['center'][1])), 
#                                     color, 1)
#                             # Draw text with same color as component
#                             cv2.putText(proximity_vis, text_item['text'], 
#                                        (int(text_item['center'][0]), int(text_item['center'][1]) - 5),
#                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(os.path.join(ocr_output_folder, "text_component_assignments.png"), proximity_vis)
        
#         # Also create a summary statistics file
#         with open(os.path.join(ocr_output_folder, "assignment_statistics.txt"), 'w', encoding='utf-8') as f:
#             f.write("Text Assignment Statistics\n")
#             f.write("="*40 + "\n\n")
            
#             total_assigned = sum(len(items) for items in component_text_mapping.values())
#             f.write(f"Total texts detected: {len(text_with_info)}\n")
#             f.write(f"Total texts with locations: {len(text_with_locations)}\n")
#             f.write(f"Total texts assigned to components: {total_assigned}\n")
#             f.write(f"Total components: {len(components)}\n\n")
            
#             f.write("Texts per component:\n")
#             for comp_id, text_items in component_text_mapping.items():
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"  {comp_id} ({comp_info.get('type', 'unknown')}): {len(text_items)} texts\n")
    
#     # Save detailed summary
#     with open(os.path.join(ocr_output_folder, "ocr_summary.txt"), 'w', encoding='utf-8') as f:
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total text items found: {len(text_with_info)}\n\n")
#         f.write("All extracted text with confidence scores:\n")
#         f.write("-" * 50 + "\n")
#         f.write("Text\t\t\tConfidence\n")
#         f.write("-" * 50 + "\n")
#         for item in text_with_info:
#             f.write(f"{item['text']:<20}\t{item['confidence']:.3f}\n")
    
#     return text_with_info


# version 10



# import cv2
# import os
# from helper_files.proximity_analysis import assign_text_to_nearest_component, get_component_center
# import re
# import easyocr
# import pytesseract
# import matplotlib
# matplotlib.use('Agg') 
# import numpy as np

# # ===== OCR CLASS DEFINITION =====
# class CircuitOCR:
#     def __init__(self, use_gpu=True):
#         """Initialize OCR engines"""
#         print("Initializing OCR engines...")
        
#         # Initialize EasyOCR (this works with your current setup)
#         self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
         
#         # No filtering - capture all text
#         self.component_patterns = []  # Empty - we'll capture everything
#         self.combined_pattern = None  # No pattern matching
        
#         # Common OCR misreads - expanded list
#         self.ocr_corrections = {
#             'O': '0', 'I': '1', 'B': '8',
#             'G': '6', 'l': '1', 'o': '0', 'i': '1', 
#         }

#     def advanced_preprocessing(self, image):
#         """Create multiple preprocessing versions optimized for OCR"""
#         versions = {}
        
#         # Original
#         versions['original'] = image.copy()
        
#         # Grayscale
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.copy()
#         versions['gray'] = gray
        
#         # High contrast binary (Otsu's method)
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         versions['binary_otsu'] = binary
        
#         # Adaptive threshold (good for varying lighting)
#         adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY, 21, 10)
#         versions['adaptive'] = adaptive
        
#         # Inverted versions (for white text on dark background)
#         versions['inverted_gray'] = cv2.bitwise_not(gray)
#         versions['inverted_binary'] = cv2.bitwise_not(binary)
#         versions['inverted_adaptive'] = cv2.bitwise_not(adaptive)
        
#         # Enhanced contrast using CLAHE
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(gray)
#         versions['enhanced_contrast'] = enhanced
        
#         # Denoised version
#         denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#         versions['denoised'] = denoised
        
#         # Edge-preserved smoothing
#         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
#         versions['bilateral'] = bilateral
        
#         # Morphological operations to clean text
#         kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#         morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
#         versions['morph_close'] = morph_close
        
#         # Dilated version (makes text thicker)
#         kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#         dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
#         versions['dilated'] = dilated
        
#         # Super resolution attempt (resize)
#         height, width = gray.shape
#         if width < 2000:
#             scale = 2000 / width
#             new_width = int(width * scale)
#             new_height = int(height * scale)
#             resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#             versions['upscaled'] = resized
        
#         return versions

#     def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7, offset_x=0, offset_y=0):
#         """Extract text using EasyOCR with confidence scores and bounding boxes"""
#         try:
#             # Run EasyOCR with specific settings
#             results = self.easy_reader.readtext(
#                 image, 
#                 detail=detail,
#                 width_ths=width_ths,
#                 height_ths=height_ths,
#                 paragraph=False,
#                 text_threshold=0.3,
#                 low_text=0.3,
#                 mag_ratio=1.5
#             )
            
#             text_with_conf_bbox = []
            
#             if detail == 1:
#                 # Return tuples of (text, confidence, bbox)
#                 for result in results:
#                     if len(result) >= 3 and result[2] > 0.2:  # confidence threshold
#                         bbox_points = result[0]
#                         text = str(result[1]).strip()
                        
#                         # Skip empty text
#                         if not text:
#                             continue
                        
#                         # Calculate center point of bbox
#                         x_coords = [p[0] + offset_x for p in bbox_points]
#                         y_coords = [p[1] + offset_y for p in bbox_points]
#                         center_x = sum(x_coords) / len(x_coords)
#                         center_y = sum(y_coords) / len(y_coords)
                        
#                         # Adjust bbox points with offset
#                         adjusted_bbox = [[p[0] + offset_x, p[1] + offset_y] for p in bbox_points]
                        
#                         text_with_conf_bbox.append({
#                             'text': text.upper(),
#                             'confidence': float(result[2]),
#                             'bbox': adjusted_bbox,
#                             'center': (center_x, center_y)
#                         })
#             else:
#                 # When detail=0, no bbox available
#                 text_with_conf_bbox = [{'text': str(text).upper(), 'confidence': 0.5, 'bbox': None, 'center': None} 
#                                       for text in results if text and str(text).strip()]
            
#             return text_with_conf_bbox
#         except Exception as e:
#             print(f"    EasyOCR error: {e}")
#             return []

#     def extract_text_with_tesseract(self, image):
#         """Extract text using Tesseract with confidence scores"""
#         text_with_conf = []
        
#         # Try getting individual words with confidence scores
#         try:
#             data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#             n_boxes = len(data['text'])
#             for i in range(n_boxes):
#                 if data['text'][i].strip():
#                     confidence = float(data['conf'][i])
#                     if confidence > 0:
#                         # Get bounding box
#                         x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#                         center_x = x + w/2
#                         center_y = y + h/2
#                         bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        
#                         text_with_conf.append({
#                             'text': data['text'][i].upper(),
#                             'confidence': confidence / 100.0,
#                             'bbox': bbox,
#                             'center': (center_x, center_y)
#                         })
#         except Exception as e:
#             print(f"    Tesseract data extraction error: {e}")
        
#         # Other configurations without confidence
#         configs = [
#             r'--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             r'--psm 8',
#             r'--psm 12',
#             r'--psm 13',
#         ]
        
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(image, config=config)
#                 if text.strip():
#                     words = text.upper().split()
#                     for word in words:
#                         word = word.strip()
#                         if word and not any(item['text'] == word for item in text_with_conf):
#                             text_with_conf.append({
#                                 'text': word,
#                                 'confidence': 0.5,
#                                 'bbox': None,
#                                 'center': None
#                             })
#             except:
#                 pass
            
#         return text_with_conf

#     def correct_ocr_errors(self, text):
#         """Correct common OCR misreads"""
#         corrected = text
#         for wrong, right in self.ocr_corrections.items():
#             corrected = corrected.replace(wrong, right)
#         return corrected

#     def extract_components_from_texts(self, text_with_conf_bbox_list):
#         """Extract ALL text with confidence scores and locations (no filtering)"""
#         # Combine all results
#         all_items = {}  # Use dict to track best result for each text
        
#         for item in text_with_conf_bbox_list:
#             if not isinstance(item, dict):
#                 continue
                
#             text = item.get('text', '')
#             conf = item.get('confidence', 0.0)
#             bbox = item.get('bbox', None)
#             center = item.get('center', None)
            
#             # Split by common separators and clean
#             items = re.split(r'[\s,;:\n\r\t]+', text)
#             for sub_item in items:
#                 sub_item = sub_item.strip()
#                 if not sub_item:
#                     continue
                    
#                 # Apply OCR corrections
#                 corrected = self.correct_ocr_errors(sub_item)
#                 if corrected and len(corrected) > 0:
#                     # Keep the highest confidence version with location info
#                     if corrected not in all_items or conf > all_items[corrected]['confidence']:
#                         all_items[corrected] = {
#                             'text': corrected,
#                             'confidence': conf,
#                             'bbox': bbox,
#                             'center': center
#                         }
        
#         # Convert to list sorted by confidence
#         return sorted(all_items.values(), key=lambda x: x['confidence'], reverse=True)

#     def process_image(self, image_path):
#         """Main processing function that returns text with confidence scores"""
#         # Read image
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"    Error: Could not read image {image_path}")
#             return [], {}
        
#         # Get original image dimensions
#         orig_height, orig_width = image.shape[:2]
        
#         # Get all preprocessing versions
#         versions = self.advanced_preprocessing(image)
        
#         all_text_with_conf = []  # List of text info dicts
        
#         print("  Processing with multiple OCR engines and preprocessing...")
        
#         # Process each version
#         for version_name, processed_img in versions.items():
#             if processed_img is None:
#                 continue
                
#             # Convert grayscale to BGR for EasyOCR if needed
#             if len(processed_img.shape) == 2:
#                 img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
#             else:
#                 img_for_ocr = processed_img
            
#             # For upscaled images, we need to scale coordinates back
#             scale_factor = 1.0
#             if version_name == 'upscaled' and img_for_ocr.shape[1] != orig_width:
#                 scale_factor = orig_width / img_for_ocr.shape[1]
            
#             # EasyOCR
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr)
#             # Scale coordinates back if needed
#             if scale_factor != 1.0:
#                 for item in text_conf_list:
#                     if item.get('bbox'):
#                         item['bbox'] = [[p[0] * scale_factor, p[1] * scale_factor] for p in item['bbox']]
#                     if item.get('center'):
#                         item['center'] = (item['center'][0] * scale_factor, item['center'][1] * scale_factor)
#             all_text_with_conf.extend(text_conf_list)
            
#             # EasyOCR with different parameters
#             text_conf_list = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
#             # Scale coordinates back if needed
#             if scale_factor != 1.0:
#                 for item in text_conf_list:
#                     if item.get('bbox'):
#                         item['bbox'] = [[p[0] * scale_factor, p[1] * scale_factor] for p in item['bbox']]
#                     if item.get('center'):
#                         item['center'] = (item['center'][0] * scale_factor, item['center'][1] * scale_factor)
#             all_text_with_conf.extend(text_conf_list)
            
#             # Tesseract - only process if dimensions match original
#             if processed_img.shape[:2] == (orig_height, orig_width):
#                 text_conf_list = self.extract_text_with_tesseract(processed_img)
#                 all_text_with_conf.extend(text_conf_list)
        
#         # Try cropping and processing regions
#         height, width = image.shape[:2]
#         quadrants = [
#             (0, 0, width//2, height//2),
#             (width//2, 0, width, height//2),
#             (0, height//2, width//2, height),
#             (width//2, height//2, width, height)
#         ]
        
#         for x1, y1, x2, y2 in quadrants:
#             cropped = image[y1:y2, x1:x2]
#             # Pass offset to adjust coordinates back to full image
#             text_conf_list = self.extract_text_with_easyocr(cropped, offset_x=x1, offset_y=y1)
#             all_text_with_conf.extend(text_conf_list)
        
#         # Filter out only clearly invalid detections (way outside bounds)
#         filtered_text_with_conf = []
#         margin = 10  # Allow small margin for edge cases
        
#         for item in all_text_with_conf:
#             if not isinstance(item, dict):
#                 continue
            
#             # Check if the detection has bounds
#             if item.get('bbox') and item.get('center'):
#                 bbox = item['bbox']
#                 if len(bbox) >= 4:
#                     x_coords = [p[0] for p in bbox]
#                     y_coords = [p[1] for p in bbox]
                    
#                     # Only filter if clearly outside bounds with margin
#                     if (all(-margin <= x <= orig_width + margin for x in x_coords) and 
#                         all(-margin <= y <= orig_height + margin for y in y_coords)):
#                         # Clamp coordinates to image bounds
#                         clamped_bbox = []
#                         for p in bbox:
#                             clamped_x = max(0, min(p[0], orig_width))
#                             clamped_y = max(0, min(p[1], orig_height))
#                             clamped_bbox.append([clamped_x, clamped_y])
#                         item['bbox'] = clamped_bbox
                        
#                         # Update center
#                         x_coords = [p[0] for p in clamped_bbox]
#                         y_coords = [p[1] for p in clamped_bbox]
#                         item['center'] = (sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords))
                        
#                         filtered_text_with_conf.append(item)
#             else:
#                 # Keep items without bbox info
#                 filtered_text_with_conf.append(item)
        
#         # Process and deduplicate
#         final_results = self.extract_components_from_texts(filtered_text_with_conf)
        
#         return final_results, versions

#     def visualize_results(self, image, text_with_info, output_path):
#         """Create visualization of detected text"""
#         try:
#             # Set matplotlib to use non-interactive backend
#             import matplotlib
#             matplotlib.use('Agg')  # Use non-GUI backend
#             import matplotlib.pyplot as plt
#             import matplotlib.patches as patches
            
#             # Create figure with proper size matching the image aspect ratio
#             height, width = image.shape[:2]
#             fig_width = 15
#             fig_height = fig_width * height / width
            
#             fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
            
#             # Display the image
#             ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
#             # Set the axis limits to match image dimensions exactly
#             ax.set_xlim(0, width)
#             ax.set_ylim(height, 0)  # Inverted for image coordinates
            
#             # Draw bounding boxes for all detected text
#             for i, item in enumerate(text_with_info):
#                 if item.get('bbox') and item.get('confidence', 0) > 0.2:
#                     # Check if bbox is within image bounds
#                     bbox = item['bbox']
#                     if bbox and len(bbox) >= 4:
#                         x_coords = [p[0] for p in bbox]
#                         y_coords = [p[1] for p in bbox]
                        
#                         # Check if coordinates are valid
#                         if (all(0 <= x <= width for x in x_coords) and 
#                             all(0 <= y <= height for y in y_coords)):
#                             # Draw bounding box
#                             points = item['bbox']
#                             rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=2)
#                             ax.add_patch(rect)
                            
#                             # Add label
#                             ax.text(points[0][0], points[0][1]-5, 
#                                    f"{item['text']} ({item['confidence']:.2f})", 
#                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
#                                    fontsize=8)
            
#             ax.set_title(f'Circuit OCR Results - Found {len(text_with_info)} text items', 
#                         fontsize=16, fontweight='bold')
#             ax.axis('off')
            
#             # Save with proper DPI and bounds
#             plt.tight_layout()
#             plt.savefig(output_path, dpi=200, bbox_inches='tight')
#             plt.close()
            
#         except Exception as e:
#             print(f"    Warning: Could not create visualization: {e}")
#             # Create a simple text visualization instead
#             with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
#                 f.write(f"Found {len(text_with_info)} text items:\n")
#                 for item in text_with_info:
#                     f.write(f"{item['text']} (conf: {item['confidence']:.3f})\n")

# # ===== OCR PROCESSING FUNCTION =====
# def process_circuit_ocr(image_path, output_folder, ocr_engine, components=None):
#     """Process OCR for circuit image and extract ALL text with confidence scores and proximity to components"""
#     print("\n  Running OCR text extraction...")
    
#     # Create OCR subfolder
#     ocr_output_folder = os.path.join(output_folder, "ocr_results")
#     os.makedirs(ocr_output_folder, exist_ok=True)
    
#     # Read image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"    Error: Could not read image {image_path}")
#         return []
    
#     # Extract all text with confidence and location
#     text_with_info, versions = ocr_engine.process_image(image_path)
    
#     # Save some preprocessing versions for debugging
#     for i, (name, img) in enumerate(list(versions.items())[:4]):
#         if img is not None:
#             cv2.imwrite(os.path.join(ocr_output_folder, f"ocr_preprocess_{name}.png"), img)
    
#     # Visualize
#     if text_with_info:
#         ocr_engine.visualize_results(image, text_with_info, 
#                                    os.path.join(ocr_output_folder, "ocr_detected_text.png"))
    
#     # Save results with confidence scores
#     with open(os.path.join(ocr_output_folder, "extracted_text_with_confidence.txt"), 'w', encoding='utf-8') as f:
#         if text_with_info:
#             for item in text_with_info:
#                 f.write(f"{item['text']}\t{item['confidence']:.3f}\n")
#             print(f"    Found {len(text_with_info)} text items")
#             # Show top 10 with confidence
#             print("    Top OCR detections:")
#             for item in text_with_info[:10]:
#                 print(f"      {item['text']} (confidence: {item['confidence']:.3f})")
#             if len(text_with_info) > 10:
#                 print(f"      ... and {len(text_with_info) - 10} more")
#         else:
#             f.write("No text detected")
#             print("    No text detected")
    
#     # Save just the text (without confidence) for compatibility
#     all_text = [item['text'] for item in text_with_info]
#     with open(os.path.join(ocr_output_folder, "extracted_text.txt"), 'w', encoding='utf-8') as f:
#         if all_text:
#             f.write('\n'.join(all_text))
    
#     # If components are provided, create proximity-based mapping
#     if components and text_with_info:
#         print("    Analyzing text proximity to components...")
        
#         # Filter text_with_info to only include items with center coordinates
#         text_with_locations = [item for item in text_with_info if item.get('center') is not None]
        
#         if not text_with_locations:
#             print("    Warning: No text items have location information for proximity analysis")
#             return text_with_info
        
#         component_text_mapping = assign_text_to_nearest_component(text_with_locations, components)
        
#         # Save proximity-based summary
#         with open(os.path.join(ocr_output_folder, "text_by_component_proximity.txt"), 'w', encoding='utf-8') as f:
#             f.write("OCR Text Assigned to Nearest Components\n")
#             f.write("="*60 + "\n")
#             f.write("Note: Each text appears only under its closest component\n")
#             f.write("="*60 + "\n\n")
            
#             for comp_id, text_items in component_text_mapping.items():
#                 # Find component info
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"Component: {comp_id} (Type: {comp_info['type']})\n")
#                     f.write(f"Bounding Box: {comp_info['bbox']}\n")
#                     f.write("-"*40 + "\n")
                    
#                     if text_items:
#                         f.write(f"Assigned Text ({len(text_items)} items, sorted by distance):\n")
#                         for i, item in enumerate(text_items):
#                             f.write(f"  {i+1}. {item['text']:<15} (conf: {item['confidence']:.3f}, dist: {item['distance']:.1f}px)\n")
#                     else:
#                         f.write("  No text assigned to this component\n")
#                     f.write("\n")
        
#         # Create visual proximity map showing assignments
#         proximity_vis = image.copy()
        
#         # Color map for components
#         colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
#         # Draw components and their assigned texts
#         for idx, (comp_id, text_items) in enumerate(component_text_mapping.items()):
#             comp_info = next((c for c in components if c['id'] == comp_id), None)
#             if comp_info and 'bbox' in comp_info:
#                 bbox = comp_info['bbox']
#                 if len(bbox) == 4:
#                     x_min, y_min, x_max, y_max = bbox
#                     color = colors[idx % len(colors)]
                    
#                     # Draw component box
#                     cv2.rectangle(proximity_vis, (x_min, y_min), (x_max, y_max), color, 2)
#                     cv2.putText(proximity_vis, comp_id, (x_min, y_min - 5),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
#                     # Draw lines to all assigned texts
#                     comp_center = get_component_center(comp_info['bbox'])
#                     for text_item in text_items:
#                         if text_item.get('center'):
#                             # Draw line from component to text
#                             cv2.line(proximity_vis, 
#                                     (int(comp_center[0]), int(comp_center[1])), 
#                                     (int(text_item['center'][0]), int(text_item['center'][1])), 
#                                     color, 1)
#                             # Draw text with same color as component
#                             cv2.putText(proximity_vis, text_item['text'], 
#                                        (int(text_item['center'][0]), int(text_item['center'][1]) - 5),
#                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
#         cv2.imwrite(os.path.join(ocr_output_folder, "text_component_assignments.png"), proximity_vis)
        
#         # Also create a summary statistics file
#         with open(os.path.join(ocr_output_folder, "assignment_statistics.txt"), 'w', encoding='utf-8') as f:
#             f.write("Text Assignment Statistics\n")
#             f.write("="*40 + "\n\n")
            
#             total_assigned = sum(len(items) for items in component_text_mapping.values())
#             f.write(f"Total texts detected: {len(text_with_info)}\n")
#             f.write(f"Total texts with locations: {len(text_with_locations)}\n")
#             f.write(f"Total texts assigned to components: {total_assigned}\n")
#             f.write(f"Total components: {len(components)}\n\n")
            
#             f.write("Texts per component:\n")
#             for comp_id, text_items in component_text_mapping.items():
#                 comp_info = next((c for c in components if c['id'] == comp_id), None)
#                 if comp_info:
#                     f.write(f"  {comp_id} ({comp_info.get('type', 'unknown')}): {len(text_items)} texts\n")
    
#     # Save detailed summary
#     with open(os.path.join(ocr_output_folder, "ocr_summary.txt"), 'w', encoding='utf-8') as f:
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total text items found: {len(text_with_info)}\n\n")
#         f.write("All extracted text with confidence scores:\n")
#         f.write("-" * 50 + "\n")
#         f.write("Text\t\t\tConfidence\n")
#         f.write("-" * 50 + "\n")
#         for item in text_with_info:
#             f.write(f"{item['text']:<20}\t{item['confidence']:.3f}\n")
    
#     return text_with_info





import cv2
import os
from helper_files.proximity_analysis import assign_text_to_nearest_component, get_component_center
import re
import easyocr
import pytesseract
import matplotlib
matplotlib.use('Agg') 
import numpy as np

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
        
        # Common OCR misreads - expanded list
        self.ocr_corrections = {
            'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8',
            'G': '6', 'l': '1', 'o': '0', 'i': '1', 's': '5',
            'D': '0', 'Q': '0', 'U': '0'  # Additional common misreads
        }

    def advanced_preprocessing(self, image):
        """Create multiple preprocessing versions optimized for OCR"""
        versions = {}
        
        # Original
        versions['original'] = image.copy()
        
        # Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
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

    def extract_text_with_easyocr(self, image, detail=1, width_ths=0.7, height_ths=0.7, offset_x=0, offset_y=0):
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
            
            text_with_conf_bbox = []
            
            if detail == 1:
                # Return tuples of (text, confidence, bbox)
                for result in results:
                    if len(result) >= 3 and result[2] > 0.2:  # confidence threshold
                        bbox_points = result[0]
                        text = str(result[1]).strip()
                        
                        # Skip empty text
                        if not text:
                            continue
                        
                        # Calculate center point of bbox
                        x_coords = [p[0] + offset_x for p in bbox_points]
                        y_coords = [p[1] + offset_y for p in bbox_points]
                        center_x = sum(x_coords) / len(x_coords)
                        center_y = sum(y_coords) / len(y_coords)
                        
                        # Adjust bbox points with offset
                        adjusted_bbox = [[p[0] + offset_x, p[1] + offset_y] for p in bbox_points]
                        
                        text_with_conf_bbox.append({
                            'text': text.upper(),
                            'confidence': float(result[2]),
                            'bbox': adjusted_bbox,
                            'center': (center_x, center_y)
                        })
            else:
                # When detail=0, no bbox available
                text_with_conf_bbox = [{'text': str(text).upper(), 'confidence': 0.5, 'bbox': None, 'center': None} 
                                      for text in results if text and str(text).strip()]
            
            return text_with_conf_bbox
        except Exception as e:
            print(f"    EasyOCR error: {e}")
            return []

    def extract_text_with_tesseract(self, image, scale_factor=1.0):
        """Extract text using Tesseract with confidence scores"""
        text_with_conf = []
        
        # Try getting individual words with confidence scores
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if data['text'][i].strip():
                    confidence = float(data['conf'][i])
                    if confidence > 0:
                        # Get bounding box
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        
                        # Apply scale factor if image was scaled
                        x = x * scale_factor
                        y = y * scale_factor
                        w = w * scale_factor
                        h = h * scale_factor
                        
                        center_x = x + w/2
                        center_y = y + h/2
                        bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        
                        text_with_conf.append({
                            'text': data['text'][i].upper(),
                            'confidence': confidence / 100.0,
                            'bbox': bbox,
                            'center': (center_x, center_y)
                        })
        except Exception as e:
            print(f"    Tesseract data extraction error: {e}")
        
        # Other configurations without confidence
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
                    words = text.upper().split()
                    for word in words:
                        word = word.strip()
                        if word and not any(item['text'] == word for item in text_with_conf):
                            text_with_conf.append({
                                'text': word,
                                'confidence': 0.5,
                                'bbox': None,
                                'center': None
                            })
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
            if not isinstance(item, dict):
                continue
                
            text = item.get('text', '')
            conf = item.get('confidence', 0.0)
            bbox = item.get('bbox', None)
            center = item.get('center', None)
            
            # Split by common separators and clean
            items = re.split(r'[\s,;:\n\r\t]+', text)
            for sub_item in items:
                sub_item = sub_item.strip()
                if not sub_item:
                    continue
                    
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
        if image is None:
            print(f"    Error: Could not read image {image_path}")
            return [], {}
        
        # Get original image dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Get all preprocessing versions
        versions = self.advanced_preprocessing(image)
        
        all_text_with_conf = []  # List of text info dicts
        
        print("  Processing with multiple OCR engines and preprocessing...")
        
        # Process each version
        for version_name, processed_img in versions.items():
            if processed_img is None:
                continue
                
            # Convert grayscale to BGR for EasyOCR if needed
            if len(processed_img.shape) == 2:
                img_for_ocr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
            else:
                img_for_ocr = processed_img
            
            # Calculate scale factor for this version
            scale_factor = 1.0
            if processed_img.shape[1] != orig_width:
                scale_factor = orig_width / processed_img.shape[1]
            
            # EasyOCR
            text_conf_list = self.extract_text_with_easyocr(img_for_ocr)
            # Scale coordinates back if needed
            if scale_factor != 1.0:
                for item in text_conf_list:
                    if item.get('bbox'):
                        item['bbox'] = [[p[0] * scale_factor, p[1] * scale_factor] for p in item['bbox']]
                    if item.get('center'):
                        item['center'] = (item['center'][0] * scale_factor, item['center'][1] * scale_factor)
            all_text_with_conf.extend(text_conf_list)
            
            # EasyOCR with different parameters
            text_conf_list = self.extract_text_with_easyocr(img_for_ocr, width_ths=0.5, height_ths=0.5)
            # Scale coordinates back if needed
            if scale_factor != 1.0:
                for item in text_conf_list:
                    if item.get('bbox'):
                        item['bbox'] = [[p[0] * scale_factor, p[1] * scale_factor] for p in item['bbox']]
                    if item.get('center'):
                        item['center'] = (item['center'][0] * scale_factor, item['center'][1] * scale_factor)
            all_text_with_conf.extend(text_conf_list)
            
            # Tesseract - pass scale factor
            text_conf_list = self.extract_text_with_tesseract(processed_img, scale_factor)
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
            # Pass offset to adjust coordinates back to full image
            text_conf_list = self.extract_text_with_easyocr(cropped, offset_x=x1, offset_y=y1)
            all_text_with_conf.extend(text_conf_list)
        
        # Filter out only clearly invalid detections
        filtered_text_with_conf = []
        margin = 5  # Small margin for edge cases
        
        for item in all_text_with_conf:
            if not isinstance(item, dict):
                continue
            
            # Keep all items without bbox (text-only detections)
            if not item.get('bbox') or not item.get('center'):
                filtered_text_with_conf.append(item)
                continue
            
            # For items with bbox, check if they're reasonable
            bbox = item['bbox']
            if len(bbox) >= 4:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                
                # Check if at least partially within bounds
                if (any(-margin <= x <= orig_width + margin for x in x_coords) and 
                    any(-margin <= y <= orig_height + margin for y in y_coords)):
                    # Clamp coordinates to image bounds
                    clamped_bbox = []
                    for p in bbox:
                        clamped_x = max(0, min(p[0], orig_width - 1))
                        clamped_y = max(0, min(p[1], orig_height - 1))
                        clamped_bbox.append([clamped_x, clamped_y])
                    item['bbox'] = clamped_bbox
                    
                    # Update center
                    x_coords = [p[0] for p in clamped_bbox]
                    y_coords = [p[1] for p in clamped_bbox]
                    item['center'] = (sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords))
                    
                    filtered_text_with_conf.append(item)
        
        # Process and deduplicate
        final_results = self.extract_components_from_texts(filtered_text_with_conf)
        
        return final_results, versions

    def visualize_results(self, image, text_with_info, output_path):
        """Create visualization of detected text"""
        try:
            # Set matplotlib to use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')  # Use non-GUI backend
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # Create figure with proper size matching the image aspect ratio
            height, width = image.shape[:2]
            fig_width = 15
            fig_height = fig_width * height / width
            
            fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
            
            # Display the image
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Set the axis limits to match image dimensions exactly
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)  # Inverted for image coordinates
            
            # Draw bounding boxes for all detected text
            for i, item in enumerate(text_with_info):
                if item.get('bbox') and item.get('confidence', 0) > 0.2:
                    # Check if bbox is within image bounds
                    bbox = item['bbox']
                    if bbox and len(bbox) >= 4:
                        x_coords = [p[0] for p in bbox]
                        y_coords = [p[1] for p in bbox]
                        
                        # Check if coordinates are valid
                        if (all(0 <= x <= width for x in x_coords) and 
                            all(0 <= y <= height for y in y_coords)):
                            # Draw bounding box
                            points = item['bbox']
                            rect = patches.Polygon(points, fill=False, edgecolor='lime', linewidth=2)
                            ax.add_patch(rect)
                            
                            # Add label
                            ax.text(points[0][0], points[0][1]-5, 
                                   f"{item['text']} ({item['confidence']:.2f})", 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                                   fontsize=8)
            
            ax.set_title(f'Circuit OCR Results - Found {len(text_with_info)} text items', 
                        fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # Save with proper DPI and bounds
            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    Warning: Could not create visualization: {e}")
            # Create a simple text visualization instead
            with open(output_path.replace('.png', '_visualization.txt'), 'w') as f:
                f.write(f"Found {len(text_with_info)} text items:\n")
                for item in text_with_info:
                    f.write(f"{item['text']} (conf: {item['confidence']:.3f})\n")

# ===== OCR PROCESSING FUNCTION =====
def process_circuit_ocr(image_path, output_folder, ocr_engine, components=None):
    """Process OCR for circuit image and extract ALL text with confidence scores and proximity to components"""
    print("\n  Running OCR text extraction...")
    
    # Create OCR subfolder
    ocr_output_folder = os.path.join(output_folder, "ocr_results")
    os.makedirs(ocr_output_folder, exist_ok=True)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"    Error: Could not read image {image_path}")
        return []
    
    # Extract all text with confidence and location
    text_with_info, versions = ocr_engine.process_image(image_path)
    
    # Save some preprocessing versions for debugging
    for i, (name, img) in enumerate(list(versions.items())[:4]):
        if img is not None:
            cv2.imwrite(os.path.join(ocr_output_folder, f"ocr_preprocess_{name}.png"), img)
    
    # Visualize
    if text_with_info:
        ocr_engine.visualize_results(image, text_with_info, 
                                   os.path.join(ocr_output_folder, "ocr_detected_text.png"))
    
    # Print summary to console
    if text_with_info:
        print(f"    Found {len(text_with_info)} text items")
        # Show top 10 with confidence
        print("    Top OCR detections:")
        for item in text_with_info[:10]:
            print(f"      {item['text']} (confidence: {item['confidence']:.3f})")
        if len(text_with_info) > 10:
            print(f"      ... and {len(text_with_info) - 10} more")
    else:
        print("    No text detected")
    
    # If components are provided, create proximity-based mapping
    if components and text_with_info:
        print("    Analyzing text proximity to components...")
        
        # Filter text_with_info to only include items with center coordinates
        text_with_locations = [item for item in text_with_info if item.get('center') is not None]
        
        if not text_with_locations:
            print("    Warning: No text items have location information for proximity analysis")
            return text_with_info
        
        component_text_mapping = assign_text_to_nearest_component(text_with_locations, components)
        
        # Create the consolidated LLM-ready output file
        with open(os.path.join(ocr_output_folder, "circuit_components_with_text.txt"), 'w', encoding='utf-8') as f:
            f.write("CIRCUIT COMPONENT ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write("This file contains all detected circuit components with their nearby text.\n")
            f.write("The text near each component may include its label, value, or other identifiers.\n")
            f.write("="*80 + "\n\n")
            
            # First, list all detected text for context
            f.write("ALL DETECTED TEXT IN CIRCUIT:\n")
            f.write("-"*40 + "\n")
            for item in text_with_info:
                f.write(f"- {item['text']} (confidence: {item['confidence']:.2f})\n")
            f.write("\n")
            
            # Now list each component with its associated text
            f.write("COMPONENTS WITH ASSOCIATED TEXT:\n")
            f.write("="*80 + "\n\n")
            
            for comp_info in components:
                comp_id = comp_info['id']
                comp_type = comp_info.get('type', 'unknown')
                bbox = comp_info.get('bbox', [])
                
                f.write(f"COMPONENT {comp_id}:\n")
                f.write(f"  Type: {comp_type}\n")
                if bbox and len(bbox) == 4:
                    f.write(f"  Location: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})\n")
                
                # Get associated text for this component
                text_items = component_text_mapping.get(comp_id, [])
                
                if text_items:
                    f.write(f"  Nearby Text (sorted by proximity):\n")
                    
                    # Categorize text as potential labels or values
                    potential_labels = []
                    potential_values = []
                    
                    for item in text_items:
                        text = item['text']
                        confidence = item['confidence']
                        distance = item['distance']
                        
                        # Simple heuristic to categorize text
                        # Labels often start with letters (R, C, L, Q, D, etc.)
                        # Values often contain numbers with units (10k, 2.2uF, etc.)
                        if re.match(r'^[A-Z]+\d*$', text):  # Like R1, C2, Q1
                            potential_labels.append((text, confidence, distance))
                        elif re.search(r'\d', text):  # Contains numbers
                            potential_values.append((text, confidence, distance))
                        else:
                            # Could be either
                            potential_labels.append((text, confidence, distance))
                    
                    if potential_labels:
                        f.write("    Potential Labels:\n")
                        for text, conf, dist in potential_labels[:3]:  # Top 3
                            f.write(f"      - {text} (conf: {conf:.2f}, dist: {dist:.0f}px)\n")
                    
                    if potential_values:
                        f.write("    Potential Values:\n")
                        for text, conf, dist in potential_values[:3]:  # Top 3
                            f.write(f"      - {text} (conf: {conf:.2f}, dist: {dist:.0f}px)\n")
                    
                    # Also list all nearby text
                    f.write("    All Nearby Text:\n")
                    for item in text_items[:5]:  # Limit to top 5 closest
                        f.write(f"      - {item['text']} (conf: {item['confidence']:.2f}, dist: {item['distance']:.0f}px)\n")
                else:
                    f.write("  No text detected near this component\n")
                
                f.write("\n" + "-"*60 + "\n\n")
            
            # Add a summary section
            f.write("SUMMARY STATISTICS:\n")
            f.write("="*80 + "\n")
            total_assigned = sum(len(items) for items in component_text_mapping.values())
            f.write(f"Total components detected: {len(components)}\n")
            f.write(f"Total text items detected: {len(text_with_info)}\n")
            f.write(f"Total text items with location data: {len(text_with_locations)}\n")
            f.write(f"Total text items assigned to components: {total_assigned}\n")
            f.write(f"Components with text: {sum(1 for items in component_text_mapping.values() if items)}\n")
            f.write(f"Components without text: {sum(1 for items in component_text_mapping.values() if not items)}\n")
            
            # Add instructions for LLM
            f.write("\n" + "="*80 + "\n")
            f.write("INSTRUCTIONS FOR ANALYSIS:\n")
            f.write("1. For each component, identify the most likely label (e.g., R1, C2, Q1)\n")
            f.write("2. For each component, identify the most likely value (e.g., 10kΩ, 2.2µF, 1N4148)\n")
            f.write("3. Consider proximity, confidence scores, and text patterns when making determinations\n")
            f.write("4. Common patterns:\n")
            f.write("   - Resistors: R# with values like 1k, 10k, 100Ω\n")
            f.write("   - Capacitors: C# with values like 10µF, 100nF\n")
            f.write("   - Transistors: Q# with part numbers like 2N2222, BC547\n")
            f.write("   - Diodes: D# with part numbers like 1N4148, 1N4007\n")
            f.write("   - Inductors: L# with values like 10mH, 100µH\n")
        
        # Create visual proximity map showing assignments
        proximity_vis = image.copy()
        
        # Color map for components
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # Draw components and their assigned texts
        for idx, (comp_id, text_items) in enumerate(component_text_mapping.items()):
            comp_info = next((c for c in components if c['id'] == comp_id), None)
            if comp_info and 'bbox' in comp_info:
                bbox = comp_info['bbox']
                if len(bbox) == 4:
                    x_min, y_min, x_max, y_max = bbox
                    color = colors[idx % len(colors)]
                    
                    # Draw component box
                    cv2.rectangle(proximity_vis, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(proximity_vis, comp_id, (x_min, y_min - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw lines to all assigned texts
                    comp_center = get_component_center(comp_info['bbox'])
                    for text_item in text_items:
                        if text_item.get('center'):
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
    
    else:
        # If no components provided, just create a simple text list
        with open(os.path.join(ocr_output_folder, "circuit_components_with_text.txt"), 'w', encoding='utf-8') as f:
            f.write("CIRCUIT TEXT DETECTION (No component mapping available)\n")
            f.write("="*80 + "\n\n")
            f.write("ALL DETECTED TEXT:\n")
            f.write("-"*40 + "\n")
            for item in text_with_info:
                f.write(f"{item['text']} (confidence: {item['confidence']:.3f})\n")
    
    return text_with_info