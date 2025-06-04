import os
import numpy as np
import cv2  
from ultralytics import YOLO
from collections import defaultdict

def validate_detection_model(latest_model_path, validate_images_path, output_dir, annotations_path=None):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(latest_model_path)

    validate_images = [os.path.join(validate_images_path, f) for f in os.listdir(validate_images_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not validate_images:
        print(f"No images found in {validate_images_path}")
        return

    metrics = {
        'per_class': defaultdict(lambda: {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'confidence': []
        }),
        'overall': {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'confidence': []
        }
    }

    all_detections = []
    class_counts = defaultdict(int)

    for img_path in validate_images:
        img_filename = os.path.basename(img_path)
        print(f"Processing {img_filename}...")

        results = model(img_path)[0]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        img_height, img_width = img.shape[:2]

        ground_truth = []
        if annotations_path:
            ann_file = os.path.join(annotations_path, os.path.splitext(img_filename)[0] + '.txt')
            if os.path.exists(ann_file):
                with open(ann_file, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        class_id = int(parts[0])
                        cx, cy, w, h = [float(x) for x in parts[1:5]]
                        class_counts[class_id] += 1

                        x1 = int((cx - w / 2) * img_width)
                        y1 = int((cy - h / 2) * img_height)
                        x2 = int((cx + w / 2) * img_width)
                        y2 = int((cy + h / 2) * img_height)

                        ground_truth.append({
                            'class_id': class_id,
                            'bbox': [x1, y1, x2, y2],
                            'matched': False
                        })

                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for GT

        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        classes = results.boxes.cls.cpu().numpy() if hasattr(results.boxes, 'cls') else []
        confidences = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, 'conf') else []

        predictions_matched = [False] * len(boxes)

        for i, box in enumerate(boxes):
            class_id = int(classes[i]) if i < len(classes) else 0
            confidence = float(confidences[i]) if i < len(confidences) else 1.0

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for predictions
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            detection_entry = {
                'class_id': class_id,
                'bbox': box.tolist(),
                'confidence': confidence,
                'matched': False,
                'image_file': img_filename
            }

            if ground_truth:
                best_match_idx = -1
                best_match_iou = 0.3

                for gt_idx, gt in enumerate(ground_truth):
                    if gt['matched'] or gt['class_id'] != class_id:
                        continue

                    iou = calculate_iou(box, gt['bbox'])
                    if iou > best_match_iou:
                        best_match_iou = iou
                        best_match_idx = gt_idx

                if best_match_idx >= 0:
                    gt = ground_truth[best_match_idx]
                    gt['matched'] = True
                    predictions_matched[i] = True
                    detection_entry['matched'] = True

                    metrics['per_class'][class_id]['true_positives'] += 1
                    metrics['overall']['true_positives'] += 1
                    metrics['per_class'][class_id]['confidence'].append(confidence)
                    metrics['overall']['confidence'].append(confidence)
                else:
                    metrics['per_class'][class_id]['false_positives'] += 1
                    metrics['overall']['false_positives'] += 1

            all_detections.append(detection_entry)

        for gt in ground_truth:
            if not gt['matched']:
                metrics['per_class'][gt['class_id']]['false_negatives'] += 1
                metrics['overall']['false_negatives'] += 1

        output_path_img = os.path.join(output_dir, img_filename)
        cv2.imwrite(output_path_img, img)

    mAP, per_class_AP = calculate_map(all_detections, model.names)
    weighted_mAP = calculate_weighted_map(per_class_AP, class_counts)

    metrics['mAP'] = mAP
    metrics['weighted_mAP'] = weighted_mAP
    metrics['per_class_AP'] = per_class_AP
    metrics['class_counts'] = dict(class_counts)

    generate_detection_report(metrics, model.names, output_dir)
    print(f"Results saved to {output_dir}")
    return metrics

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0.0

def calculate_map(all_detections, class_names, iou_threshold=0.5):
    class_detections = defaultdict(list)
    for det in all_detections:
        class_detections[det['class_id']].append(det)

    ap_values = {}
    for class_id, detections in class_detections.items():
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        tp = np.array([det['matched'] for det in detections], dtype=np.float32)
        fp = 1 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        total_gt = tp.sum()
        if total_gt == 0:
            ap_values[class_id] = 0.0
            continue
        precision = tp_cum / (tp_cum + fp_cum + 1e-10)
        recall = tp_cum / total_gt
        ap = 0
        for r in np.linspace(0, 1, 11):
            prec_at_rec = precision[recall >= r]
            ap += np.max(prec_at_rec) if prec_at_rec.size > 0 else 0
        ap_values[class_id] = ap / 11

    mAP = sum(ap_values.values()) / len(ap_values) if ap_values else 0.0
    return mAP, ap_values

def calculate_weighted_map(per_class_AP, class_counts):
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    weighted_sum = sum(per_class_AP[c] * (class_counts[c] / total) for c in per_class_AP)
    return weighted_sum

def generate_detection_report(metrics, class_names, output_dir):
    report_path = os.path.join(output_dir, 'detection_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("      DETECTION MODEL PERFORMANCE REPORT\n")
        f.write("="*60 + "\n\n")

        # Categorize by performance
        class_results = {'GOOD': [], 'WARNING': [], 'CRITICAL': []}
        total_instances = sum(metrics['class_counts'].values())

        for class_id, data in sorted(metrics['per_class'].items()):
            tp = data['true_positives']
            fp = data['false_positives']
            fn = data['false_negatives']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            ap = metrics['per_class_AP'].get(class_id, 0)
            count = metrics['class_counts'].get(class_id, 0)
            name = class_names.get(class_id, f"Class {class_id}")
            conf = np.mean(data['confidence']) if data['confidence'] else 0

            if tp < 5:
                status = 'CRITICAL'
            elif f1 < 0.6 or ap < 0.5:
                status = 'WARNING'
            else:
                status = 'GOOD'

            class_results[status].append({
                'name': name,
                'id': class_id,
                'f1': f1,
                'ap': ap,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'count': count,
                'conf': conf
            })

        # Define how to write the table
        def write_class_table(title, rows, icon):
            if not rows:
                return
            f.write(f"\n{icon} {title}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Class':<17} {'F1':>6} {'AP':>4} {'TP':>5} {'FP':>5} {'FN':>4} {'Samples':>10}\n")
            f.write("-" * 60 + "\n")
            for r in rows:
                f.write(f"{r['name']:<20} {r['f1']:.2f} {r['ap']:.2f} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['count']:>8}\n")
            f.write("\n")

        write_class_table("GOOD PERFORMANCE", class_results['GOOD'], "✅")
        write_class_table("WARNING – Needs Attention", class_results['WARNING'], "⚠️")
        write_class_table("CRITICAL – Very Poor or Data Deficient", class_results['CRITICAL'], "❌")

        # Overall metrics
        f.write("\n" + "="*60 + "\n")
        f.write("🔍 OVERALL MODEL PERFORMANCE\n")
        f.write("-" * 60 + "\n")
        overall = metrics['overall']
        tp = overall['true_positives']
        fp = overall['false_positives']
        fn = overall['false_negatives']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_conf = np.mean(overall['confidence']) if overall['confidence'] else 0

        f.write(f"Total Predictions: {tp + fp}\n")
        f.write(f"Total Ground Truth: {tp + fn}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"F1 Score: {f1:.2f}\n")
        f.write(f"Mean Average Precision (mAP@0.5): {metrics['mAP']:.2f}\n")
        f.write(f"Weighted mAP@0.5: {metrics['weighted_mAP']:.2f}\n")
        f.write(f"Average Confidence: {avg_conf:.2f}\n")

        f.write("\nClass Distribution:\n")
        for class_id, count in sorted(metrics['class_counts'].items(), key=lambda x: x[1], reverse=True):
            name = class_names.get(class_id, f"Class {class_id}")
            pct = (count / total_instances) * 100 if total_instances else 0
            f.write(f"  - {name} (ID: {class_id}): {count} instances ({pct:.1f}%)\n")

        f.write("\n📖 LEGEND (Abbreviations Used)\n")
        f.write("-" * 60 + "\n")
        f.write("TP  = True Positives\n")
        f.write("FP  = False Positives\n")
        f.write("FN  = False Negatives\n")
        f.write("F1  = F1 Score (harmonic mean of precision and recall)\n")
        f.write("AP  = Average Precision at IoU=0.5\n")
        f.write("mAP = Mean Average Precision across all classes\n")
        f.write("IoU = Intersection over Union (box overlap metric)\n")
        f.write("Conf = Model’s average confidence for correct detections\n")

    print(f"📝  report saved to {report_path}")


def main():
    current_dir = os.getcwd()
    PROJECT_PATH = os.path.dirname(os.path.dirname(current_dir))
    model_folder = os.path.join(PROJECT_PATH, 'current_trained_model')
    train_folders = [folder for folder in os.listdir(model_folder) if folder.startswith('train')]

    if not train_folders:
        raise FileNotFoundError("No 'train' folders found in the detect directory.")

    def extract_suffix(folder_name):
        return int(folder_name[5:]) if folder_name != "train" else 0

    latest_train_folder = max(train_folders, key=extract_suffix)
    latest_model_path = os.path.join(model_folder, latest_train_folder, 'weights', 'last.pt')

    validate_images_path = os.path.join(current_dir, 'inputs', 'validation_images')
    annotations_path = os.path.join(current_dir, 'inputs', 'validation_annotations')
    output_dir = os.path.join(current_dir, 'outputs', 'validation_results')

    print(f"Using model: {latest_model_path}")

    validate_detection_model(latest_model_path, validate_images_path, output_dir, annotations_path)

    print("Testing completed.")

if __name__ == '__main__':
    main()