import os
import re

def extract_labels_and_values(filepath):
    labels = []
    values = []

    def normalize_value(value):
        return value.replace(" ", "").lower()

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove prefix before colon (if present)
        if ':' in line:
            line = line.split(':', 1)[1].strip()

        # Extract label (first item before comma)
        parts = line.split(',')
        if parts:
            label = parts[0].strip()
            labels.append(label)

        # Extract and normalize value
        if len(parts) > 2:
            value = normalize_value(parts[2].strip())
            values.append(value)
        elif len(parts) > 1:
            match = re.search(r'(\d+\.?\d*\s*[A-Za-zΩ]+)', parts[1])
            if match:
                value = normalize_value(match.group(1).strip())
                values.append(value)

    return labels, values

def compare_folders(folder1, folder2):
    total_comparisons = 0
    correct_matches = 0
    results = []

    files1 = [f for f in os.listdir(folder1) if f.endswith('.txt')]
    files2 = [f for f in os.listdir(folder2) if f.endswith('.txt')]

    matched_files = set(files1) & set(files2)

    for filename in matched_files:
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)

        labels1, values1 = extract_labels_and_values(path1)
        labels2, values2 = extract_labels_and_values(path2)

        label_matches = sum(1 for label in labels1 if label in labels2)
        value_matches = sum(1 for value in values1 if value in values2)

        total = len(labels1) + len(values1)
        correct = label_matches + value_matches

        total_comparisons += total
        correct_matches += correct

        results.append({
            'file': filename,
            'label_matches': label_matches,
            'value_matches': value_matches,
            'total': total,
            'correct': correct,
            'accuracy': round(100 * correct / total, 2) if total > 0 else 0
        })

    overall_accuracy = round(100 * correct_matches / total_comparisons, 2) if total_comparisons > 0 else 0

    return {
        'per_file': results,
        'overall': {
            'total_comparisons': total_comparisons,
            'correct_matches': correct_matches,
            'accuracy': overall_accuracy
        }
    }

def main():
    # 🧭 Set up relative paths from the current directory
    folder1 = os.path.join(os.getcwd(), 'inputs', 'true_netlists')
    folder2 = os.path.join(os.getcwd(), 'outputs', 'test_results')

    results = compare_folders(folder1, folder2)

    print("\nPer-file results:")
    for r in results['per_file']:
        print(f"{r['file']}: {r['correct']}/{r['total']} correct ({r['accuracy']}%)")

    print("\nOverall summary:")
    overall = results['overall']
    print(f"Total comparisons: {overall['total_comparisons']}")
    print(f"Correct matches: {overall['correct_matches']}")
    print(f"Accuracy: {overall['accuracy']}%")

if __name__ == "__main__":
    main()
