import os
import re

def normalize_value(value):
    """Normalize values by removing units and standardizing format."""
    value = value.lower().replace(" ", "")
    value = value.replace('µ', 'u')  # Normalize micro symbol
    value = re.sub(r'(v|a|f|ohm|Ω)$', '', value)  # Remove trailing units
    return value

def extract_labels_and_values(filepath):
    labels = []
    values = []

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Handle long-form label: LabelName:ShortLabel
        long_label = None
        if ':' in line:
            long_label, line = line.split(':', 1)
            long_label = long_label.strip()

        parts = [p.strip() for p in line.split(',')]

        # Label extraction
        if parts:
            short_label = parts[0].lower()
            if long_label:
                labels.append(long_label.lower())
            labels.append(short_label)

        if len(parts) == 3:
            raw_value = parts[2]
            match = re.search(r'(\d+\.?\d*)\s*([kKmMuUnNpPµ]?)([a-zA-ZΩ]*)', raw_value)
            if match:
                number, prefix, _ = match.groups()
                prefix = prefix.lower().replace('µ', 'u')
                value = f"{number}{prefix}"
                values.append(normalize_value(value))

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

        unmatched_labels = [label for label in labels1 if label not in labels2]
        unmatched_values = [value for value in values1 if value not in values2]

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
            'accuracy': round(100 * correct / total, 2) if total > 0 else 0,
            'unmatched_labels': unmatched_labels,
            'unmatched_values': unmatched_values
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
    folder1 = os.path.join(os.getcwd(), 'inputs', 'true_netlists')
    folder2 = os.path.join(os.getcwd(), 'outputs', 'test_results')

    results = compare_folders(folder1, folder2)

    print("\nPer-file results:")
    for r in results['per_file']:
        print(f"{r['file']}: {r['correct']}/{r['total']} correct ({r['accuracy']}%)")
        if r['unmatched_labels']:
            print(f"  Unmatched Labels: {r['unmatched_labels']}")
        if r['unmatched_values']:
            print(f"  Unmatched Values: {r['unmatched_values']}")

    print("\nOverall summary:")
    overall = results['overall']
    print(f"Total comparisons: {overall['total_comparisons']}")
    print(f"Correct matches: {overall['correct_matches']}")
    print(f"Accuracy: {overall['accuracy']}%")

if __name__ == "__main__":
    main()
