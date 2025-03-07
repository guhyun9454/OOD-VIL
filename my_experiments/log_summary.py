import os
import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="logs")
args = parser.parse_args()
log_dir = args.log_dir

log_files = [f for f in os.listdir(log_dir) if f.endswith('.out')]

data = []

for file in log_files:
    filepath = os.path.join(log_dir, file)
    with open(filepath, 'r') as f:
        content = f.read()
    
    # H-score, Known Accuracy, Unknown Accuracy 추출
    metric_match = re.search(
        r'H-score:\s*([\d\.]+)\s*\|\s*Known Accuracy:\s*([\d\.]+)\s*\|\s*Unknown Accuracy:\s*([\d\.]+)',
        content
    )
    if metric_match:
        h_score = float(metric_match.group(1))
        known_acc = float(metric_match.group(2))
        unknown_acc = float(metric_match.group(3))
    else:
        h_score, known_acc, unknown_acc = None, None, None

    # Confusion Matrix 추출 (여러 줄에 걸쳐 있는 경우)
    cm_match = re.search(
        r'Confusion Matrix:\n((?:\[[^\]]*\]\n?)+)', content, 
        re.DOTALL
    )
    confusion_matrix = cm_match.group(1).strip() if cm_match else None

    data.append({
        "Filename": file,
        "H-score": h_score,
        "Known Accuracy": known_acc,
        "Unknown Accuracy": unknown_acc,
        "Confusion Matrix": confusion_matrix
    })

df = pd.DataFrame(data)
print(df)