import os
import subprocess

LR_LIST = [0.001, 0.0005, 0.0001]
BATCH_SIZE_LIST = [32, 64, 128]
THRESHOLD_LIST = [0.3, 0.5, 0.7]

os.makedirs("logs", exist_ok=True)

for lr in LR_LIST:
    for batch_size in BATCH_SIZE_LIST:
        for threshold in THRESHOLD_LIST:
            log_file = f"logs/lr{lr}_bs{batch_size}_th{threshold}.out"

            command = [
                "python", "OVANet.py",
                "--train_dataset", "mnist",
                "--test_known_dataset", "mnist",
                "--test_unknown_dataset", "emnist",
                "--epochs", "5",
                "--batch_size", str(batch_size),
                "--lr", str(lr),
                "--threshold", str(threshold)
            ]

            print(f"Experiment started: lr={lr}, batch_size={batch_size}, threshold={threshold} (Log: {log_file})")
            with open(log_file, "w") as f:
                subprocess.run(command, stdout=f, stderr=f, text=True)
            print(f"Experiment done: lr={lr}, batch_size={batch_size}, threshold={threshold} (Log: {log_file})")