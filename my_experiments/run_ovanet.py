import os
import subprocess
import itertools


LR_LIST = [0.001, 0.0005, 0.0001]    
BATCH_SIZE_LIST = [32, 64, 128]      
THRESHOLD_LIST = [0.3, 0.5, 0.7]  
EPOCHS_LIST = [3, 5, 10]          
SEED_LIST = [42, 100, 1234]         

param_combinations = list(itertools.product(
    LR_LIST, BATCH_SIZE_LIST, THRESHOLD_LIST, EPOCHS_LIST, SEED_LIST
))

for params in param_combinations:
    lr, batch_size, threshold, epochs, seed = params

    log_file = f"logs/lr{lr}_bs{batch_size}_th{threshold}_ep{epochs}_seed{seed}.out"

    command = [
        "python", "OVANet.py",
        "--train_dataset", "mnist",
        "--test_known_dataset", "mnist",
        "--test_unknown_dataset", "emnist",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--threshold", str(threshold),
        "--seed", str(seed)
    ]

    print(f"Launching experiment: {command}")

    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=f, text=True)

    print(f"Experiment done: lr={lr}, batch_size={batch_size}, threshold={threshold}, epochs={epochs}, seed={seed} (Log: {log_file})")

print("All experiments completed!")