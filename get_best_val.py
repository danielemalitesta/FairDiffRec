import os
import subprocess
import re
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Run grid search and find the best hyperparameters")
    parser.add_argument('--dataset', type=str, default='ml-1m', help='Dataset name')
    args = parser.parse_args()

    dataset = args.dataset

    log_dir = os.path.join("/content/logs", dataset)
    if not os.path.exists(log_dir):
        print(f"Log directory '{log_dir}' does not exist.")
        return

    best_valid_recall = -1.0
    best_test_recall = -1.0
    best_log_file = None
    best_hyperparams = ""

    metric_pattern = re.compile(r"_valid_recall_([0-9.]+)_test_recall_([0-9.]+)")

    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            if "End. Best Epoch" not in content:
                continue 

            match = metric_pattern.search(content)
            if match:
                valid_recall = float(match.group(1))
                test_recall = float(match.group(2))

                if valid_recall > best_valid_recall:
                    best_valid_recall = valid_recall
                    best_test_recall = test_recall
                    best_log_file = log_file
                    
                    best_hyperparams = os.path.basename(log_file).replace('.log', '')

    if best_log_file:
        print("\n" + "="*50)
        print("GRID SEARCH COMPLETED - BEST RESULTS FOUND")
        print("="*50)
        print("Best Hyperparameters:")
        
        params = re.findall(r'([a-zA-Z0-9_]+=[a-zA-Z0-9.\-, \[\]]+)', best_hyperparams)
        for p in params:
            print(f"  - {p}")
            
        print(f"\nBest Validation Recall: {best_valid_recall:.4f}")
        print(f"Associated Test Recall:   {best_test_recall:.4f}")
        print(f"Log File: {best_log_file}")
        print("="*50 + "\n")
    else:
        print("No completed logs with valid metrics found. Please check your log files for errors.")

if __name__ == '__main__':
    main()
