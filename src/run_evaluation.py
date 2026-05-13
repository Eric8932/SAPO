import os
import json
import argparse
import re
import numpy as np
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Continual Learning Performance")
    parser.add_argument(
        "--outputs_dir", 
        type=str, 
        required=True, 
        help="Path to the output folder containing subfolders like '1-taskXXX', '2-taskXXX'..."
    )
    parser.add_argument(
        "--initial_metric_file", 
        type=str, 
        required=True, 
        help="Path to the json file containing the model's initial performance (zero-shot/pre-trained)."
    )
    return parser.parse_args()

def extract_metric_value(results_dict, task_identifier, metric_keyword="rougeL"):
    """
    """
    for key, value in results_dict.items():
        if metric_keyword in key and task_identifier in key:
            return value
    
    print(f"Warning: Could not find key with '{metric_keyword}' and '{task_identifier}' in results.")
    return 0.0

def calculate_cl_metrics(l, l0):
    """
    l: trained_performance (N x N list)
    l0: original_performance (1 x N list)
    """
    l_np = np.array(l)
    l0_np = np.array(l0)

    N = l_np.shape[0]
    R = np.vstack([l0_np, l_np])
    
    #AP: Average Performance
    ap = np.mean(R[N, :])

    # BWT: Backward Information Transfer (Forgetting)
    bwt_terms = []
    for i in range(2, N+1):
        current_perf_on_old_tasks = R[i, :i-1]
        original_perf_on_old_tasks = np.array([R[j+1, j] for j in range(0, i-1)])
        
        diff = current_perf_on_old_tasks - original_perf_on_old_tasks
        inner_sum = np.sum(diff)
        term = (1 / (i-1)) * inner_sum
        bwt_terms.append(term)
    bwt = (1 / len(bwt_terms)) * sum(bwt_terms) if bwt_terms else 0.0


    # FWT: Forward Information Transfer (Generalization)
    fwt_terms = []
    for i in range(1, N): 
        diff = R[i, i:] - R[0, i:]
        inner_sum = np.sum(diff)
        term = (1 / (N - i)) * inner_sum
        fwt_terms.append(term)
    
    fwt = (1 / len(fwt_terms)) * sum(fwt_terms) if fwt_terms else 0.0

    return {
        "AP": round(float(ap), 4), 
        "BWT": round(float(bwt), 4), 
        "FWT": round(float(fwt), 4)
    }

def main():
    args = parse_args()
    
    if not os.path.exists(args.outputs_dir):
        raise FileNotFoundError(f"Directory not found: {args.outputs_dir}")
    
    # 1. Sort the Folder (1-taskA, 2-taskB ...)
    subdirs = []
    for d in os.listdir(args.outputs_dir):
        path = os.path.join(args.outputs_dir, d)
        if os.path.isdir(path):
            match = re.match(r"^(\d+)-(.+)$", d)
            if match:
                order = int(match.group(1))
                task_name = match.group(2)
                subdirs.append({
                    "order": order,
                    "task_identifier": task_name,
                    "folder_path": path,
                    "folder_name": d
                })
    
    subdirs.sort(key=lambda x: x["order"])
    
    if not subdirs:
        print("No valid task directories found (pattern: 'N-taskXXX').")
        return
    
    task_identifiers = [item["task_identifier"] for item in subdirs]
    print(f"Detected {len(task_identifiers)} tasks in order: {task_identifiers}")

    # 2. Read the Original Performance) - l0
    print(f"Loading initial metrics from: {args.initial_metric_file}")
    with open(args.initial_metric_file, 'r', encoding='utf-8') as f:
        initial_data = json.load(f)
    
    original_performance = []
    for task_id in task_identifiers:
        val = extract_metric_value(initial_data, task_id)
        original_performance.append(val)
    
    print(f"Original Performance: {original_performance}")

    # 3. Read the Trained Performance - l
    trained_performance = []
    
    for step_idx, item in enumerate(subdirs):
        result_file = os.path.join(item["folder_path"], "all_results.json")
        
        if not os.path.exists(result_file):
            print(f"Error: {result_file} not found.")
            current_step_scores = [0.0] * len(task_identifiers)
        else:
            with open(result_file, 'r', encoding='utf-8') as f:
                res_data = json.load(f)
            
            current_step_scores = []
            for task_id in task_identifiers:
                val = extract_metric_value(res_data, task_id)
                current_step_scores.append(val)
        
        trained_performance.append(current_step_scores)
        print(f"Metrics after Task {item['order']} ({item['task_identifier']}): {current_step_scores}")

    metrics = calculate_cl_metrics(trained_performance, original_performance)
    print("\nCalculated Metrics:")
    print(json.dumps(metrics, indent=4))

    # 5. 保存结果
    output_data = {
        "original_performance": original_performance,
        "trained_performance": trained_performance,
        "metrics": metrics
    }
    
    save_path = os.path.join(args.outputs_dir, "results.json")
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, cls=NumpyEncoder)
    
    print(f"\nAll results saved to: {save_path}")

if __name__ == "__main__":
    main()