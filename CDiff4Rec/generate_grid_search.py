import os
import numpy as np
import argparse
import random
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='/content/FairDiffRec/datasets/', help='load data path')

args = parser.parse_args()


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = [1 - alpha_bar[0]]
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


# Unmodified hyperparameters list as requested
hyperparams = ParameterGrid({
    "--lr": [1e-5, 1e-4, 5e-5, 5e-4],
    "--weight_decay": [0.0, 1e-4],
    "--steps": [10, 40, 50],
    "--reweight": [True, False],
    "--alpha": [0.1, 0.3, 0.5, 0.7, 0.9],
    "--topk": [10, 20, 50],
    "--r_agg": ['att', 'avg', 'sim', None]
})


def summary(configuration):
    # Skip None values so they don't corrupt the logfile name
    final_list = [('%s=%s' % (k[2:], v)) for (k, v) in configuration.items() if v is not None]
    return '_'.join(final_list)


def to_cmd(c):
    # Skip None values to avoid passing "None" as a literal string argument
    command = ' '.join([f'{k}={v}' for k, v in c.items() if v is not None])
    return command


def to_logfile(c):
    outfile = "{}.log".format(summary(c).replace("/", "_"))
    return outfile


def main():
    logs_path = 'logs'
    log_dir = os.path.join(logs_path, args.dataset)

    # Safely create directory
    os.makedirs(log_dir, exist_ok=True)

    command_lines = set()

    print(f'Total configurations: {len(hyperparams)}')

    for hyperparam in hyperparams:
        # Use .get() with fallback values since these keys aren't in the ParameterGrid
        noise_scale = hyperparam.get('--noise_scale', 1.0)
        noise_min = hyperparam.get('--noise_min', 0.0001)
        noise_max = hyperparam.get('--noise_max', 0.02)

        start = noise_scale * noise_min
        end = noise_scale * noise_max
        
        variance_schedule = np.linspace(start, end, hyperparam['--steps'], dtype=np.float64)
        betas = betas_from_linear_variance(hyperparam['--steps'], variance_schedule)
        
        if not (len(betas.shape) == 1):
            continue
        if not (len(betas) == hyperparam['--steps']):
            continue
        if not ((betas > 0).all() and (betas <= 1).all()):
            continue
            
        logfile = to_logfile(hyperparam)
        log_filepath = os.path.join(log_dir, logfile)
        completed = False
        
        if os.path.isfile(log_filepath):
            with open(log_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'End. Best Epoch' in content

        if not completed:
            command_line = f'python "/content/FairDiffRec/CDiff4Rec/main.py" {to_cmd(hyperparam)} --cuda --dataset={args.dataset} --data_path={args.data_path} > "{log_filepath}" 2>&1'
            command_lines.add(command_line)

    # Sort command lines and convert back to list
    sorted_command_lines = sorted(list(command_lines))

    print(f'Admissible configurations: {len(sorted_command_lines)}')

    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    print(f'Actual configurations: {len(sorted_command_lines)}')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sh_file_path = os.path.join(script_dir, f'train_all_{args.dataset}.sh')

    with open(sh_file_path, 'w') as f:
        print('#!/bin/bash', file=f)
        for cmdl in sorted_command_lines:
            print(cmdl, file=f)

if __name__ == '__main__':
    main()
