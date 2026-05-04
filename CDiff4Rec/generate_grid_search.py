import os
import argparse
import random
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='/content/FairDiffRec/datasets/', help='load data path')

args = parser.parse_args()

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
    
    os.makedirs(log_dir, exist_ok=True)

    command_lines = set()

    print(f'Total configurations: {len(hyperparams)}')

    for hyperparam in hyperparams:
            
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

    sorted_command_lines = sorted(list(command_lines))

    print(f'Configurations to run: {len(sorted_command_lines)}')

    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sh_file_path = os.path.join(script_dir, f'train_all_{args.dataset}.sh')

    with open(sh_file_path, 'w') as f:
        print('#!/bin/bash', file=f)
        for cmdl in sorted_command_lines:
            print(cmdl, file=f)

if __name__ == '__main__':
    main()
