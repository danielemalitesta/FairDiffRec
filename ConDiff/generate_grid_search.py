import os.path
import numpy as np

import argparse
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='foursquare_tky', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='/content/FairDiffRec/datasets', help='load data path')

args = parser.parse_args()


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

hyperparams = ParameterGrid({
    "--lr": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4],
    "--weight_decay": [0.0, 1e-1, 1e-2, 1e-3],
    "--batch_size": [400],
    "--dims": ['[200,200]', '[1000]', '[4000]'],
    "--emb_size": [10],
    "--mean_type": ['x0'],
    "--steps": [5, 10, 50, 100],
    "--noise_scale": [1e-4, 1e-3, 1e-2, 1e-1],
    "--noise_min": [1e-4, 1e-3, 1e-2, 2e-2],
    "--noise_max": [1e-2, 2e-1, 1e-1],
    "--sampling_steps": [0],
    "--reweight": [1],
    "--lamd1": [0.1],
    "--lamd2": [0.1]
})

def summary(configuration):
    final_list = [('%s=%s' % (k[2:], v)) for (k, v) in configuration.items()]
    return '_'.join(final_list)


def to_cmd(c):
    command = ' '.join([f'{k}={v}' for k, v in c.items()])
    return command


def to_logfile(c):
    outfile = "{}.log".format(summary(c).replace("/", "_"))
    return outfile


def main():
    logs_path = 'logs'

    if not os.path.exists(logs_path + f'/{args.dataset}'):
        os.makedirs(logs_path + f'/{args.dataset}')

    command_lines = set()

    print(f'Total configurations: {len(hyperparams)}')

    for hyperparam in hyperparams:
        start = hyperparam['--noise_scale'] * hyperparam['--noise_min']
        end = hyperparam['--noise_scale'] * hyperparam['--noise_max']
        betas = betas_from_linear_variance(hyperparam['--steps'], np.linspace(start, end, hyperparam['--steps'], dtype=np.float64))
        if not (len(betas.shape) == 1):
            continue
        if not (len(betas) == hyperparam['--steps']):
            continue
        if not ((betas > 0).all() and (betas <= 1).all()):
            continue
        logfile = to_logfile(hyperparam)
        completed = False
        if os.path.isfile(f'{logs_path}/{args.dataset}/{logfile}'):
            with open(f'{logs_path}/{args.dataset}/{logfile}', 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'End. Best Epoch' in content

        if not completed:
            command_line = f'python /content/FairDiffRec/ConDiff/main.py {to_cmd(hyperparam)} --cuda --dataset={args.dataset} --data_path={args.data_path}/ > {logs_path}/{args.dataset}/{logfile} 2>&1'
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    print(f'Admissible configurations: {len(sorted_command_lines)}')

    import random
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
