#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import numpy as np
import datetime

import argparse
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='../datasets/', help='load data path')
parser.add_argument('--batch_size_jobs', type=int, default=5, help='batch size for jobs')
parser.add_argument('--cluster', type=str, default='mesocentre', help='cluster name')

args = parser.parse_args()

def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

hyperparams = ParameterGrid({
    "--lr": [1e-5, 1e-4, 1e-3, 1e-2],
    "--weight_decay": [0.0],
    "--batch_size": [400],
    "--dims": ['[300]', '[200,600]', '[1000]'],
    "--emb_size": [10],
    "--mean_type": ['x0'],
    "--steps": [2, 5, 10, 40, 50, 100],
    "--noise_scale": [1e-5, 1e-4, 5e-3, 1e-2, 1e-1],
    "--noise_min": [5e-4, 1e-3, 5e-3],
    "--noise_max": [5e-3, 1e-2],
    "--sampling_steps": [0],
    "--reweight": [True, False]
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
    scripts_path = 'scripts'

    if not os.path.exists(logs_path + f'/{args.dataset}'):
        os.makedirs(logs_path + f'/{args.dataset}')

    if not os.path.exists(scripts_path + f'/{args.dataset}'):
        os.makedirs(scripts_path + f'/{args.dataset}')

    command_lines = set()

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
            command_line = f'$HOME/.conda/envs/diffrec/bin/python main_cluster.py {to_cmd(hyperparam)} --dataset={args.dataset} --data_path={args.data_path}/{args.dataset}/ > {logs_path}/{args.dataset}/{logfile} 2>&1'
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    if args.batch_size_jobs == -1:
        args.batch_size_jobs = nb_jobs

    if args.cluster == 'margaret':
        header = None
    else:
        header = """#!/bin/bash -l

#SBATCH --output=/workdir/%u/slogs/diffrec-%A_%a.out
#SBATCH --error=/workdir/%u/slogs/diffrec-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --job-name=diffrec
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB # memory in Mb
#SBATCH --cpus-per-task=4 # number of cpus to use - there are 32 on each node.
#SBATCH --time=4:00:00 # time requested in days-hours:minutes:seconds
#SBATCH --array=1-{0}

echo "Setting up bash environment"
source ~/.bashrc
set -x

# Modules
module purge
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/11.8.0/gcc-11.2.0

# Conda environment
source activate diffrec

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/FairDiffRec/DiffRec

"""

    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    if header:
        for index, offset in enumerate(range(0, nb_jobs, args.batch_size_jobs), 1):
            offset_stop = min(offset + args.batch_size_jobs, nb_jobs)
            with open(scripts_path + f'/{args.dataset}/' + date_time + f'__{index}.sh', 'w') as f:
                print(header.format(offset_stop - offset), file=f)
                current_command_lines = sorted_command_lines[offset: offset_stop]
                for job_id, command_line in enumerate(current_command_lines, 1):
                    print(f'test $SLURM_ARRAY_TASK_ID -eq {job_id} && sleep 10 && {command_line}', file=f)


if __name__ == '__main__':
    main()
