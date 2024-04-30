import subprocess
from sklearn.model_selection import ParameterGrid

data = 'ml-1m'
data_path = f'../datasets/{data}/'

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

for idx, hyperparam in enumerate(hyperparams):
    print(f'Exploration number: {idx + 1}')
    arg_list = ['python3.10', './main.py']
    for arg_name, arg_value in hyperparam.items():
        arg_list.append(arg_name)
        arg_list.append(str(arg_value))
    arg_list.append('--dataset')
    arg_list.append(data)
    arg_list.append('--data_path')
    arg_list.append(data_path)
    try:
        subprocess.run(arg_list, check=True)
        print("Python script executed successfully!")
    except subprocess.CalledProcessError as e:
        print("Error running Python script:", e)
