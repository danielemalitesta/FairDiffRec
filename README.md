# Fairness in Diffusion Recommender Models
This is the official implementation of the paper "_How Fair is Your Diffusion Recommender Model?_", accepted in the LBR track at ACM RecSys 2025.

The current repository is inspired by the code of the paper "_Diffusion Recommender Model_", published at SIGIR 2023:
```
@inproceedings{wang2023diffrec,
title = {Diffusion Recommender Model},
author = {Wang, Wenjie and Xu, Yiyan and Feng, Fuli and Lin, Xinyu and He, Xiangnan and Chua, Tat-Seng},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {832–841},
publisher = {ACM},
year = {2023}
}
```

so refer to the original authors' [code](https://github.com/YiyanXu/DiffRec) for the official implementation.

### Requirements
To begin with, install the useful packages as indicated in the original code (the authors suggest using Anaconda 3):

- torch
- numPy
- Bottleneck
- kmeans-pytorch
- scikit-learn

### Fairness analysis

#### Datasets [Zenodo DOI 10.5281/zenodo.11502753](https://doi.org/10.5281/zenodo.11502753)
For this RQ, you need two more recommendation datasets (Movielens-1M_A and Foursquare_TKY) which come with users' metadata to calculate fairness metrics. While we already provide the item embeddings for the two datasets in this repo, you need to download the other dataset files from Zenodo.

Once you have downloaded them, run the following script: 
```sh
python DiffRec/convert_datasets.py
```
by changing the dataset name inside the script. This will generate the train, validation, and test sets in a compatible version to run with the authors' original code.

#### Reproduce results

To reproduce the results, you first need to train DiffRec and L-DiffRec by exploring the hyper-parameters through a grid search. The original authors' code does not allow, by design, to explore various hyper-parameters settings and choose the best one according to the results on the validation set. Thus, we provide a possible implementation to create the whole grid search exploration, train the models, and eventually select the best configuration on the validation set.

First, generate the bash script that will run (sequentially) all configurations. You may accelerate the training process in different manners (e.g., parallelizing the scripts):

```sh
python generate_grid_search.py --dataset <dataset_name>
```

Where you need to specify the dataset name. Note that you should run this for both DiffRec and L-DiffRec, as the two exploration spaces differ. Also note that this script randomly selects 200 and 500 explorations (for DiffRec and L-DiffRec) to run from the total explorations. As stated in the paper, and for the sake of time, we decided not to explore the whole search space, since it would have been too large and computationally expensive. We believe this might represent a sufficient approximation of the optimal results. Finally, we also put a check on the generated configurations (following the original authors' code) since some of them may not be admissable.

This will generate the bash file train_all_<dataset-name>.sh. Run it through:

```sh
./train_all_<dataset_name>.sh
```

Once the training is done, you will see the model weights files at ./saved_models/<dataset_name> and the log files at the path ./logs/<dataset-name>/. To find the best configuration according to the Recall@20 on the validation set, run the bash script:

```sh
./get_best_val.sh <dataset_name>
```
that sorts all log names according to the Recall@20 on the validation set. Again, you'll need to run this for both DiffRec and L-DiffRec.

For your convenience, here, we report the best hyper-parameters for DiffRec and L-DiffRec on Movielens-1M_A and Foursquare_TKY as we found them:

|                    | **DiffRec**                                                                                                                                                                                                             | **L-DiffRec**                                                                                                                                                                                                                                                                                        |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Movielens-1M-A** | batch_size=400<br/>dims=[200,600]<br/>emb_size=10<br/>lr=0.0001<br/>mean_type=x0<br/>noise_max=0.01<br/>noise_min=0.0005<br/>noise_scale=0.005<br/>reweight=False<br/>sampling_steps=0<br/>steps=5<br/>weight_decay=0.0 | batch_size=400<br/>emb_size=10<br/>in_dims=[300]<br/>lamda=0.01<br/>lr1=0.001<br/>lr2=0.001<br/>mean_type=x0<br/>mlp_dims=[300]<br/>n_cate=2<br/>noise_max=0.01<br/>noise_min=0.0005<br/>noise_scale=0.1<br/>out_dims=[]<br/>reweight=False<br/>sampling_steps=0<br/>steps=5<br/>wd1=0.0<br/>wd2=0.0 |
| **Foursquare-TKY** | batch_size=400<br/>dims=[1000]<br/>emb_size=10<br/>lr=0.0001<br/>mean_type=x0<br/>noise_max=0.01<br/>noise_min=0.001<br/>noise_scale=0.005<br/>reweight=False<br/>sampling_steps=0<br/>steps=40<br/>weight_decay=0.0    | batch_size=400<br/>emb_size=10<br/>in_dims=[300]<br/>lamda=0.03<br/>lr1=0.001<br/>lr2=0.001<br/>mean_type=x0<br/>mlp_dims=[300]<br/>n_cate=2<br/>noise_max=0.005<br/>noise_min=0.001<br/>noise_scale=0.1<br/>out_dims=[]<br/>reweight=True<br/>sampling_steps=0<br/>steps=2<br/>wd1=0.0<br/>wd2=0.0  |

Then, run the inference scripts for all settings. This will produce a tsv file (in ./saved_models/<dataset_name>/) containing the predicted recommendation lists for each user.

All other recommendation baselines for RQ2 are trained with [RecBole](https://github.com/RUCAIBox/RecBole). You will use the scripts trainer.py and metrics.py (heavily dependent on RecBole) to calculate the fairness metrics from the obtained recommendation lists (tsv files).

The best hyper-parameters found on the two datasets are:

- BPRMF:
  - foursquare_tky:
      - embedding_size: 128
      - learning_rate: 2e-4
  - ml-1m:
      - embedding_size: 128
      - learning_rate: 2e-4
- ItemkNN
  - foursquare_tky:
    - k: 200
    - learning_rate: 2e-4
    - shrink: 0
  - ml-1m:
    - k: 400
    - learning_rate: 2e-4
    - shrink: 2
- NeuMF:
    - foursquare_tky:
      - dropout_prob: 0.3
      - learning_rate: 2e-4
      - mf_embedding_size: 128
      - mlp_embedding_size: 64
      - mlp_hidden_size: [128,64]
    - ml-1m:
      - dropout_prob: 0.3
      - learning_rate: 2e-4
      - mf_embedding_size: 64
      - mlp_embedding_size: 64
      - mlp_hidden_size: [128,64]
- LightGCN:
    - foursquare_tky:
      - learning_rate: 2e-4
    - ml-1m:
      - learning_rate: 2e-4
- UltraGCN:
    - foursquare_tky:
      - ILoss_lambda: 1e-7
      - learning_rate: 2e-4
      - negative_weight: 10
      - w1: 1.0
      - w2: 1e-7
      - w3: 1.0
      - w4: 1e-7
    - ml-1m:
      - ILoss_lambda: 1e-7
      - learning_rate: 2e-4
      - negative_weight: 10
      - w1: 1.0
      - w2: 1e-7
      - w3: 1.0
      - w4: 1e-7
- XSimGCL:
    - foursquare_nyc:
      - eps: 0.2
      - lambda: 0.05
      - learning_rate: 2e-4
      - temperature: 0.1
    - ml-1m:
      - eps: 0.2
      - lambda: 0.05
      - learning_rate: 2e-4
      - temperature: 0.1
- EASE:
    - foursquare_nyc:
      - learning_rate: 1e-4
      - reg_weight: 100
    - ml-1m:
      - learning_rate: 2e-4
      - reg_weight: 2000
- MultiVAE
  - foursquare_tky:
    - dropout_prob: 0.5
    - mlp_hidden_size: [600]
    - learning_rate: 2e-4
    - latent_dimension: 200
  - ml-1m:
    - dropout_prob: 0.3
    - mlp_hidden_size: [600]
    - learning_rate: 2e-4
    - latent_dimension: 200
- RecVAE
  - foursquare_tky:
    - dropout_prob: 0.5
    - hidden_dimension: 600
    - gamma: 5e-3
    - learning_rate: 2e-4
  - ml-1m:
    - dropout_prob: 0.1
    - hidden_dimension: 600
    - gamma: 5e-3
    - learning_rate: 2e-4
