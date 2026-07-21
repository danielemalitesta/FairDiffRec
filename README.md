# FairDiffRec

FairDiffRec is a benchmarking suite for studying **fairness in diffusion-based recommender systems**. It brings together seven diffusion-family recommendation models under a common data format, adds grid-search and best-checkpoint-selection tooling, and provides scripts to measure both **accuracy** and **fairness** (consumer and provider-side) of the generated recommendation lists, alongside a set of traditional/graph-based baselines run through RecBole.

## What's inside

### Diffusion recommenders
Each model lives in its own top-level folder with its own `main.py`, `data_utils.py`, `evaluate_utils.py`, and `models/` implementation, so it can be trained independently:

| Folder | Model | Notes |
|---|---|---|
| `DiffRec/` | DiffRec | Base diffusion recommender operating directly on the user-item interaction matrix |
| `L-DiffRec/` | L-DiffRec | Latent-space variant of DiffRec, clusters items and diffuses in a learned latent space via an autoencoder |
| `CF_Diff/` | CF-Diff | Collaborative-filtering diffusion model that incorporates 1-hop and 2-hop neighborhood information (see `hops/`) |
| `CDiff4Rec/` | CDiff4Rec | Diffusion model with an attention-based denoiser (`models/Attention.py`) |
| `ConDiff/` | ConDiff | Conditional diffusion variant with a dedicated DNN denoiser |
| `CODIGEM/` | CODIGEM | Denoising diffusion generative model adapted for recommendation |
| `GiffCF/` | GiffCF | Graph-signal diffusion recommender, configured via TOML files in `configs/` (one per dataset) |
| `BSPM/` | BSPM | Blurring-Sharpening Process Model, a score/heat-diffusion based graph recommender (uses its own `world.py`/`register.py` runtime) |

Each model implementation was cloned directy from the authors' github repositories.

### Baselines
`recbole/` wraps traditional and graph-based recommenders (BPR, ItemKNN, NeuMF, LightGCN, UltraGCN, XSimGCL, EASE, MultiVAE, RecVAE) through the [RecBole](https://github.com/RUCAIBox/RecBole) library, with per-model hyper-parameter search spaces in `recbole/hyper/` and a `convert_to_recbole.py` script to turn the project's `.npy` interaction files into RecBole's `.inter` format.

### Evaluation & fairness
- `evaluate_fairness.py` — computes consumer-side and provider-side fairness metrics from a saved `.tsv` of top-K recommendations.
- `evaluate_graph.py` — compares the graph structure of the ground-truth interaction matrix against the predicted interaction matrix.
- `get_best_val.py` — scans a directory of training logs produced by a grid search and selects the run with the best validation Recall@20.

### Data
`datasets/` contains four ready-to-use datasets, each with pre-split `train_list.npy`, `valid_list.npy`, `test_list.npy` interaction matrices and `users_map.tsv` / `items_map.tsv` ID mappings:

- `ml-1m/` — MovieLens-1M, with user metadata (`ml-1m.user`) for gender-based fairness analysis
- `foursquare_tky/` — Foursquare check-ins (Tokyo), with user metadata for gender-based fairness analysis
- `books/` — Book-Crossing style data, with `users.csv` for age-based fairness analysis
- `lastfm/` — LastFM listening data, used for activity-based fairness analysis

`hyperparameters_diffusion.txt` and `hyperparameters_traditional.txt` report the best hyper-parameter configurations found for the diffusion models and the RecBole baselines, respectively, across all four datasets.

## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy, scikit-learn, pandas
- Bottleneck
- kmeans-pytorch
- RecBole (for the `recbole/` baselines)
- tabulate, tensorflow (for `GiffCF/`, which also uses TensorFlow-based ops)

## Usage

### 1. Train a diffusion model

Most models folder exposes a `run.sh` wrapper around `main.py` or a direct `main.py` call .

Model weights are written to `./saved_models/<dataset_name>/` and training logs to `./log/<dataset_name>/`.

### 2. Explore hyper-parameters with a grid search

Every model folder includes a `generate_grid_search.py` script that builds a shell script running many configurations sequentially:

```sh
python generate_grid_search.py --dataset <dataset_name>
./train_all_<dataset_name>.sh
```

### 3. Select the best checkpoint

```sh
python get_best_val.py --dataset <dataset_name>
```

This reads the training logs and reports the configuration with the highest validation Recall@20.

### 4. Run inference

Models that expose an `inference.py` (`DiffRec`, `L-DiffRec`, `CF_Diff`) can be used to generate a top-K recommendation list `.tsv` file (`user_id`, `item_id`, `score`) from a trained checkpoint.

### 5. Evaluate accuracy and fairness

```sh
python evaluate_fairness.py --dataset <dataset_name> --tsv_path <path_to_recommendations.tsv>
python evaluate_graph.py --orig_path <train_matrix.npz> --pred_path <predicted_matrix.npy>
```

### 6. Run RecBole baselines

```sh
python recbole/convert_to_recbole.py
python recbole/run_hyper.py --model <model_name> --dataset <dataset_name> \
       --params_file recbole/hyper/<model_name>.hyper
```

## Notes

- Several scripts (e.g. `evaluate_fairness.py`, `recbole/convert_to_recbole.py`) reference absolute paths such as `/content/FairDiffRec/...`, reflecting a Colab-based development environment — update these paths to match your local setup before running.
- Grid-search scripts sample a subset of the full hyper-parameter space rather than exploring it exhaustively, to keep training time and compute cost tractable.
