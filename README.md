# TCSD

Implementation of TCSD for temporal link prediction.

## Requirements

- Python 3.9
- PyTorch 2.0.0
- PyTorch Geometric 2.4.0

Install the dependencies:

```bash
conda create -n tcsd python=3.9
conda activate tcsd
pip install -r requirements.txt
```

## Datasets

The following datasets are supported:

- **CSV datasets:** `uci`, `mooc`, `lastfm`, `wikipedia`, `Flights`, `tgbl-review`, `tgbl-flight`.
- **TGB-Seq datasets:** `GoogleLocal`, `YouTube`, `Flickr`, `Taobao`, `WikiLink`.
- **TGB datasets:** `tgbl-coin`, `tgbl-comment`.

Place the prepared datasets under `./data/`. Data files are not included in this repository.

- CSV datasets require `data/<name>/ml_<name>.csv`, `ml_<name>.npy` and `ml_<name>_node.npy`.
- TGB-Seq datasets use `data/<name>/<name>.csv` and `<name>_test_ns.npy`.
- TGB datasets use the bundled loader with folders `data/tgbl_coin/` and `data/tgbl_comment/`.

For `tgbl-flight`, the folder and file prefix are `tgbl_flight`. Keep the same processed data, splits and negative samples for reproduction. Use `--dataset_path /path/to/data` to specify another data directory.

## Run Examples

Run the commands from the repository root. Dataset-specific hyperparameters are loaded automatically from `configs/datasets.json`.

### Training and evaluation

```bash
python train_link_prediction.py --dataset_name GoogleLocal --gpu 0 --seed 0
python train_link_prediction.py --dataset_name lastfm --gpu 0 --seed 0
```

Replace `--dataset_name` with any dataset listed above. Each run trains the model, selects the best validation checkpoint and evaluates it on the test set.

### Multiple seeds

```bash
for seed in 0 1 2; do
  python train_link_prediction.py --dataset_name GoogleLocal --gpu 0 --seed "$seed" --num_runs 1
done
```

### Test a saved checkpoint

Use the same dataset, seed and model settings as the training run:

```bash
python train_link_prediction.py --dataset_name GoogleLocal --gpu 0 --seed 0 --load_pretrained --num_epochs 0
```

Checkpoints and results are saved in `saved_models/` and `saved_results/`. Testing again with the same settings replaces the corresponding result JSON.
