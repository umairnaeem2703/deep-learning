# MNIST Training Monitoring Experiment

A class assignment on handwritten-digit classification (MNIST) with PyTorch. The goal is to **watch and understand how a network trains**, not to reach state-of-the-art accuracy. A small CNN ([main.py](main.py)) is trained, and everything that happens during training is logged to TensorBoard.

## What the experiment is

The assignment asks us to monitor five things during training:

| Goal | Where it shows up in TensorBoard |
|---|---|
| Training loss and validation loss | Custom Scalars → Loss (`Loss/train`, `Loss/val`); also `Loss/train_batch` |
| Training accuracy and validation accuracy | Custom Scalars → Accuracy (`Accuracy/train`, `Accuracy/val`) |
| Gradient magnitudes during backpropagation | Scalars → `GradNorm/<param>` and `GradNorm/total` |
| How weights and gradients change between epochs | Histograms → `Weights/<param>`, `Gradients/<param>`; Scalars → `WeightChange/<param>` |
| Whether the model converges, stalls, or overfits | The train/val overlays, `LearningRate`, and the per-epoch status line printed to the terminal |

Two experiments cover these goals:

1. **Baseline:** train on the full dataset with the default settings. This shows normal convergence: loss falls, accuracy rises, gradients and weight changes shrink.
2. **Overfitting demo:** train on only 1,000 images, with no learning-rate decay, for 30 epochs. The model memorizes the small training set, so training loss keeps falling while validation loss starts to rise.

### Model

`Net` is a small convolutional network: two 3x3 convolutions (32 and 64 channels), a 2x2 max-pool, dropout (0.25), a fully connected layer of 128 units, dropout (0.5), and a 10-way log-softmax output. It has 1,199,882 trainable parameters, about 98% of them in `fc1`. See [model.md](model.md) for the layer-by-layer shapes and parameter counts.

### Data and training setup

| Item | Value |
|---|---|
| Dataset | MNIST, downloaded automatically to `./data` |
| Preprocessing | `ToTensor()` then `Normalize(mean=0.1307, std=0.3081)`; no augmentation |
| Split | 55,000 train / 5,000 validation from the 60k training set (fixed seed); 10,000 test images evaluated once at the end |
| Loss | Negative log-likelihood on log-softmax (equivalent to cross-entropy) |
| Optimizer | Adadelta, `lr=1.0` |
| LR schedule | `StepLR`: LR multiplied by 0.7 every epoch |
| Epochs / batch size | 14 epochs; batch 64 for training, 1000 for validation/test |

The test set is kept apart from all training decisions, so its result is an honest final score.

## How to carry it out

All commands below are run from this `mnist/` folder, so that `data/`, `runs/` and `mnist_cnn.pt` land here.

### 1. Set up the environment

The virtual environment lives at the repo root (`deep-learning/.venv`). Activate it (PowerShell):

```
& "..\.venv\Scripts\Activate.ps1"
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.9.0+cu129` and points pip at the PyTorch CUDA 12.9 index, so this installs the CUDA build. PyTorch 2.6 or newer is required (the script uses `torch.accelerator`).

### 2. Check the GPU (optional)

```bash
python -c "import torch; print('CUDA Engine:', torch.version.cuda); print('GPU Available:', torch.cuda.is_available())"
```

The script uses the GPU automatically when one is available and falls back to the CPU otherwise. Pass `--no-accel` to force the CPU.

### 3. Run the experiments

Quick sanity check (stops each epoch after the first logged batch; validation and test still run in full):

```bash
python main.py --dry-run
```

Experiment 1, baseline:

```bash
python main.py --run-name baseline
```

Experiment 2, overfitting demo:

```bash
python main.py --train-subset 1000 --gamma 1.0 --epochs 30 --run-name overfit
```

Each run writes its logs to `runs/<date>_<host>_<run-name>`. At the end of every epoch the terminal prints a summary line in this format (the numbers below are illustrative, not from a real run):

```
Epoch 3: train loss 0.0512, acc 98.41% | val loss 0.0398, acc 98.86% | lr 0.49 | improving (0 epochs since best val loss)
```

The status is `improving` when validation loss reaches a new best, `overfitting?` when validation loss gets worse while training loss still falls, and `stalled` when both stop improving. It is a rough hint only; the TensorBoard charts are the real evidence.

### 4. View the results in TensorBoard

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006. Both runs appear side by side, so the baseline and the overfitting demo can be compared directly.

### Command-line options

| Flag | Default | Purpose |
|---|---|---|
| `--epochs N` | 14 | Number of training epochs |
| `--batch-size N` | 64 | Training batch size |
| `--test-batch-size N` | 1000 | Validation/test batch size |
| `--lr LR` | 1.0 | Adadelta learning rate |
| `--gamma M` | 0.7 | LR decay per epoch; `1.0` turns decay off |
| `--train-subset N` | 0 (all) | Train on only the first N training images |
| `--run-name NAME` | none | Suffix for the run folder in `runs/` |
| `--log-interval N` | 10 | Batches between batch-level logs |
| `--seed S` | 1 | Random seed (also fixes the train/val split) |
| `--no-accel` | off | Force CPU |
| `--dry-run` | off | Quick single-pass check |
| `--save-model` | off | Save weights to `mnist_cnn.pt` |

## How to read the results

- **Loss and accuracy (Custom Scalars tab):** train and validation are drawn on the same chart. Curves that fall (loss) or rise (accuracy) together and then flatten mean convergence. A widening gap, with validation loss rising, means overfitting.
- **Gradient norms (`GradNorm/*`):** values near 0 suggest vanishing gradients; sudden spikes suggest exploding gradients. They normally shrink as the model converges.
- **Weight change (`WeightChange/*`):** the relative change `‖w_now − w_prev‖ / ‖w_prev‖` per layer per epoch. It falls over time, partly because of convergence and partly because the LR schedule shrinks the steps (about 0.01x by epoch 14). The overfitting demo uses `--gamma 1.0` to remove that second effect.
- **Histograms (`Weights/*`, `Gradients/*`):** show how each layer's value distribution evolves. The gradient histograms come from the last batch of each epoch, so they are a snapshot, not an average.

Things to keep in mind:

- Training accuracy is measured with dropout on, so early in training it can sit below validation accuracy. That is expected, not a bug.
- On the full dataset, dropout and LR decay keep overfitting mild. Use the overfitting demo to see it clearly.

## Troubleshooting

If `main.py` doesn't run (for example `ModuleNotFoundError: torch`), check which Python is being used:

```bash
python -c "import sys; print(sys.executable)"
```

The path should point inside `deep-learning\.venv`. If it doesn't, activate the environment as in step 1 and try again.

## Files

| Path | Contents |
|---|---|
| [main.py](main.py) | Model, training loop and TensorBoard logging |
| [model.md](model.md) | Detailed model description |
| [requirements.txt](requirements.txt) | Pinned dependencies |
| `data/` | MNIST dataset (downloaded on first run) |
| `runs/` | TensorBoard logs, one folder per run (gitignored) |
