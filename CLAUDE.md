# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goals

This is an educational project for a class assignment: MNIST handwritten-digit classification in PyTorch. The point is to watch and understand training, not to reach state-of-the-art accuracy. The assignment asks us to monitor:

| Goal | Where it shows up in TensorBoard |
|---|---|
| Training loss and validation loss | Custom Scalars → Loss (`Loss/train`, `Loss/val`); also `Loss/train_batch` |
| Training accuracy and validation accuracy | Custom Scalars → Accuracy (`Accuracy/train`, `Accuracy/val`) |
| Gradient magnitudes during backpropagation | Scalars → `GradNorm/<param>` and `GradNorm/total` |
| How weights and gradients change between epochs | Histograms → `Weights/<param>`, `Gradients/<param>`; Scalars → `WeightChange/<param>` |
| Whether the model converges, stalls, or overfits | The train/val overlays, `LearningRate`, and the per-epoch status line printed to the terminal |

## Working rules

- **Keep the code simple.** Everything stays in `main.py`, with plain loops and small helper functions. Don't add new frameworks or dependencies; TensorBoard is already in `requirements.txt`. Readability matters more than speed or cleverness. Add short comments where they teach something.
- **Log to `runs/`.** All metrics go through `torch.utils.tensorboard.SummaryWriter`, one subfolder per run (`runs/<date>_<host>_<run-name>`). `runs/` is gitignored.
- **Explain each change.** When editing code, say what changed, why, and what to look for in TensorBoard as a result. Assume the reader is learning.

## Repository state

This is a small single-script PyTorch MNIST example. There is no package layout, test suite, linter, formatter, or CI config. Don't assume one exists; check before adding commands to docs or scripts.

## Commands

Setup (the README's install line):

```bash
pip install -r requirements.txt
```

Run training with defaults (14 epochs, CPU or accelerator auto-detected):

```bash
python main.py
```

Useful flags for quick checks and experiments:

```bash
python main.py --dry-run          # breaks after the first logged batch per epoch
python main.py --no-accel         # force CPU
python main.py --epochs 1 --save-model   # short run that writes mnist_cnn.pt
python main.py --run-name baseline       # names the run folder in runs/
CUDA_VISIBLE_DEVICES=2 python main.py    # pin to a specific GPU
```

Overfitting demo (train on 1000 images, no learning-rate decay, many epochs):

```bash
python main.py --train-subset 1000 --gamma 1.0 --epochs 30 --run-name overfit
```

View the logs (then open http://localhost:6006):

```bash
tensorboard --logdir runs
```

There is no single-test command because there are no tests.

## Code architecture

Everything lives in `main.py`, which started as a port of the official PyTorch MNIST example:

- `Net` is the CNN: two conv layers, max-pool, two dropouts, two fully connected layers, `log_softmax` output. The `fc1` input size of 9216 is hard-coded and only matches 28x28 MNIST input after the conv/pool stack, so changing the input shape or conv hyperparameters requires updating it.
- `train()` runs one epoch. Every `--log-interval` batches it prints the batch loss, logs `Loss/train_batch`, and calls `log_gradient_norms()`. It returns the epoch's average train loss and accuracy.
- `evaluate()` runs the model on any loader without gradients and returns `(loss, accuracy)`. It is used for the validation set every epoch and for the test set once at the end.
- `log_gradient_norms()` logs the L2 norm of each parameter's gradient, plus the total, at a global batch step.
- `log_weights()` runs once per epoch. It logs weight and gradient histograms, and the relative weight change `‖w_now − w_prev‖ / ‖w_prev‖` against a snapshot taken before the epoch.
- `main()` parses CLI args, picks the device through `torch.accelerator` (falls back to CPU), and builds the data. The 60k MNIST training images are split 55k train / 5k validation with a fixed seed; `--train-subset N` keeps only the first N training images. The 10k test set is untouched until the end. The loop per epoch is `train` → `evaluate(val)` → log → `scheduler.step()`. The optimizer is Adadelta with a `StepLR` schedule (step size 1, gamma from `--gamma`).
- Data is downloaded to `./data`, TensorBoard logs go to `./runs`, and the model is saved to `./mnist_cnn.pt` (not gitignored), all relative to the current working directory. Run from the repo root so they land in the expected place.
- The train loader always shuffles. Accelerator runs also get `num_workers=1`, `persistent_workers`, and `pin_memory` on all loaders.

## Things to know

- `requirements.txt` pins `torch==2.9.0+cu129` and includes the PyTorch cu129 `--extra-index-url`, so `pip install -r requirements.txt` pulls the CUDA wheel.
- `torch.accelerator` requires a recent PyTorch (2.6+). Older versions will fail at `torch.accelerator.is_available()`.
- `--dry-run` only stops inside the training loop after the first logged batch; validation and test passes still run in full.
- Train accuracy is measured with dropout on, so early in training it can sit below validation accuracy. That's expected, not a bug.
- `StepLR` multiplies the learning rate by `--gamma` (0.7 by default) every epoch, reaching about 0.01× by epoch 14. Shrinking `WeightChange` is partly caused by this schedule, not only by convergence. Use `--gamma 1.0` to turn the decay off.
- The `Gradients/*` histograms come from the last batch of each epoch, so they're a snapshot, not an average.
- On full MNIST with dropout and LR decay, overfitting is mild. Use the overfitting demo command above to see it clearly.
