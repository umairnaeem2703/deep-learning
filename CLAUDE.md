# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
CUDA_VISIBLE_DEVICES=2 python main.py    # pin to a specific GPU
```

There is no single-test command because there are no tests.

TensorBoard is listed in `requirements.txt` but nothing in `main.py` writes TensorBoard logs yet.

## Code architecture

Everything lives in `main.py`, which is a port of the official PyTorch MNIST example:

- `Net` is the CNN: two conv layers, max-pool, two dropouts, two fully connected layers, `log_softmax` output. The `fc1` input size of 9216 is hard-coded and only matches 28x28 MNIST input after the conv/pool stack, so changing the input shape or conv hyperparameters requires updating it.
- `train()` and `test()` are plain loops that take `device` explicitly. `test()` returns nothing and only prints.
- `main()` parses CLI args, picks the device through `torch.accelerator` (falls back to CPU), builds the MNIST loaders, and runs the loop: `train` → `test` → `scheduler.step()` per epoch. The optimizer is Adadelta with a `StepLR` schedule (step size 1, gamma from `--gamma`).
- Data is downloaded to `./data` (gitignored) and the model is saved to `./mnist_cnn.pt` (not gitignored), both relative to the current working directory. Run from the repo root so they land in the expected place.
- Accelerator runs get `num_workers=1`, `persistent_workers`, and `pin_memory`. The CPU path uses default DataLoader settings.

## Things to know

- `requirements.txt` pins `torch==2.9.0+cu129`, which needs the PyTorch CUDA wheel index rather than the default PyPI index. `torchvision==0.1.6` is not a real pairing for torch 2.9 (torchvision 0.24.x matches), so this pin likely needs fixing before a fresh install works.
- `torch.accelerator` requires a recent PyTorch (2.6+). Older versions will fail at `torch.accelerator.is_available()`.
- `--dry-run` only stops inside the training loop after the first logged batch; the test pass still runs in full.
