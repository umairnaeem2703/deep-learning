# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This is a class-assignment project: PyTorch MNIST digit classification, working in `mnist/`. Only `mnist/` matters here. The goal is to monitor training: train vs. validation loss and accuracy, gradient magnitudes, how weights and gradients change between epochs, and whether the model converges, stalls, or overfits.

## Commands

Run from `mnist/`. There is no test suite or linter.

```bash
pip install -r requirements.txt     # torch, torchvision
python main.py                      # full run, 14 epochs
python main.py --epochs 1 --dry-run # quick smoke test (breaks after the first logged batch)
python main.py --no-accel           # force CPU
python main.py --save-model         # writes mnist_cnn.pt
```

## Architecture

Everything is in `mnist/main.py`:

- `Net`: 2 conv layers (`conv1`, `conv2`), max-pool, dropout (0.25), `fc1` (9216→128), dropout (0.5), `fc2` (128→10), `log_softmax` output. Pair it with `F.nll_loss`.
- `train()` runs one epoch and prints loss every `--log-interval` batches. It does not track accuracy.
- `test()` evaluates on the MNIST **test** set once per epoch and prints average loss and accuracy. Nothing is returned or stored.
- `main()` uses Adadelta (`lr=1.0`) with `StepLR(step_size=1, gamma=0.7)`, so the learning rate decays every epoch. Keep that in mind when judging "stalling".

## Things to know

- There is **no validation split**: `test()` is the only held-out evaluation. Monitoring for overfitting needs a split from the training set (e.g. 55k/5k) so the test set stays untouched.
- MNIST downloads to `../data` (relative to `mnist/`), i.e. the repo-root `data/` folder.
- Dropout is active in `model.train()` and off in `model.eval()`, so training loss/accuracy computed during training is not directly comparable to evaluation numbers. For a fair comparison, re-evaluate the training set in eval mode.

## Work in progress

`mnist/main.py` has an uncommitted gradient-norm printout after `loss.backward()`. It prints the global L2 gradient norm every 100 batches, alongside the loss.
