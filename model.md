# Model description

This describes the `Net` class in [main.py](main.py): a small convolutional neural network (CNN) that classifies 28x28 grayscale MNIST digits into 10 classes (0-9).

## Summary

| Property | Value |
|---|---|
| Type | Convolutional neural network (CNN) |
| Input | 1 x 28 x 28 grayscale image |
| Output | 10 log-probabilities (one per digit) |
| Learnable layers | 4 (2 convolutional + 2 fully connected) |
| Other layers | 1 max-pool, 2 dropout, 1 flatten |
| Hidden activation | ReLU |
| Output activation | log-softmax |
| Trainable parameters | 1,199,882 |

## Layer-by-layer

Shapes are `channels x height x width` for one image (the batch dimension is left out).

| # | Layer | Code | Output shape | Parameters |
|---|---|---|---|---|
| 0 | Input | | 1 x 28 x 28 | 0 |
| 1 | Convolution | `conv1 = Conv2d(1, 32, 3, 1)` | 32 x 26 x 26 | 320 |
| 2 | ReLU | `F.relu` | 32 x 26 x 26 | 0 |
| 3 | Convolution | `conv2 = Conv2d(32, 64, 3, 1)` | 64 x 24 x 24 | 18,496 |
| 4 | ReLU | `F.relu` | 64 x 24 x 24 | 0 |
| 5 | Max-pool | `F.max_pool2d(x, 2)` | 64 x 12 x 12 | 0 |
| 6 | Dropout (p=0.25) | `dropout1` | 64 x 12 x 12 | 0 |
| 7 | Flatten | `torch.flatten(x, 1)` | 9216 | 0 |
| 8 | Fully connected | `fc1 = Linear(9216, 128)` | 128 | 1,179,776 |
| 9 | ReLU | `F.relu` | 128 | 0 |
| 10 | Dropout (p=0.5) | `dropout2` | 128 | 0 |
| 11 | Fully connected | `fc2 = Linear(128, 10)` | 10 | 1,290 |
| 12 | Log-softmax | `F.log_softmax(x, dim=1)` | 10 | 0 |

### Details

- **Convolutions:** both use a 3x3 kernel, stride 1 and no padding, so each one shrinks the image by 2 pixels per side (28 -> 26 -> 24). `conv1` goes from 1 to 32 channels, `conv2` from 32 to 64.
- **Max-pool:** a 2x2 window with stride 2 halves the spatial size (24 -> 12). It has no parameters.
- **Flatten:** 64 x 12 x 12 = 9216 values per image. This is why `fc1` has a hard-coded input size of 9216; changing the input shape or the conv/pool settings means updating it.
- **Dropout:** randomly zeroes 25% of the pooled feature map values and 50% of the 128 hidden units during training. It is switched off automatically by `model.eval()` (used for validation and test).
- **Output:** `log_softmax` turns the 10 raw scores (logits) into log-probabilities. The predicted digit is the `argmax`.

### Parameter count

| Layer | Weights | Biases | Total |
|---|---|---|---|
| `conv1` | 32 x 1 x 3 x 3 = 288 | 32 | 320 |
| `conv2` | 64 x 32 x 3 x 3 = 18,432 | 64 | 18,496 |
| `fc1` | 9216 x 128 = 1,179,648 | 128 | 1,179,776 |
| `fc2` | 128 x 10 = 1,280 | 10 | 1,290 |
| **Total** | | | **1,199,882** |

About 98% of the parameters sit in `fc1`, so that layer dominates both the model size and the overfitting risk (hence the 0.5 dropout after it).

These layers appear in TensorBoard under names such as `conv1.weight`, `conv1.bias`, `fc1.weight` (see `GradNorm/<param>`, `Weights/<param>`, `Gradients/<param>`, `WeightChange/<param>`).

## Loss function

`F.nll_loss` (negative log-likelihood) on the log-softmax output. Together, log-softmax + NLL is equivalent to cross-entropy loss. The validation and test passes use `reduction='sum'` and divide by the dataset size to get the average.

## Training setup

| Setting | Value |
|---|---|
| Optimizer | Adadelta, `lr=1.0` (`--lr`) |
| LR schedule | `StepLR(step_size=1, gamma=0.7)`: LR is multiplied by 0.7 every epoch (`--gamma`) |
| Epochs | 14 (`--epochs`) |
| Batch size | 64 train, 1000 validation/test |
| Weight initialization | PyTorch defaults (Kaiming-uniform for conv and linear layers); seed 1 |
| Regularization | Dropout only (no weight decay, no batch norm) |

## Data

| Item | Value |
|---|---|
| Dataset | MNIST (downloaded to `./data`) |
| Preprocessing | `ToTensor()` (pixels scaled to 0-1), then `Normalize(mean=0.1307, std=0.3081)` |
| Train / validation split | 55,000 / 5,000 images from the 60k training set, fixed seed |
| Test set | 10,000 images, evaluated once after training |
| Augmentation | None |
