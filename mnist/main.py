import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, Subset
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def log_gradient_norms(model, writer, step):
    # L2 norm of each layer's gradient from the last backward pass.
    # Values near 0 suggest vanishing gradients; sudden spikes suggest exploding ones.
    total_sq = 0.0
    for name, param in model.named_parameters():
        norm = param.grad.norm().item()
        writer.add_scalar('GradNorm/' + name, norm, step)
        total_sq += norm ** 2
    writer.add_scalar('GradNorm/total', total_sq ** 0.5, step)


def log_weights(model, writer, epoch, prev_weights):
    for name, param in model.named_parameters():
        writer.add_histogram('Weights/' + name, param, epoch)
        writer.add_histogram('Gradients/' + name, param.grad, epoch)  # from the epoch's last batch
        # How far this layer moved during the epoch, relative to its size
        change = (param.detach() - prev_weights[name]).norm() / prev_weights[name].norm()
        writer.add_scalar('WeightChange/' + name, change.item(), epoch)


def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    total_loss = 0
    correct = 0
    seen = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # Running totals for the epoch's average loss and accuracy (dropout is on here)
        total_loss += loss.item() * len(data)
        correct += output.argmax(dim=1).eq(target).sum().item()
        seen += len(data)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss.item(), step)
            # optimizer.step() doesn't touch .grad, so these are still this batch's gradients
            log_gradient_norms(model, writer, step)
            if args.dry_run:
                break
    return total_loss / seen, 100. * correct / seen


def evaluate(model, device, loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    n = len(loader.dataset)
    return total_loss / n, 100. * correct / n


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    parser.add_argument('--train-subset', type=int, default=0, metavar='N',
                        help='train on only the first N training images, to make overfitting visible (default: 0 = all)')
    parser.add_argument('--run-name', type=str, default='', metavar='NAME',
                        help='suffix for the TensorBoard run folder in runs/')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    # Only training data needs shuffling; evaluation order doesn't change the result
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    # Hold out 5k training images for validation so the test set is only used once, at the end.
    # The fixed generator keeps the split identical between runs.
    train_set, val_set = random_split(dataset1, [55000, 5000],
                                      generator=torch.Generator().manual_seed(args.seed))
    if args.train_subset:
        # A tiny training set is easy to memorize, which makes overfitting visible
        train_set = Subset(train_set, range(args.train_subset))
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Logs go to runs/<date>_<host>_<run-name>; view them with: tensorboard --logdir runs
    writer = SummaryWriter(comment='_' + args.run_name if args.run_name else '')
    # Draw train and val on the same chart (Custom Scalars tab) so the gap between them is easy to see.
    # Tags here are regexes, so ^...$ keeps 'Loss/train' from also matching 'Loss/train_batch'.
    writer.add_custom_scalars({'Train vs val': {
        'Loss': ['Multiline', ['^Loss/train$', '^Loss/val$']],
        'Accuracy': ['Multiline', ['^Accuracy/train$', '^Accuracy/val$']],
    }})

    best_val_loss = float('inf')
    epochs_since_best = 0
    prev_train_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        # Snapshot the weights so log_weights() can measure how much they moved this epoch
        prev_weights = {name: param.detach().clone() for name, param in model.named_parameters()}
        lr = scheduler.get_last_lr()[0]

        train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, writer)
        val_loss, val_acc = evaluate(model, device, val_loader)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LearningRate', lr, epoch)
        log_weights(model, writer, epoch, prev_weights)

        # Rough hint only; the TensorBoard charts are the real evidence
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best = 0
            status = 'improving'
        else:
            epochs_since_best += 1
            status = 'overfitting?' if train_loss < prev_train_loss else 'stalled'
        prev_train_loss = train_loss

        print('\nEpoch {}: train loss {:.4f}, acc {:.2f}% | val loss {:.4f}, acc {:.2f}% | lr {:.4g} | '
              '{} ({} epochs since best val loss)\n'.format(
                  epoch, train_loss, train_acc, val_loss, val_acc, lr, status, epochs_since_best))
        scheduler.step()

    # The test set is evaluated once, after all training decisions are made
    test_loss, test_acc = evaluate(model, device, test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_acc))
    writer.add_scalar('Loss/test', test_loss, args.epochs)
    writer.add_scalar('Accuracy/test', test_acc, args.epochs)
    writer.close()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
