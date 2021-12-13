import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm

# W10 Modern ConvNets.pdf - 23 page
# https://pytorch.org/assets/images/resnet.png

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_batch_norm=False):
        super(BasicBlock, self).__init__()

        self.use_batch_norm = use_batch_norm

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet34(nn.Module):
    def __init__(self, input_channel, output_dim, learning_rate, reg_lambda, device, use_batch_norm=False, use_mixup=False, mixup_alpha=0.2):
        super(ResNet34, self).__init__()

        self.output_dim = output_dim
        self.device = device
        self.use_batch_norm = use_batch_norm
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        self.CONV1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=(7, 7), stride=2, padding=3)
        self.BN1 = nn.BatchNorm2d(64)
        self.POOL1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)


        self.relu = nn.ReLU(inplace=True)
        self.CONV2 = nn.Conv2d(in_channels=64, out_channels=64,kernel_size=(3,3),stride=1,padding=1)
        self.CONV3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(3,3),stride=4,padding=1)
        self.CONV4 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(3,3),stride=6,padding=1)
        self.CONV5 = nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(3,3),stride=3,padding=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * 1, 10)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=reg_lambda)


    def forward(self, x):
        out = torch.zeros((x.shape[0], self.output_dim))


        h = self.CONV1(x)
        h = self.BN1(h)
        h = self.relu(h)
        h = self.POOL1(h)
        h=self.CONV2(h)
        h=self.CONV3(h)
        h=self.CONV4(h)
        h=self.CONV5(h)
        h=self.avg_pool(h)
        out= h.reshape(x.shape[0],-1)
        out = self.fc(out)

        return out

    def predict(self, data_loader):
        self.eval()
        correct_y = []
        pred_y = []
        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                batch_x, batch_y = batch_data
                pred = self.forward(batch_x.to(self.device))
                _, predicted = torch.max(pred.data, 1)

                correct_y.append(batch_y.numpy())
                pred_y.append(predicted.cpu().numpy())
        correct_y = np.concatenate(correct_y, axis=0)
        pred_y = np.concatenate(pred_y, axis=0)
        return correct_y, pred_y

    def train_(self, trainloader, validloader, num_epochs, test_every=10, print_every=10):
        self.train_accuracy = []
        self.valid_accuracy = []
        best_epoch = -1
        best_acc = -1
        self.num_epochs = num_epochs
        self.test_every = test_every
        self.print_every = print_every

        total = 0
        correct = 0
        self.train()
        for epoch in range(1, num_epochs+1):
            start = time.time()
            epoch_loss = 0.0
            # model Train
            for b, batch_data in enumerate(tqdm(trainloader, desc="Training")):
                batch_x, batch_y = batch_data
                if self.use_mixup == False:
                    pred_y = self.forward(batch_x.to(self.device))
                    loss = self.loss_function(pred_y, batch_y.to(self.device))
                else:
                    mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=self.mixup_alpha, use_cuda=True if torch.cuda.is_available() else False)
                    pred_y = self.forward(mixed_x.to(self.device))
                    loss = self.mixup_criterion(pred_y, y_a.to(self.device), y_b.to(self.device), lam)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss

                _, predicted = torch.max(pred_y.data, 1)
                total += batch_y.size(0)
                correct += (predicted.cpu() == batch_y).sum().item()

            epoch_loss /= len(trainloader)
            end = time.time()
            lapsed_time = end - start

            if epoch % print_every == 0:
                print(f'Epoch {epoch} took {lapsed_time} seconds\n')
                print('[EPOCH %d] Loss = %.5f' % (epoch, epoch_loss))

            if epoch % test_every == 0:
                # TRAIN ACCURACY
                train_acc = correct / total
                self.train_accuracy.append(train_acc)

                # VAL ACCURACY
                real_y, pred_y = self.predict(validloader)
                correct = (pred_y == real_y).sum().item()
                total = len(pred_y)
                valid_acc = correct / total
                self.valid_accuracy.append(valid_acc)

                if best_acc < valid_acc:
                    best_acc = valid_acc
                    best_epoch = epoch
                    torch.save(self.state_dict(), './best_model/ResNet34.pt')
                if epoch % print_every == 0:
                    print('Train Accuracy = %.3f' % train_acc + ' // ' + 'Valid Accuracy = %.3f' % valid_acc)
                    if best_acc < valid_acc:
                        print('Best Accuracy updated (%.4f => %.4f)' % (best_acc, valid_acc))
        print('Training Finished...!!')
        print('Best Valid acc : %.2f at epoch %d' % (best_acc, best_epoch))
        
        return best_acc

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.loss_function(pred, y_a) + (1 - lam) * self.loss_function(pred, y_b)

    def restore(self):
        with open(os.path.join('./best_model/ResNet34.pt'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    def plot_accuracy(self):
        """
            Draw a plot of train/valid accuracy.
            X-axis : Epoch
            Y-axis : train_accuracy & valid_accuracy
            Draw train_acc-epoch, valid_acc-epoch graph in 'one' plot.
        """
        epochs = list(np.arange(1, self.num_epochs+1, self.print_every, dtype=np.int32))

        plt.plot(epochs, self.train_accuracy, label='Train Acc.')
        plt.plot(epochs, self.valid_accuracy, label='Valid Acc.')

        plt.title('Epoch - Train/Valid Acc.')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig('ResNet34.png')
        plt.show()

# https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
