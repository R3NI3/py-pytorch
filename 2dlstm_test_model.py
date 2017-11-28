import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable, Function
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from mdlstm import mdlstm
import os


has_cuda = torch.cuda.is_available()

path_resume = './save_model/trained_model_mdlstm_fc.pth.tar'
directory = os.path.dirname(path_resume)
if not os.path.exists(directory):
    os.makedirs(directory)

batch_size = 5
test_batch_size = 10000
log_interval = 100
epochs = 100

kwargs = {'num_workers': 1, 'pin_memory': True} if has_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=test_batch_size, shuffle=True, **kwargs)

class my_test_model(nn.Module):
    def __init__(self, feature_sz, hidden_sz, fc_hd_sz, output_sz):
        super(my_test_model, self).__init__()
        self.feature_sz = feature_sz
        self.hidden_sz = hidden_sz
        self.my_mdlstm = mdlstm(feature_sz, hidden_sz, [28, 28])
        self.fc1 = nn.Linear(feature_sz * hidden_sz,fc_hd_sz)
        self.fc2 = nn.Linear(fc_hd_sz, output_sz)
        self.softmax = nn.Softmax()

    def forward(self, input):
        data = input.view(-1, 784)
        x = self.my_mdlstm(data)
        x = F.relu(self.fc1(x))
        return self.softmax(self.fc2(x))

model = my_test_model(784, 5, 200, 10)
if has_cuda:
    model.cuda()

criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(data)
        target = Variable(target)
        if has_cuda:
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        data = data.view(-1,784)
        y_pred = model(data)

        loss = criterion(y_pred, target)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, target) in enumerate(test_loader):
        if has_cuda:
            data = data.cuda()
            target = target.cuda()
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        y_pred = model(data)
        test_loss += criterion(y_pred, target).data[0]
        pred = y_pred.data.max(1)[1]
        d = pred.eq(target.data).cpu()
        accuracy = d.sum()/d.size()[0]
        print('====> Test Epoch:{} Accuracy {}'.format(epoch, accuracy))

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def main():
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)

    torch.save({
            'epoch': epochs + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}, path_resume)


if __name__ == "__main__":
    main()
