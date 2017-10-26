import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms

#Network Architecture
#10 x 8 x 10

#random input data
#data = torch.rand(10000, 300)
#train_rand_set = data_utils.TensorDataset(data, data)
#train_loader = data_utils.DataLoader(train_rand_set, batch_size=50, shuffle=True)

train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=True, download=True,
				transform=transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.1307,), (0.3081,))
				])),
	batch_size=64, shuffle=True)

#test_dt = torch.rand(1000, 300)
#test_rand_set = data_utils.TensorDataset(test_dt, test_dt)
#test_loader = data_utils.DataLoader(test_rand_set, batch_size=1000)

test_loader = data_utils.DataLoader(
	datasets.MNIST('./data', train=False, transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))
			])),
	batch_size=100, shuffle=True)

class my_autoencoder(nn.Module):#inheriting from nn.Module
	def __init__(self, in_size, hd_size):
		#call nn.Module init
		super(my_autoencoder, self).__init__()
		self.in_size = in_size
		self.hd_size = hd_size

		#define layer in-out parameters
		self.encode = nn.Sequential(
								nn.Linear(in_size, hd_size),
								nn.ReLU(True))

		self.decode = nn.Sequential(
								nn.Linear(hd_size, in_size),
								nn.Tanh())

	def forward(self, in_vec):#foward computation
		hidden = self.encode(in_vec.view(-1, self.in_size))
		return self.decode(hidden)

class my_stacked_autoencoder(nn.Module):
	def __init__(self,in_size, form):
		super(my_stacked_autoencoder, self).__init__()

		#define layers
		self.net = []
		last_sz = in_size
		for sz in form:
			self.net.append(my_autoencoder(last_sz,sz))
			last_sz = sz

	def forward(self, in_vec):#foward computation
		sample = in_vec.view(-1, self.net[0].in_size)
		#encode
		for layer in self.net:
			sample = layer.encode(sample)
		#decode
		for layer in self.net[::-1]:
			sample = layer.decode(sample)

		return sample

def train(model, train_loader, loss_function):
	i = 0
	for layer in model.net:
		i += 1
		optimizer = optim.Adam(layer.parameters(), lr=1e-3, weight_decay=1e-5)
		for epoch in range(2):
			for batch_idx, (data, target) in enumerate(train_loader):
				vec = autograd.Variable(torch.FloatTensor(data)).view(-1, model.net[0].in_size)
				for layer2 in model.net:
					if layer2 == layer:
						break;
					vec = layer2.encode(vec)
				vec = vec.detach()
				layer.zero_grad()#clear gradients

				res = layer(vec)

				loss = loss_function(res, vec)
				loss.backward()#compute gradients
				optimizer.step()#update weights
				if batch_idx % 100 == 0:
					print('AutoEnc: {} Form:{}->{}->{}\n Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
							i, layer.in_size, layer.hd_size, layer.in_size,
							epoch, batch_idx * len(data), len(train_loader.dataset),
							100. * batch_idx / len(train_loader), loss.data[0]))

model = my_stacked_autoencoder(784,[512, 256, 128, 32, 16])

loss_function = nn.MSELoss()

train(model, train_loader, loss_function)


#test
for batch_idx, (data, target) in enumerate(test_loader):
	vec = autograd.Variable(torch.FloatTensor(data))
	res = model(vec)
	loss = loss_function(res, vec)
	print('Loss: {:.6f}'.format(loss.data[0]))

