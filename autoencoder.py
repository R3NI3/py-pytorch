import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Network Architecture
#10 x 8 x 10

#random input data
data = torch.rand(100, 10)

class my_autoencoder(nn.Module):#inheriting from nn.Module
	def __init__(self, in_size):
		#call nn.Module init
		super(my_autoencoder, self).__init__()

		#define layer in-out parameters
		self.net = nn.Sequential(
								nn.Linear(in_size, 8),
								nn.ReLU(True),
								nn.Linear(8, in_size),
								nn.Tanh())

	def forward(self, in_vec):#foward computation
		return self.net(in_vec)

model = my_autoencoder(10)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
losses = []

for epoch in range(300):
	total_loss = torch.Tensor([0])
	for dt in data:
		model.zero_grad()#clear gradients (not using batch)

		vec = autograd.Variable(torch.FloatTensor(dt))
		res = model(vec)

		loss = loss_function(res, vec)
		loss.backward()#compute gradients
		optimizer.step()#update weights

	print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 100, loss.data[0]))


#test
test_data = torch.rand(10)
vec = autograd.Variable(torch.FloatTensor(test_data))
res = model(vec)
print([test_data,res])

