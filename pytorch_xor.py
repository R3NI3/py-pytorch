import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#A xor B = Label
data = [([0,0],0),
		([0,1],1),
		([1,0],1),
		([1,1],0)]

test_data = data

class my_xor(nn.Module):#inheriting from nn.Module
	def __init__(self, n_labels, in_size):
		#call nn.Module init
		super(my_xor, self).__init__()

		#define layer in-out parameters
		self.linear1 = nn.Linear(in_size, 2)
		self.linear2 = nn.Linear(2, n_labels)


	def forward(self, in_vec):#foward computation
		return self.linear2(F.sigmoid(self.linear1(in_vec)))

label = torch.FloatTensor([0,1])

model = my_xor(1,2)

loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.2)

for epoch in range(1000):
	for instance, label in data:
		model.zero_grad()#clear gradients (not using batch)

		vec = autograd.Variable(torch.FloatTensor(instance))
		target = autograd.Variable(torch.FloatTensor([label]))
		clss = model(vec)

		loss = loss_function(clss, target)
		loss.backward()#compute gradients
		optimizer.step()#update weights

#test
for instance, label in test_data:
	vec = autograd.Variable(torch.FloatTensor(instance))
	clss = model(vec)
	print([instance,clss])




