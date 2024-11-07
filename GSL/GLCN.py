from torch.nn import Parameter
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GNNlayer(nn.Module):
	
	def __init__(self,input_dimension,output_dimension):
		
		super(GNNlayer,self).__init__()
		
		self.linear=nn.Linear(input_dimension,output_dimension)
	
	def forward(self,x,A):
		
		x=A@x
		x=self.linear(x)
		
		return x


class Model(nn.Module):
	
	def __init__(self,linear_list1,linear_list2,gnn_list,dropout=None):
		super(Model,self).__init__()
		self.dropout=dropout
		
		self.linear_list1=nn.ModuleList([nn.Linear(linear_list1[i],linear_list1[i+1]) for i in range(len(linear_list1)-1)])
		self.linear_list2=nn.ModuleList([nn.Linear(linear_list2[i],linear_list2[i+1]) for i in range(len(linear_list2)-1)])
		self.gnn_list=nn.ModuleList([GNNlayer(gnn_list[i],gnn_list[i+1]) for i in range(len(gnn_list)-1)])
		self.a=Parameter(torch.randn(linear_list1[-1]).to(device))
		
		self.softmax=nn.Softmax(dim=-1)
		self.leakyrelu = nn.LeakyReLU(0.1)
		if dropout is not None:
			self.Dropout = nn.Dropout(p=dropout)
	
	def forward(self,x,A):
		for i,layer in enumerate(self.linear_list1):
			x=layer(x)
			if self.dropout is not None:
				x=self.Dropout(x)
			x=self.leakyrelu(x)
		
		S=torch.unsqueeze(x,dim=-2)-torch.unsqueeze(x,dim=-3)
		S=S@self.a
		S=self.leakyrelu(S)
		S=torch.exp(S)
		S=S*A
		S=S/torch.unsqueeze(torch.sum(S,-1),-1)
		
		L1=torch.unsqueeze(x,dim=-2)-torch.unsqueeze(x,dim=-3)
		L1=L1**2
		L1=torch.mean(L1,-1)
		L1=L1*S
		L1=torch.mean(L1)
		
		L2=S**2
		L2=torch.mean(L2)
		
		penalty_term=[L1,L2]
		
		for i,layer in enumerate(self.gnn_list):
			x=layer(x,S)
			x=self.leakyrelu(x)
		
		for i,layer in enumerate(self.linear_list2):
			x=layer(x)
		
		return x,penalty_term

