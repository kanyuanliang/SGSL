from torch.nn import Parameter
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

#from GNN.GAT import GNNlayer
from GNN.GCN import GNNlayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SGSL(nn.Module):
	def __init__(self,dimension,nodesize,invariant):
		super(SGSL,self).__init__()
		self.invariant=invariant
		self.nodesize=nodesize
		
		if invariant:
			self.optimized_A=Parameter(torch.rand(nodesize,nodesize).to(device))
		else:
			self.w1=Parameter(torch.randn(dimension,dimension).to(device))
			self.w2=Parameter(torch.randn(dimension,dimension).to(device))
			self.a=Parameter(torch.randn(dimension).to(device))
		self.alpha=Parameter(torch.zeros(1).to(device))
		
		self.softmax=nn.Softmax(dim=-1)
		self.leakyrelu = nn.LeakyReLU(0.1)
	
	def normlized(self,A):
		A=A/torch.unsqueeze(torch.sum(A,-1),-1)
		return A
	
	def S(self,x):
		num1=torch.exp(torch.tensor(1))**(-4)
		num2=torch.exp(torch.tensor(1))**(4)
		x=torch.where(x>num1,torch.log(x)+5,num2*x)
		return x
	
	def regularization(self,A):
		A=self.S(A)
		A=torch.mean(A)
		return A
		
	def forward(self,x,A,mask):
		alpha=torch.sigmoid(self.alpha)
		
		if self.invariant:
			if mask is not None:
				optimized_A=torch.exp(optimized_A)
				optimized_A=optimized_A*(mask+1e-32)
				optimized_A=self.normlized(optimized_A)
			else:
				optimized_A=self.softmax(self.optimized_A)
			
			penalty_term=[self.regularization(optimized_A)]
			
		else:
			x1=x@self.w1
			x2=x@self.w2
			x1=torch.unsqueeze(x1,-2)
			x2=torch.unsqueeze(x2,-3)
			
			optimized_A=x1+x2
			optimized_A=self.leakyrelu(optimized_A)
			optimized_A=optimized_A@self.a
			optimized_A=torch.tanh(optimized_A/8)*8
			
			if mask is not None:
				optimized_A=torch.exp(optimized_A)
				optimized_A=optimized_A*(mask+1e-32)
				optimized_A=self.normlized(optimized_A)
			else:
				optimized_A=self.softmax(self.optimized_A)
			
			penalty_term=[self.regularization(optimized_A)]
		
		optimized_A=(1-alpha)*self.normlized(A)+alpha*optimized_A
		
		return optimized_A,penalty_term


class Model(nn.Module):
	def __init__(self,linear_list1,linear_list2,gnn_list,nodesize,dropout=None,invariant=True):
		super(Model,self).__init__()
		self.invariant=invariant
		self.dropout=dropout
		if len(gnn_list)>4 or invariant==True:
			self.over_multilayer_GNN=True
		else:
			self.over_multilayer_GNN=False
		
		self.linear_list1=nn.ModuleList([nn.Linear(linear_list1[i],linear_list1[i+1]) for i in range(len(linear_list1)-1)])
		self.linear_list2=nn.ModuleList([nn.Linear(linear_list2[i],linear_list2[i+1]) for i in range(len(linear_list2)-1)])
		self.gnn_list=nn.ModuleList([GNNlayer(gnn_list[i],gnn_list[i+1]) for i in range(len(gnn_list)-1)])
		if self.over_multilayer_GNN:
			self.sgsl_list=SGSL(gnn_list[0],nodesize,invariant)
		else:
			self.sgsl_list=nn.ModuleList([SGSL(gnn_list[i],nodesize,invariant) for i in range(len(gnn_list)-1)])
		
		self.softmax=nn.Softmax(dim=-1)
		self.leakyrelu = nn.LeakyReLU(0.1)
		if dropout is not None:
			self.Dropout = nn.Dropout(p=dropout)
	
	def forward(self,x,A,mask=None):
		penalty_term=[]
		
		for i,layer in enumerate(self.linear_list1):
			x=layer(x)
			if self.dropout is not None:
				x=self.Dropout(x)
			x=self.leakyrelu(x)
		
		if self.over_multilayer_GNN:
			optimized_A,penalty_term=self.sgsl_list(x,A,mask)

		for i,layer in enumerate(self.gnn_list):
			if self.over_multilayer_GNN is False:
				optimized_A,penalty=self.sgsl_list[i](x,A,mask)
				penalty_term.extend(penalty)
			x=layer(x,optimized_A)
			x=self.leakyrelu(x)

		for i,layer in enumerate(self.linear_list2):
			x=layer(x)
		
		return x,penalty_term
