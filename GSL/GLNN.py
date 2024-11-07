from torch.nn import Parameter
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Search_A(nn.Module):
	def __init__(self,nodesize):
		super(Search_A,self).__init__()
		self.A=nn.Parameter(torch.empty(nodesize,nodesize).to(device))
		nn.init.xavier_uniform_(self.A)
	
	def forward(self):
		out_A=self.A+self.A.transpose(-2,-1)
		
		return out_A


class GNNlayer(nn.Module):
	def __init__(self,input_dimension,output_dimension,nodesize):
		super(GNNlayer,self).__init__()
		self.search_A=Search_A(nodesize)
		
		self.linear0=nn.Linear(input_dimension,output_dimension,bias=False)
		self.linear1=nn.Linear(output_dimension,output_dimension,bias=False)
		
		self.I_d=torch.eye(input_dimension).to(device)
		self.I_N=torch.eye(nodesize).to(device)
		
		self.nodesize=nodesize
		
		self.leakyrelu = nn.LeakyReLU(0.1)
		self.softmax=nn.Softmax(dim=-1)
		
	def regularization(self,out_A,in_A,x):
		GLR=x.transpose(-2,-1)@(self.I_N-out_A)@x
		GLR=GLR*self.I_d
		GLR=GLR**2
		GLR=torch.mean(GLR)
		
		sparsity=torch.abs(out_A)
		sparsity=torch.mean(sparsity)
		
		properties=torch.sum(out_A,dim=-1)
		properties=properties-1
		properties=properties**2
		properties=torch.mean(properties)
		
		in_A=in_A/torch.unsqueeze(torch.sum(in_A,-1),-1)
		
		gt=out_A-in_A
		gt=gt**2
		gt=torch.mean(gt)
		
		penalty_term=[GLR,sparsity,properties,gt]
		
		return penalty_term
		
	def forward(self,x,in_A):
		out_A=self.search_A()
		
		penalty_term=self.regularization(out_A,in_A,x)
		
		x=self.linear0(out_A@x)
		x=self.leakyrelu(x)
		
		x=self.linear1(out_A@x)
		x=self.leakyrelu(x)
		
		return x,penalty_term


class Model(nn.Module):
	def __init__(self,linear_list1,linear_list2,gnn_list,nodesize,dropout=None):
		super(Model,self).__init__()
		self.dropout=dropout
		
		self.linear_list1=nn.ModuleList([nn.Linear(linear_list1[i],linear_list1[i+1]) for i in range(len(linear_list1)-1)])
		self.linear_list2=nn.ModuleList([nn.Linear(linear_list2[i],linear_list2[i+1]) for i in range(len(linear_list2)-1)])
		self.gnn_list=nn.ModuleList([GNNlayer(gnn_list[i],gnn_list[i+1],nodesize) for i in range(len(gnn_list)-1)])
		
		self.softmax=nn.Softmax(dim=-1)
		self.leakyrelu = nn.LeakyReLU(0.1)
		if dropout is not None:
			self.Dropout = nn.Dropout(p=dropout)
	
	def forward(self,x,A):
		penalty_term=[]
		
		for i,layer in enumerate(self.linear_list1):
			x=layer(x)
			if self.dropout is not None:
				x=self.Dropout(x)
			x=self.leakyrelu(x)
		
		for i,layer in enumerate(self.gnn_list):
			x,penalty=layer(x,A)
			x=self.leakyrelu(x)
			penalty_term.extend(penalty)
		
		for i,layer in enumerate(self.linear_list2):
			x=layer(x)
		
		return x,penalty_term
