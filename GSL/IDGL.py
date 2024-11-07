from torch.nn import Parameter
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GNNlayer(nn.Module):
	def __init__(self,input_dimension,output_dimension,m=4):
		super(GNNlayer,self).__init__()
		self.linear=nn.Linear(input_dimension,output_dimension)
		self.w=Parameter(torch.randn(m,input_dimension).to(device))
		
		self.m=m
		self.I_d=torch.eye(input_dimension).to(device)

	def Laplacian(self,A):
		nodesize=len(A[-1])
		I=torch.eye(nodesize).to(device)
		
		D=torch.sum(A,-1)
		D=torch.unsqueeze(D,-1)
		D=D**0.5
		D=1/D
		D=D*I
		
		L=D@A@D
		
		return L
	
	def cos(self,x):
		s=x**2
		s=torch.sum(s,-1)
		s=s**0.5
		s=torch.unsqueeze(s,dim=-1)
		s=x/s
		s=s@(s.transpose(-1,-2))
		
		return s
	
	def regularization(self,A,x):
		nodesize=len(A[-1])
		I=torch.eye(nodesize).to(device)

		D=torch.sum(A,-1)
		D=torch.unsqueeze(D,-1)
		D=D*I
		
		L1=x.transpose(-2,-1)@(D-A)@x
		L1=L1*self.I_d
		L1=L1**2
		L1=torch.mean(L1)
		
		L2=torch.sum(A,-1)
		L2=torch.log(L2)
		L2=torch.mean(L2)
		L2=-1*L2
		
		L3=A**2
		L3=torch.mean(L3)
		
		penalty_term=[L1,L2,L3]
		
		return penalty_term
	
	def forward(self,x,A,k=4,lam=0.5):
		s=torch.unsqueeze(x,dim=-3)
		w=torch.unsqueeze(self.w,dim=-2)
		s=s*w
		s=self.cos(s)
		s=torch.mean(s,-3)

		S_sort=torch.argsort(s,descending=True)
		S_sort=torch.argsort(S_sort)
		S_sort=torch.where(S_sort<k,1,0)
		s=S_sort*s
		s=s/(torch.unsqueeze(torch.sum(s,-1),dim=-1)+1e-10)
		
		L=self.Laplacian(A)
		
		A=lam*L+(1-lam)*s
		
		penalty_term=self.regularization(A,x)
		
		x=A@x
		x=self.linear(x)
		
		return x,penalty_term


class Model(nn.Module):
	def __init__(self,linear_list1,linear_list2,gnn_list,dropout=None):
		super(Model,self).__init__()
		self.dropout=dropout
		self.linear_list1=nn.ModuleList([nn.Linear(linear_list1[i],linear_list1[i+1]) for i in range(len(linear_list1)-1)])
		self.linear_list2=nn.ModuleList([nn.Linear(linear_list2[i],linear_list2[i+1]) for i in range(len(linear_list2)-1)])
		self.gnn_list=nn.ModuleList([GNNlayer(gnn_list[i],gnn_list[i+1]) for i in range(len(gnn_list)-1)])
		
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
		
