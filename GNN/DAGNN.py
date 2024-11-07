from torch.nn import Parameter
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GNNlayer(nn.Module):
	def __init__(self,input_dimension,output_dimension,k=4):
		super(GNNlayer,self).__init__()
		self.k=k
		
		self.s=nn.Parameter(torch.ones(input_dimension,1))
		
		self.linear=nn.Linear(input_dimension,output_dimension)
		
		self.softmax=nn.Softmax(dim=-1)

	def forward(self,x,A_shape):
		D=torch.unsqueeze(torch.sum(A_shape,-1),-1)
		D=D**(-0.5)
		A=A_shape*D
		A=A*(D.permute(1,0))
		
		A_k=A
		
		H=[]
		H.append(torch.unsqueeze(x,-2))
		
		for i in range(self.k):
			
			h=A_k@x
			H.append(torch.unsqueeze(h,-2))
			A_k=A_k@A
		H=torch.cat(H,-2)
		
		S=H@self.s
		S=torch.sigmoid(S)
		S=S.transpose(-1,-2)
		
		x=S@H
		x=torch.squeeze(x)
		x=self.softmax(x)
		x=self.linear(x)
		
		return x


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
		for i,layer in enumerate(self.linear_list1):
			x=layer(x)
			if self.dropout is not None:
				x=self.Dropout(x)
			x=self.leakyrelu(x)
		
		for i,layer in enumerate(self.gnn_list):
			x=layer(x,A)
			x=self.leakyrelu(x)
		
		for i,layer in enumerate(self.linear_list2):
			x=layer(x)
		
		penalty_term=[]
		
		return x,penalty_term
