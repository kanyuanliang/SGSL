from torch.nn import Parameter
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GNNlayer(nn.Module):
	def __init__(self,input_dimension,output_dimension,k=2):
		super(GNNlayer,self).__init__()
		self.linear1=nn.Linear(input_dimension,output_dimension)
		self.theta=Parameter(torch.randn(k).to(device))
		
		self.k=k

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

	def Poly(self,A):
		theta=torch.sigmoid(self.theta)
		
		nodesize=len(A[-1])
		I=torch.eye(nodesize).to(device)
		
		seg=I
		P=I
		
		for i in range(self.k):
			seg=seg@A
			P=P+seg*theta[i]
		
		return P
	
	def forward(self,x,A,k=4):
		S=self.Laplacian(A)
		S=self.Poly(S)
		S=self.Laplacian(S)
		K=torch.argsort(S,descending=True)
		K=torch.argsort(K)
		K=torch.where(K<k,1,0)
		S=S*K
		
		x=S@x
		x=self.linear1(x)
		
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

