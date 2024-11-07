from torch.nn import Parameter
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CPAlayer(nn.Module):
	def __init__(self,input_dimension,output_dimension):
		super(CPAlayer,self).__init__()
		self.w=Parameter(torch.randn(input_dimension,output_dimension).to(device))
		self.a=Parameter(torch.randn(output_dimension).to(device))
		
		self.leakyrelu = nn.LeakyReLU(0.1)

	def forward(self,x,A_shape):
		y=x@self.w
		x1=torch.unsqueeze(y,-2)
		x2=torch.unsqueeze(y,-3)
		
		A=x1+x2
		A=A@self.a
		
		A=torch.tanh(A/8)*8
		
		A=self.leakyrelu(A)
		
		A=torch.exp(A)*A_shape
		A=A/torch.unsqueeze(torch.sum(A,-1),-1)
		A=A*(torch.unsqueeze(torch.sum(A_shape,-1),-1))
		
		x=A@x
		x=x@self.w
		
		return x


class GNNlayer(nn.Module):
	def __init__(self,input_dimension,output_dimension,num_heads=4):
		super(GNNlayer,self).__init__()
		self.attention_module = nn.ModuleList([CPAlayer(input_dimension,output_dimension) for _ in range(num_heads)])
		
		self.linear= nn.Linear(output_dimension*num_heads,output_dimension)
		
	def forward(self,x,A_shape):
		x = torch.cat([attn(x,A_shape) for attn in self.attention_module], dim=-1)
		x = self.linear(x)
		
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
