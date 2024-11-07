from torch.nn import Parameter
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HyperGATlayer(nn.Module):
	def __init__(self,input_dimension,output_dimension):
		super(HyperGATlayer,self).__init__()
		self.w1=Parameter(torch.randn(input_dimension,output_dimension).to(device))
		self.w2=Parameter(torch.randn(output_dimension,output_dimension).to(device))
		self.a1=Parameter(torch.randn(output_dimension,1).to(device))
		self.a21=Parameter(torch.randn(output_dimension,1).to(device))
		self.a22=Parameter(torch.randn(output_dimension,1).to(device))
		
		self.leakyrelu = nn.LeakyReLU(0.1)
	
	def forward(self,x,H_shape):
		f=x@self.w1
		f=self.leakyrelu(f)
		f=f@self.a1
		f=torch.tanh(f/8)*8
		f=torch.exp(f)*H_shape
		f=f/torch.unsqueeze(torch.sum(f,-2),-2)
		f=(f.transpose(-2,-1))@x@self.w1
		f=self.leakyrelu(f)
		
		b=(self.leakyrelu(f@self.w2)@self.a21).transpose(-2,-1)+self.leakyrelu(x@self.w1)@self.a22
		b=torch.tanh(b/8)*8
		b=torch.exp(b)*H_shape
		b=b/torch.unsqueeze(torch.sum(b,-1),-1)
		
		x=b@f@self.w2
		x=self.leakyrelu(x)
		
		return x


class GNNlayer(nn.Module):
	def __init__(self,input_dimension,output_dimension,num_heads=4):
		super(GNNlayer,self).__init__()
		self.attention_module = nn.ModuleList([HyperGATlayer(input_dimension,output_dimension) for _ in range(num_heads)])
		self.linear= nn.Linear(output_dimension*num_heads,output_dimension,bias=False)
	
	def forward(self,x,H_shape):
		x = torch.cat([attn(x,H_shape) for attn in self.attention_module], dim=-1)
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
