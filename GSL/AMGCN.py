from torch.nn import Parameter
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GNNlayer(nn.Module):
	def __init__(self,input_dimension,output_dimension):
		super(GNNlayer,self).__init__()
		self.linear=nn.Linear(input_dimension,output_dimension)
		self.leakyrelu = nn.LeakyReLU(0.1)

	def forward(self,x,A):
		x=A@x
		x=self.linear(x)
		x=self.leakyrelu(x)
		
		return x


class Model(nn.Module):
	def __init__(self,linear_list1,linear_list2,gnn_list,dropout=None):
		super(Model,self).__init__()
		self.len_gnn_list=len(gnn_list)
		self.dropout=dropout
		
		self.linear_list1=nn.ModuleList([nn.Linear(linear_list1[i],linear_list1[i+1]) for i in range(len(linear_list1)-1)])
		self.linear_list2=nn.ModuleList([nn.Linear(linear_list2[i],linear_list2[i+1]) for i in range(len(linear_list2)-1)])
		self.gnn_list_T=nn.ModuleList([GNNlayer(gnn_list[i],gnn_list[i+1]) for i in range(len(gnn_list)-1)])
		self.gnn_list_C=nn.ModuleList([GNNlayer(gnn_list[i],gnn_list[i+1]) for i in range(len(gnn_list)-1)])
		self.gnn_list_F=nn.ModuleList([GNNlayer(gnn_list[i],gnn_list[i+1]) for i in range(len(gnn_list)-1)])
		self.linear=nn.Linear(gnn_list[-1],gnn_list[-1])
		self.q=Parameter(torch.randn(gnn_list[-1]).to(device))
		
		self.softmax=nn.Softmax(dim=-1)
		self.leakyrelu = nn.LeakyReLU(0.1)
		if dropout is not None:
			self.Dropout = nn.Dropout(p=dropout)

	def Laplacian(self,A):
		nodesize=len(A[-1])
		I=torch.eye(nodesize).to(device)
		
		D=torch.sum(A,-1)
		D=torch.unsqueeze(D,-1)+1e-10
		D=D**(-0.5)
		D=D*I
		
		L=D@A@D
		
		return L

	def cos(self,x):
		s=x**2
		s=torch.sum(s,-1)
		s=s**0.5
		s=torch.unsqueeze(s,dim=-1)
		s=x/s+1e-10
		s=s@(s.transpose(-1,-2))
		
		return s
		
	def heat(self,x):
		s=x.transpose(-1,-2)
		s=torch.unsqueeze(s,dim=-1)
		s=s-s.transpose(-1,-2)
		s=s**2+1e-10
		s=torch.sum(s,dim=-3)
		s=-0.5*s
		s=torch.exp(s)
		
		return s
	
	def L2(self,x):
		L2=x**2
		L2=torch.mean(L2,-1)
		L2=L2**0.5+1e-10
		L2=torch.unsqueeze(L2,dim=-1)
		L2=x/L2
		
		return L2
	
	def HSIC(self,x1,x2):
		nodesize=x1.size()[-2]
		self.R=(torch.eye(nodesize)-torch.ones(nodesize,nodesize)/nodesize).to(device)
		I=torch.eye(nodesize).to(device)
		
		x1=x1@(x1.transpose(-1,-2))
		x2=x2@(x2.transpose(-1,-2))
		L2=self.R@x1@self.R@x2
		L2=I*L2
		L2=torch.mean(L2)*(nodesize**2)/((nodesize-1)**2)
		
		return L2
		
	def forward(self,x,A,k=4):
		for i,layer in enumerate(self.linear_list1):
			x=layer(x)
			if self.dropout is not None:
				x=self.Dropout(x)
			x=self.leakyrelu(x)

		At=self.Laplacian(A)
		
		Af=self.heat(x)
		Af=torch.argsort(Af,descending=True)
		Af=torch.argsort(Af)
		Af=torch.where(Af<k,1,0)
		Af=self.cos(x)*Af
		Af=self.Laplacian(Af)
		
		zf,zt,zcf,zct=x,x,x,x
		
		for i in range(self.len_gnn_list-1):
			zf=self.gnn_list_F[i](zf,Af)
			zt=self.gnn_list_T[i](zt,At)
			zcf=self.gnn_list_C[i](zcf,Af)
			zct=self.gnn_list_C[i](zct,At)
		
		Lcf=self.L2(zcf)
		Lct=self.L2(zct)
		Lc=Lcf@(Lcf.transpose(-1,-2))-Lct@(Lct.transpose(-1,-2))
		Lc=Lc**2
		Lc=torch.mean(Lc)
		
		Ld=self.HSIC(zt,zct)+self.HSIC(zf,zcf)
		
		penalty_term=[Lc,Ld]
		
		zc=(zcf+zct)/2
		
		zf=torch.unsqueeze(zf,dim=-2)
		zt=torch.unsqueeze(zt,dim=-2)
		zc=torch.unsqueeze(zc,dim=-2)
		
		z=torch.cat((zf,zt,zc),-2)
		
		a=self.linear(z)
		a=torch.tanh(a)
		a=a@self.q
		a=self.softmax(a)
		
		z=z.transpose(-1,-2)@(torch.unsqueeze(a,dim=-1))
		z=torch.squeeze(z,dim=-1)
		
		for i,layer in enumerate(self.linear_list2):
			z=layer(z)
		
		return z,penalty_term
