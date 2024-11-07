from torch.nn import Parameter
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from SGSL import *

#from GNN.AGNN import *
#from GNN.CPA_Additive import *
#from GNN.CPA_Scaled import *
#from GNN.DAGNN import * #only one GNN-layer
#from GNN.DPGAT import *
#from GNN.GATv2 import *
#from GNN.GAT import *
#from GNN.GCN import *

#from HNN.HCHA import *
#from HNN.HGC import *
#from HNN.HyperGAT import *
#from HNN.SHARE import *
#from HNN.STHAN_SR import *

#from GSL.AGCN import *
#from GSL.AMGCN import *
#from GSL.GDC import *
#from GSL.GLCN import *
#from GSL.GLNN import *
#from GSL.GRCN import *
#from GSL.IDGL import *

from EarlyStopping import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Loaddata(Dataset):
	
	def __init__(self,data1):
		super().__init__()
		self.data1=data1
		self.len=len(data1)
	
	def __getitem__(self,index):
		return self.data1[index]
	
	def __len__(self):
		return self.len


def divide_A(A,range_list):
	divided_A=A[:,range_list[0]:range_list[1]]
	divided_A=torch.sum(divided_A,-1)
	
	divided_A_index=[]
	
	for i in range(len(divided_A)):
		if divided_A[i]>0:
			divided_A_index.append(i)
	
	return divided_A_index

def accuracy(y,label):
	y=torch.argsort(y,descending=True)
	y=torch.argsort(y)
	y=torch.where(y>0,0,1)
	
	num=torch.abs(y-label)
	num=torch.sum(num)
	num=num/2
	num=num/(label.size()[-2])
	
	return 1-num

def mask_A(A,k):
	output=A
	for i in range(k-1):
		output=output@A
	output=torch.where(output>0,1.0,0.0)
	return output
	
	
def train_node_invariant_task(data,A,hiden_dimension,path,device=device,train_SGSL=True):
	train_num=int(len(data)*0.5)
	val_num=int(len(data)*0.2)
	
	test_data=data[(train_num+val_num):]
	val_data=data[train_num:(train_num+val_num)]
	train_data=data[:train_num]
	
	batch_size=128
	nodesize=len(A)
	train_data = Loaddata(train_data)
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	
	linear_list1,gnn_list,linear_list2=hiden_dimension[0],hiden_dimension[1],hiden_dimension[2]
	
	if train_SGSL:
		net=Model(linear_list1,linear_list2,gnn_list,nodesize,invariant=True).to(device)
	else:
		net=Model(linear_list1,linear_list2,gnn_list).to(device)
	optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
	MSELoss=nn.MSELoss().to(device)
	leakyrelu = nn.LeakyReLU(0.1)
	earlystopping = EarlyStopping(path)
	
	#state_dict = torch.load(path)
	#net.load_state_dict(state_dict)

	i=0
	min_test_loss=1e8
	t=6
	k=3

	while True:
		net.train()

		i+=1
		train_loss=0

		for data in train_loader:
			optimizer.zero_grad()
			
			train_data1,train_data2=data[:,:,:t],data[:,:,t+k]
			
			x,penalty_term=net(train_data1,A)
			x=torch.squeeze(x)
			x=leakyrelu(x)

			L1=MSELoss(x,train_data2)
			L2=sum(penalty_term)
			if len(penalty_term)==0:
				train_epoch_loss=L1
			else:
				train_epoch_loss=L1+1e-3*L2/len(penalty_term)

			train_loss+=L1*len(train_data1)
			
			train_epoch_loss.backward()
			optimizer.step()

		train_loss=train_loss/len(train_data)
		
		penalty_term=['%.8f'% i.item() for i in penalty_term]

		with torch.no_grad():
			net.eval()
			x,_=net(val_data[:,:,:t],A)
			x=torch.squeeze(x)
			x=leakyrelu(x)
			val_loss=MSELoss(x,val_data[:,:,t+k])

		earlystopping(val_loss,net)
		if earlystopping.early_stop:
			print("Early stopping")
			break

		if earlystopping.is_improve:
			with torch.no_grad():
				net.eval()
				x,_=net(test_data[:,:,:t],A)
				x=torch.squeeze(x)
				x=leakyrelu(x)
				test_loss=MSELoss(x,test_data[:,:,t+k])
				
				print("val_loss decrease")
				print("val_loss",'%.8f' % val_loss.item())
				print("train_loss",'%.8f' % train_loss.item())
				print("test_loss",'%.8f' % test_loss.item())
				print("penalty_term",penalty_term)
				print("")

		if i%200==0:
			with torch.no_grad():             
				net.eval()
				x,_=net(test_data[:,:,:t],A)
				x=torch.squeeze(x)
				x=leakyrelu(x)
				test_loss=MSELoss(x,test_data[:,:,t+k])
				
				print("period calculation")
				print("train_loss",'%.8f' % train_loss.item())
				print("test_loss",'%.8f' % test_loss.item())
				print("penalty_term",penalty_term)  
				print("")

		if i==16000:
			break

	with torch.no_grad():             
		net.eval()
		state_dict = torch.load(path)
		net.load_state_dict(state_dict)

		x,_=net(test_data[:,:,:t],A)
		x=torch.squeeze(x)
		x=leakyrelu(x)
		test_loss=MSELoss(x,test_data[:,:,t+k])
		print("----------------------------------------------------------------")
		print("test_loss",'%.8f' % test_loss.item())

def train_node_variant_task(feature,label,A,hiden_dimension,path,device=device,train_SGSL=True):
	train_num=200
	val_num=500
	
	mask=mask_A(A,2)
	train_A_index=divide_A(A,[0,train_num])
	train_A=A[train_A_index,:][:,train_A_index]
	train_mask=mask[train_A_index,:][:,train_A_index]
	train_data=feature[train_A_index,:]

	linear_list1,gnn_list,linear_list2=hiden_dimension[0],hiden_dimension[1],hiden_dimension[2]
	
	if train_SGSL:
		net=Model(linear_list1,linear_list2,gnn_list,len(train_A),dropout=0.5,invariant=False).to(device)
	else:
		net=Model(linear_list1,linear_list2,gnn_list,dropout=0.5).to(device)
	optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
	CrossEntropyLoss=nn.CrossEntropyLoss().to(device)
	softmax=nn.Softmax(dim=-1)
	earlystopping = EarlyStopping(path)

	#state_dict = torch.load(path)
	#net.load_state_dict(state_dict)

	i=0
	min_test_loss=1e8

	while True:
		net.train()
		optimizer.zero_grad()
		
		i+=1
		train_loss=0
		
		if train_SGSL:
			x,penalty_term=net(train_data,train_A,train_mask)
		else:
			x,penalty_term=net(train_data,train_A)
		x=softmax(x)
		train_loss=CrossEntropyLoss(x[:train_num],label[:train_num])
		if len(penalty_term)==0:
			train_loss=train_loss
		else:
			train_loss=train_loss+1e-3*sum(penalty_term)/len(penalty_term)
		
		train_loss.backward()
		optimizer.step()

		penalty_term=['%.8f'% i.item() for i in penalty_term]

		with torch.no_grad():
			net.eval()
			if train_SGSL:
				x,_=net(feature,A,mask)
			else:
				x,_=net(feature,A)
			x=softmax(x)
			val_loss=CrossEntropyLoss(x[train_num:train_num+val_num],label[train_num:train_num+val_num])

		earlystopping(val_loss,net)
		if earlystopping.early_stop:
			print("Early stopping")
			break

		if earlystopping.is_improve:
			with torch.no_grad():
				net.eval()
				test_loss=CrossEntropyLoss(x[train_num+val_num:],label[train_num+val_num:])
				acc=accuracy(x[train_num+val_num:],label[train_num+val_num:])
				
				print("val_loss decrease")
				print("val_loss",'%.8f' % val_loss.item())
				print("train_loss",'%.8f' % train_loss.item())
				print("test_loss",'%.8f' % test_loss.item())
				print("accuracy",'%.8f' % acc)
				print("penalty_term", penalty_term)
				print("")

		if i%400==0:
			with torch.no_grad():             
				net.eval()
				test_loss=CrossEntropyLoss(x[train_num+val_num:],label[train_num+val_num:])
				acc=accuracy(x[train_num+val_num:],label[train_num+val_num:])
				
				print("period calculation")
				print("train_loss",'%.8f' % train_loss.item())
				print("test_loss",'%.8f' % test_loss.item())
				print("accuracy",'%.8f' % acc)
				print("penalty_term", penalty_term)
				print("")

		if i==4000:
			break

	with torch.no_grad():             
		net.eval()
		state_dict = torch.load(path)
		net.load_state_dict(state_dict)
		
		if train_SGSL:
			x,_=net(feature,A,mask)
		else:
			x,_=net(feature,A)
		x=softmax(x)
		test_loss=CrossEntropyLoss(x[train_num+val_num:],label[train_num+val_num:])
		print("----------------------------------------------------------------")
		print("test_loss",'%.8f' % test_loss.item())
		print("accuracy",'%.8f' % accuracy(x[train_num+val_num:],label[train_num+val_num:]))

path='output/save_sample.pt'

#Take traffic flow prediction as example, in node-invariant tasks, please uncomment the following lines of code.
"""data=torch.load('input/switzerland_meteorological_data.pt').to(device)
A=torch.load('input/switzerland_A.pt').to(device)
A=torch.tensor(A, dtype=torch.float32)

linear_list1=[6,4,4]
gnn_list=[4]*5
linear_list2=[4,1]
hiden_dimension=[linear_list1,gnn_list,linear_list2]

train_node_invariant_task(data,A,hiden_dimension,path)"""

#Take citeseer as example, in node-variant tasks, please uncomment the following lines of code.
"""feature=torch.load('input/citeseer_features.pt').to(device)
label=torch.load('input/citeseer_labels.pt').to(device)
A=torch.load('input/citeseer_A.pt').to(device)

linear_list1=[feature.size()[-1],32]
gnn_list=[32]+[16]*2
linear_list2=[16,label.size()[-1]]
hiden_dimension=[linear_list1,gnn_list,linear_list2]

train_node_variant_task(feature,label,A,hiden_dimension,path)"""

#To use the control group GNN model, please uncomment the corresponding GNN model lines (8-29). 
#For example, to train the control group model on citeseer, use the following code.
"""train_node_variant_task(feature,label,A,hiden_dimension,path,train_SGSL=False)"""
