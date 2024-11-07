import numpy as np
import torch

class EarlyStopping:
	def __init__(self, save_path, patience=800):
		
		self.save_path = save_path
		self.patience = patience
		self.counter = 0
		self.best_score = 1e10
		self.early_stop = False
		self.is_improve=False

	def __call__(self, val_loss, model):
		
		score = val_loss
		
		if score>self.best_score*1.001:
			
			self.is_improve=False
			
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			
			if score<self.best_score:
				
				self.is_improve=True
				self.best_score = score
				torch.save(model.state_dict(),self.save_path)
				self.counter = 0

			else:
				self.is_improve=False
    
