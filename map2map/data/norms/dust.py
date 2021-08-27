import numpy as np
import torch

def identity(x,**kwargs):
    pass

""" Equate everything for testing purpose 

Parameters:
------------

	x	: np.array
		  Input field

Return:
------------
	None 
	Changes are done inplace

"""
def equateAll(x,**kwargs):
	x[1] = x[2] = x[0]


""" Merge all fields into one

Parameters:
------------

	x	: np.array
		  Input field

Return:
------------
	None 
	Changes are done inplace

"""
def mergeAll(x,**kwargs):
	print('Before:',x.shape)
	x = torch.sum(x,axis=-4)
	print('After:',x.shape)


""" Normalization of form exp(- alpha* |xi|)

Parameters:
------------

	x	: np.array
		  Input field

	alpha	: float
		  Scale for exponent

Return:
------------
	None 
	Changes are done inplace

"""
def negExp(x, alpha=10,**kwargs):
	if 'negExp' in kwargs:
		alpha = kwargs['negExp']['alpha']
	torch.abs(x,out=x)
	torch.exp(-alpha*x,out=x)


""" Log normalization for dust particles 

Parameters:
------------

	x	: np.array
		  Input field
------------
	None 
	Changes are done inplace

"""
def simpleLog(x,**kwargs):
	if 'log-eps' in kwargs:
		eps = float(kwargs['log-eps'])
	torch.log(x+eps,out=x)

""" Filtering with threshold 

Parameters:
------------

	x 	: np.array
		  Input field
------------
	None 
	Changes are done inplace

"""
def filter(x,**kwargs):
	if 'filter-threshold' in kwargs:
		threshold = float(kwargs['filter-threshold'])
	mask = torch.where(x > threshold, 1, 0)
	x *= mask

""" Normalization by Yin 

Parameters:
------------
	x 	: np.array
		  Input fields
------------
	None; Changes are done inplace
"""
def lnExp(x,**kwargs):
	eps = 1e-8
	if 'lnexp-rho0' in kwargs:
		rho0 = float(kwargs['lnexp-rho0'])
	
	zero = torch.zeros(size=x.shape)
	#print('Before: ',x.get_device(), zero.get_device())
	if x.get_device() != zero.get_device():
		print('Inconsistent devices. Trying to fix it')
		zero = zero.to(device='cuda')
	#print('After: ',x.get_device(), zero.get_device())
	x /= rho0
	torch.exp(x,out=x)
	x -= 1
	torch.maximum(x,zero,out=x)
	torch.log(x+eps,out=x)
	#print(x)
	#torch.log(torch.exp(x/rho0) + eps -1)

def lnExp2(x,undo=False,**kwargs):
	eps = 1e-8 
	rho0 = float(kwargs['lnexp-rho0'])
	if undo == False:
		x/=rho0
		torch.expm1(x,out=x)
		torch.log(x+eps,out=x)
	else:
		torch.exp(x,out=x)
		x+=1
		torch.log(x+eps,out=x)
		x*= rho0

def lnExp3(x,undo=False,**kwargs):
	eps = 1e-8
	rho0 = float(kwargs['lnexp-rho0'])
	zero = torch.zeros_like(x)
	if undo == False:
		x/=rho0
		y = torch.where(x > 1,x+torch.log(1-torch.exp(-x)),torch.log(torch.expm1(x)+eps))
		x *= 0
		x += y
	else:
		torch.exp(x,out=x)
		x+=1
		torch.log(x+eps,out=x)
		x*=rho0


def lnExpShift(x,undo=False,**kwargs):
	eps = 1e-8
	rho0 = float(kwargs['lnexp-rho0'])
	shift = float(kwargs['lnexp-shift'])
	zero = torch.zeros_like(x)
	if undo == False:
		x/=rho0
		y = torch.where(x > 1,x+torch.log(1-torch.exp(-x)),torch.log(torch.expm1(x)+eps))
		x *= 0
		x += y
		x += shift
	else:
		x -= shift
		torch.exp(x,out=x)
		x+=1
		torch.log(x+eps,out=x)
		x*=rho0
