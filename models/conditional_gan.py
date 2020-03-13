import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable

from .networks import define_G,define_D
from .losses import init_loss

class ConditionalGAN():
	def __init__(self,opt):
		# super(ConditionalGAN, self).__init__(opt)
		self.opt=opt
		self.Tensor=torch.cuda.FloatTensor if opt.use_gpu else torch.Tensor

		self.net_G=define_G(3,3)
		self.net_D=define_D(3)
		
		self.net_G=self.net_G.cuda() if opt.use_gpu else self.net_G
		self.net_D=self.net_D.cuda() if opt.use_gpu else self.net_D

		self.optimizer_G = torch.optim.Adam( self.net_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
		self.optimizer_D = torch.optim.Adam( self.net_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )


		self.discLoss, self.contentLoss = init_loss(opt, self.Tensor)

		print(self.net_G,self.net_D)

	#real_A is input nohaze
	#real_B is input haze
	#train B to A
	def train(self,real_A,real_B):
		fake_B=self.net_G.forward(real_B)

		#train D
		self.optimizer_D.zero_grad()
		self.backward_D(real_B,fake_B,real_A)
		self.optimizer_D.step()

		#train G
		self.optimizer_G.zero_grad()
		self.backward_G(real_B,fake_B,real_A)
		self.optimizer_G.step()

	def backward_D(self,real_B,fake_B,real_A):
		self.loss_D = self.discLoss.get_loss(self.net_D, real_B, fake_B, real_A)
		print("loss_D:",self.loss_D)
		self.loss_D.backward(retain_graph=True)

	def backward_G(self,real_B,fake_B,real_A):
		self.loss_G_GAN = self.discLoss.get_g_loss(self.net_D, real_B, fake_B)
		# Second, G(A) = B
		self.loss_G_Content = self.contentLoss.get_loss(fake_B, real_A) * self.opt.lambda_A

		self.loss_G = self.loss_G_GAN + self.loss_G_Content
		print("loss_G:",self.loss_G)
		self.loss_G.backward()

	def save(self,path,epoch):
		G_path=path+"G_net_{}.pth".format(epoch)
		D_path=path+"D_net_{}.pth".format(epoch)

		torch.save(self.net_G.cpu().state_dict(),G_path)
		torch.save(self.net_D.cpu().state_dict(),D_path)

		self.net_G=self.net_G.cuda() if self.opt.use_gpu else self.net_G
		self.net_D=self.net_D.cuda() if self.opt.use_gpu else self.net_D

		print("sucess saved net")

	def load(self,path,epoch):
		G_path=path+"G_net_{}.pth".format(epoch)
		D_path=path+"D_net_{}.pth".format(epoch)

		self.net_G.load_state_dict(torch.load(G_path))
		self.net_D.load_state_dict(torch.load(D_path))

		self.net_G=self.net_G.cuda() if self.opt.use_gpu else self.net_G
		self.net_D=self.net_D.cuda() if self.opt.use_gpu else self.net_D

		print("sucess load net")

class opt():
	def __init__(self):
		self.lr=0.0001
		self.beta1=0.5
		self.use_gpu=torch.cuda.is_available()
		self.lambda_A=100