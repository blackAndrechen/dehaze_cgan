import torch
import os
from PIL import Image
import cv2
import numpy as np
import torchvision
from tqdm import tqdm

from data.dataloader import dehazing_loader
from models.conditional_gan import ConditionalGAN

class opt():
	def __init__(self):
		self.lr=0.0001
		self.beta1=0.5
		self.use_gpu=torch.cuda.is_available()
		self.lambda_A=100
		

if __name__=="__main__":
	test_path="test/"
	result_path="result/"
	load_model_epoch=9
	test_num=10

	opt=opt()
	cond_gan=ConditionalGAN(opt)
	cond_gan.load("checkpoints/",load_model_epoch)

	test_img=os.listdir(test_path)
	for i in tqdm(range(len(test_img))):
		ori_img_path=test_path+test_img[i]
		ori_img = Image.open(ori_img_path)
		
		ori_img=ori_img.resize((480,640), Image.ANTIALIAS)
		ori_img=(np.asarray(ori_img)/255.0)
		ori_img=torch.from_numpy(ori_img).float()
		ori_img=ori_img.permute(2,0,1).unsqueeze(0)

		output=cond_gan.net_G.forward(ori_img)

		save_path=result_path+test_img[i]

		#todo
		# clean_image=output[0]*255.0
		# dehaze_img=output[0].permute(1,2,0).detach().numpy()
		# tmp=dehaze_img[:,:,2]
		# dehaze_img[:,:,2]=dehaze_img[:,:,0]
		# # dehaze_img[:,:,1]=dehaze_img[:,:,0]
		# dehaze_img[:,:,0]=tmp
		# cv2.imwrite(save_path, dehaze_img*255.0)

		torchvision.utils.save_image(torch.cat((ori_img, output),0), save_path)





