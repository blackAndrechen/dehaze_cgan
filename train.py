import torch


from data.dataloader import dehazing_loader
from models.conditional_gan import ConditionalGAN

class opt():
	def __init__(self):
		self.lr=0.0001
		self.beta1=0.5
		self.use_gpu=torch.cuda.is_available()
		self.lambda_A=100
		

def train(model,dataload,start_epoch,end_epoch):

	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	if start_epoch!=0:
		model.load("checkpoints/",start_epoch-1)

	for epoch in range(start_epoch,end_epoch):
		for ite,(ori_img,haze_img) in enumerate(dataload):

			ori_img=ori_img.to(device)
			haze_img=haze_img.to(device)
		
			model.train(ori_img,haze_img)

		model.save("checkpoints/",epoch)
		# break12a


if __name__=="__main__":
	ori_img_path="/mnt/ssd/czp_dataset/haze/image/"
	haze_img_path="/mnt/ssd/czp_dataset/haze/train/"

	opt=opt()
	cond_gan=ConditionalGAN(opt)
	dataload=dehazing_loader(ori_img_path,haze_img_path)

	train(cond_gan,dataload,0,10)






