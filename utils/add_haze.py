import cv2
import numpy as np
from PIL import Image




def show_matrix(img):
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    tmp=Image.fromarray(img)
    tmp.show()

def addHaze(m,A,bita):
	t=np.exp(-bita)
	tmp=np.ones(m.shape)
	tmp=np.multiply(tmp,A*(1-t))
	return np.add(np.multiply(m,t),tmp)


if __name__ == '__main__':
    path="NYU2_1.jpg"

    As=[0.6,1.0]
    bitas=[0.4,0.6,0.8,1.0,1.2,1.4,1.6]
    idx=0
    for A in As:
    	for bita in bitas:

		    m = addHaze(cv2.imread(path)/255.0,A,bita)*255
		    save_path=path.split(".")[0]+"_{}.jpg".format(idx)
		    idx+=1
		    cv2.imwrite(save_path, m)
		    print(idx)
