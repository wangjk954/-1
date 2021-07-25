#进行序列图像的左右翻转
import os
import numpy as np
for i in range(5):
	path1='./data/train/000'+repr(i)#一个文件夹下多个npy文件，
	path2='./data/train/200'+repr(i)+'A'
	namelist=[x for x in os.listdir(path1)]
	for i in range( len(namelist) ):
		datapath=os.path.join(path1,namelist[i])
		data = np.load(datapath)
		a = data[0][0]
		b = a*(-1)
		data[0][0] = b
		np.save('%s/%s'%(path2,namelist[i]),data)
print ('over')

