#对序列图像进行逆序操作
import os
import numpy as np
for i in range(5):
	path1='./data/train/000'+repr(i)#一个文件夹下多个npy文件，
	path2='./data/train/200'+repr(i)+'B'
	namelist=[x for x in os.listdir(path1)]
	for i in range( len(namelist) ):
		datapath=os.path.join(path1,namelist[i])
		data = np.load(datapath)
		data[0][0] =np.flip(data[0][0], axis=0)
		data[0][2] =np.flip(data[0][2], axis=0)
		np.save('%s/%s'%(path2,namelist[i]),data)
print ('over')

