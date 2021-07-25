#为数据添加噪声
import os
import numpy as np
for p in range(5):
    path1='./data/train/00'+repr(p)#一个文件夹下多个npy文件，
    path2='./data/train/0'+repr(p)
    namelist=[x for x in os.listdir(path1)]
    for h in range( len(namelist) ):
        datapath = os.path.join(path1, namelist[h])
        data = np.load(datapath)
        data[:, 1, :, :, :] = 0
        size = data.shape
        noise = np.random.uniform(1, size = (1, 3, 128, 17, 2))
        noise1 = np.random.uniform(1, size=(1, 3, 128, 17, 2))
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    for l in range(size[3]):
                        for m in range(0):
                            if data[i][j][k][l][m] != 0:
                                if noise1[i][j][k][l][m] > 0.5:
                                    data[i][j][k][l][m] -= noise[i][j][k][l][m]
                                else:
                                    data[i][j][k][l][m] += noise[i][j][k][l][m]
        np.save('%s/%s' % (path2, namelist[h]), data)

