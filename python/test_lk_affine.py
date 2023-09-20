import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from LucasKanadeAffine import LucasKanadeAffine
from file_utils import mkdir_if_missing

data_name = 'landing'      # could choose from (car1, car2, landing) 

# load data name
data = np.load('../data/%s.npy' % data_name)
# data = np.load("/Users/quocdetran/Documents/HUST/CMU-16385-CV-fall-2022/assgn6/data/landing.npy")

# obtain the initial rect with format (x1, y1, x2, y2)
if data_name == 'car1':
    initial = np.array([170, 130, 290, 250])   
elif data_name == 'car2':
    initial = np.array([59,116,145,151])    
elif data_name == 'landing':
    initial = np.array([440, 80, 560, 140])     
else:
    assert False, 'the data name must be one of (car1, car2, landing)'

numFrames = data.shape[2]
w = initial[2] - initial[0]
h = initial[3] - initial[1]

# load over frames
rects = []
rects.append(initial)
fig = plt.figure(1)
ax = fig.add_subplot(111)
for i in range(numFrames-1):
    print("frame****************", i)
    It = data[:,:,i]
    It1 = data[:,:,i+1]
    rect = rects[i]

    # run algorithm
    s = time.time()
    M = LucasKanadeAffine(It, It1, rect)
    print('M ', M)
    print("\033[92m" + f"Time : {time.time() - s} seconds" + "\033[0m")
    # print(f"Time : {time.time() - s} seconds")

    # transform the old rect to new one
    corners = np.array([[rect[0], rect[1], 1], 
                        [rect[2], rect[3], 1]]).transpose()
    newRect = np.matmul(M, corners).transpose().reshape((4, ))
    rects.append(newRect)

    # Show image
    print("Plotting: ", rect)
    ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0]+1, rect[3]-rect[1]+1, linewidth=2, edgecolor='red', fill=False))
    plt.imshow(It1, cmap='gray')
    plt.axis('off')
    save_path = "../results/lk_affine/%s/frame%06d.jpg" % (data_name, i+1)
    mkdir_if_missing(save_path)
    plt.savefig(save_path)
    plt.pause(0.04)
    ax.clear()