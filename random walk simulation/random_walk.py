import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time
from reward import *
######################################################
def generate():
    # random.seed(20)
    x = np.linspace(0,10,200)
    a = random.random()*5
    b = random.random()
    c = random.random()
    d = random.random()
    e = random.random()
    f = random.random()*3
    y = 10 * e - a*x + b*x**2 - c*x**3 + d*x**4 - f*np.sin(x)
    return np.array(list(zip(x,y)))
######################################################
def test():
    test = generate()
    result = []
    for loc in test:
        if (loc[1] < 0 or loc[1] > 10):
            break
        else:
            result.append(loc)
    result = np.array(result)
    return result
######################################################


###   main   ###

num = 0
result = []
Xo = 10/2
Yo = 10
out_position = (Xo, Yo)
while(num < 100):
    result = test()
    num = len(result)

# df = pd.DataFrame(result,columns =['x_axis', 'y_axis'],dtype = float)
# df.to_csv("random walk simulation/postion.csv",index=False)

path = result[:100]
plt.scatter(path[:,0],path[:,1],1)
plt.xlim(0,10)
plt.ylim(0,10)
lastx = path[-1][0]
lasty = path[-1][1]
LSTMpoint = np.array([lastx+0.2,lasty+0.1])
plt.scatter(LSTMpoint[0],LSTMpoint[1],5,"red")

count = 20
while(count > 0):
    count -= 1
    sign = random.random()*1.0
    if (sign > 0.75):
        dx = random.random()*1.0
        dy = random.random()*1.0
    elif (sign > 0.5):
        dx = -random.random()*1.0
        dy = random.random()*1.0
    elif (sign > 0.25):
        dx = random.random()*1.0
        dy = -random.random()*1.0
    else:
        dx = -random.random()*1.0
        dy = -random.random()*1.0
    RLpoint = np.array([lastx+dx,lasty+dy])
    rew = reward(path,LSTMpoint,RLpoint,out_position)
    if (rew > 0.8):
        a = 1
    elif (rew > 0.6):
        a = 0.8
    elif (rew > 0.4):
        a = 0.6
    else:
        a = 0.4
    plt.scatter(RLpoint[0],RLpoint[1],5,"blue",alpha=a)
plt.savefig("random walk simulation/random_walk_path.png")