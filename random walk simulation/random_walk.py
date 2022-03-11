import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time
######################################################
def generate():

    # random.seed(time)
    x = np.linspace(0,10,200)

    ####
    a = random.random()*5
    b = random.random()
    c = random.random()
    d = random.random()
    e = random.random()
    f = random.random()*3

    ####

    y = 10 * e - a*x + b*x**2 - c*x**3 + d*x**4 - f*np.sin(x)

    return np.array(list(zip(x,y)))
######################################################
def test():

    test = generate()
    result = []

    for loc in test:
        # print(loc[1])
        if (loc[1] < 0 or loc[1] > 10):
            break
        else:
            result.append(loc)

    result = np.array(result)
    # print(result)

    return result
######################################################



###   main   ###

num = 0
result = []
while(num < 100):
    result = test()
    num = len(result)

print(result)

plt.scatter(result[:,0],result[:,1])
plt.xlim(0,10)
plt.ylim(0,10)
plt.savefig("random walk simulation/random_walk_path.png")

df = pd.DataFrame(result,columns =['x_axis', 'y_axis'],dtype = float)
df.to_csv("random walk simulation/postion.csv",index=False)
