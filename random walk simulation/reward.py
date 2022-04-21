from re import X
#from cv2 import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

def reward(path,LSTMpoint,RLpoint,out_position):
    Xr = path[-1][0]  # the last point of the real path
    Yr = path[-1][1]
    Xp = LSTMpoint[0]  # the prediction point of the LSTM
    Yp = LSTMpoint[1]
    Xa = RLpoint[0]  # the adversarial point of RL
    Ya = RLpoint[1]
    Xl1 = path[-2][0] # the last i point of the real path, here it is 2 for i.
    Yl1 = path[-2][1] 
    Xo = out_position[0]
    Yo = out_position[1]
    alpha1 = 0.5
    alpha2 = 0.2
    alpha3 = 0.3
    # 1. the distance from the exit     50%
    # 2. the effect of competing the LSTM point     20%
    # 3. the status of corresponding with the origin path      30%
    projection_in_exit = ((Xa - Xr)*(Xo - Xr) + (Ya - Yr)*(Yo - Yr)) / math.sqrt(((Xo - Xr)**2 + (Yo - Yr)**2))   # the distance to the exit
    cos_path_to_exit = projection_in_exit / math.sqrt((Xa - Xr)**2 + (Ya - Yr)**2)
    V1 = math.sqrt((Xa - Xr)**2 + (Ya - Yr)**2) # the length of vector1, from last point to adversarial point
    V3 = math.sqrt((Xp - Xr)**2 + (Yp - Yr)**2) # the length of vector3, from last point to prediction point
    cos_V1_to_V3 = np.abs((Xa - Xr)*(Xp - Xr) + (Ya - Yr)*(Yp - Yr)) / (V1 * V3)   # the angle between V1 and V3
    V01 = math.sqrt((Xl1 - Xr)**2 + (Yl1 - Yr)**2) # the length of vector0_1, from last one point to the last point
    cos_V01_to_V1 = (Xa - Xr)*(Xr - Xl1) + (Ya - Yr)*(Yr - Yl1) / (V01 * V1)    # the angle between V01 to V1
    

    reward = (math.pi - np.arccos(cos_path_to_exit))/(math.pi) * alpha1 + np.arccos(cos_V1_to_V3)/(math.pi) * alpha2 + (np.pi-np.arccos(cos_V01_to_V1))/(math.pi) * alpha3
    return reward
