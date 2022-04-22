# Reward function

## Inputs

#### 1. 上一步落点 (X~route~ , Y~route~) 以及在此之前的路径点集

#### 2. LSTM预测点 (X~prediction~ , Y~prediction~)

#### 3. RL对抗点 (X~adversarial~ , Y~adversarial~)



## 全局变量

#### 1. 步长用时 (每帧间隔时长) t

#### 2. 出口位置 (X~out~ , Y~out~)

#### 3. 步长取值范围 (对应速度快慢) [step_length~min~ , step_length~max~]



## 对于对抗点的reward要素

#### 1. 离出口距离 (change to the vector between the RLpoint and the exit)

**1.1.** 上一步落点到对抗点的*vector~1~*在上一步落点到出口的*vector~2~*上的投影长度*d*
$$
d=\frac{\vec{V_1}·\vec{V_2}}{|\vec{V_2}|}=\frac{(X_a-X_r)·(X_o-X_r)+(Y_a-Y_r)·(Y_o-Y_r)}{\sqrt{(X_o-X_r)^2+(Y_o-Y_r)^2}}
$$

#### 2. 对抗LSTM的效果

**2.1.** 上一步落点到对抗点的*vector~1~*与预测点到对抗点的*vector~3~* 之间的角度$\theta_0$
$$
\cos{\theta_0}=\frac{\vec{V_1}·\vec{V_3}}{|\vec{V_1}|·|\vec{V_3}|}=\frac{(X_a-X_r)·(X_p-X_r)+(Y_a-Y_r)·(Y_p-Y_r)}{\sqrt{(X_a-X_r)^2+(Y_a-Y_r)^2}·\sqrt{(X_p-X_r)^2+(Y_p-Y_r)^2}}
$$
**2.2.**  *vector~1~*和*vector~3~*的长度$|\vec{V_1}|$ , $|\vec{V_3}|$
$$
|\vec{V_1}|=\sqrt{(X_a-X_r)^2+(Y_a-Y_r)^2}
$$

$$
|\vec{V_3}|=\sqrt{(X_p-X_r)^2+(Y_p-Y_r)^2}
$$

#### 3. 与实际route的拟合情况

**3.1.** 对抗路线步长$|\vec{V_1}|$与上一步长$|\vec{V_0}|$之差 (对应速度差距、变化量)

**3.2.** 对抗路线与前一步、两步路线角度变化





