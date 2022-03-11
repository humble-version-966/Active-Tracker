import matplotlib.pyplot as plt
import random

def randomwalk2D(start,end,length,width):
    x = start[0]
    y = start[1]
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    locx = []
    locy = []
    locx.append(x)
    locy.append(y)

    plt.title("2D Random Walk in Python")
    plt.scatter(x,y)

    while(x != end[0] or y != end[1]):
        # Pick a direction at random
        step = random.choice(directions)
        
        # Move the object according to the direction
        if step == "RIGHT":
            if x >= length/2:
                continue
            x += 1
        elif step == "LEFT":
            if x <= -length/2:
                continue
            x -= 1
        elif step == "UP":
            if y >= width/2:
                continue
            y += 1
        elif step == "DOWN":
            if y <= -width/2:
                continue
            y -= 1
        locx.append(x)
        locy.append(y)

    
    # Return all the x and y positions of the object
    plt.scatter(x,y)
    plt.plot(locx,locy)
    plt.savefig('path.png')
    return

