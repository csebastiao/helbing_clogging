# -*- coding: utf-8 -*-
"""
Animation from the results of helbing_model.py with the package matplotlib
"""

from matplotlib import pyplot as plt
from matplotlib import animation
import pandas as pd

def set_walls(ax, L, hole_size):
    "Set the wall in the figure."
    #exit wall
    ax.add_patch(plt.Rectangle([L/2, hole_size/2], hole_size,
                               2 * L - hole_size/2, color = 'gray'))
    ax.add_patch(plt.Rectangle([L/2, -2 * L], hole_size,
                               2 * L - hole_size/2, color = 'gray'))
    ax.add_patch(plt.Rectangle([-4 * L, -2 * L], hole_size,
                               4 * L, color = 'gray')) #back wall
    ax.add_patch(plt.Rectangle([4 * L, -2 * L], hole_size,
                               8 * L, angle = 90., color = 'gray')) #down wall  
    ax.add_patch(plt.Rectangle([4 * L, 2 * L - hole_size], hole_size,
                               8 * L, angle = 90., color = 'gray')) #top wall

def show_single_frame(history, L, hole_size, frame):
    """
    Show a single frame of the history

    Parameters
    ----------
    history : pandas.DataFrame
        Dataframe with the whole history of the simulation from helbing_model,
        with 'Timestep', 'FishID', 'X', 'Y' and 'R'.
    walls : numpy.array
        Coordinates of every dot from the walls.
    frame : int
        Frame we want to see.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize = (16, 8))
    ax.axes.get_xaxis().set_visible(False) 
    ax.axes.get_yaxis().set_visible(False) 
    ax.axes.set_xlim([-4 * L, 4 * L])
    ax.axes.set_ylim([-2 * L, 2 * L])
    set_walls(ax, L, hole_size) #plot walls
    xx = list(h[h['Timestep'] == frame]['X'])[1:] #get results, we need to
    yy = list(h[h['Timestep'] == frame]['Y'])[1:] #ignore first line which is
    R = list(h[h['Timestep'] == frame]['R'])[1:] #the name of the index
    for i in range(len(R)): #plot every fish depending on their size
        ax.add_patch((plt.Circle(
            (xx[i], yy[i]), R[i], facecolor = 'b', edgecolor = 'black')))


class Iterator():
    """
    Iterate the content of the xx and yy for the animation.
    
    xpool : list
        List of x coordinates
    ypool : list
        List of y coordinates
    x : float
        Actual x of the iterator, from the xpool
    y : float
        Actual y of the iterator, from the ypool
    count : int
        Number of iteration
    
    """
    def __init__(self, xx, yy, cc):
        self.xpool = xx
        self.ypool = yy
        self.cpool = cc
        self.x = self.xpool[0]
        self.y = self.ypool[0]
        self.c = self.cpool[0]
        self.count = 0
        
    def update(self):
        "Update the count and the x and y of the iterator"
        self.x = self.xpool[self.count]
        self.y = self.ypool[self.count]
        self.c = self.cpool[self.count]
        self.count += 5

def animate(i, iterator, circles):
    "Function need for matplotlib.animation.FuncAnimation(), iterate position"
    for circ in circles:
        iterator.update()
        circ.set(center=(iterator.x, iterator.y), facecolor = iterator.c)
    return circles


def get_animation(history, L, hole_size):
    """
    Make an animation from the history of the position of the fish

    Parameters
    ----------
    history : pandas.DataFrame
        Dataframe with the whole history of the simulation from helbing_model,
        with 'Time', 'FishID', 'X', 'Y' and 'R'.
    walls : numpy.array
        Coordinates of every dot from the walls.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        Useful object if we want to save the animation afterwards.

    """
    fig, ax = plt.subplots()
    ax.axes.get_xaxis().set_visible(False) 
    ax.axes.get_yaxis().set_visible(False) 
    ax.axes.set_xlim([-4 * L, 4 * L])
    ax.axes.set_ylim([-2 * L, 2 * L])
    set_walls(ax, L, hole_size)
    max_iteration = int(len(history['Time'])/(max(history['FishID'])+1))
    circles = []
    for i in range(max(history['FishID']) + 1): #plot fish
        circles.append(ax.add_patch(plt.Circle((
            list(history[history['FishID'] == i]['X'])[1],
            list(history[history['FishID'] == i]['Y'])[1]),
            list(history[history['FishID'] == i]['R'])[1],
            facecolor = 'b', edgecolor='black')))
    it = Iterator(list(history['X']), list(history['Y']), list(history['C']))
    ani = animation.FuncAnimation(fig, animate, fargs = (it, circles),
                              frames = max_iteration, interval = 40,
                              repeat = False)
    return ani

    
#TODO : Standardized namefile where we can extract the number of fish, the
#size of the walls and maybe the number of timestep and the dt ?

title = 'fixed_history_49_ds12_ka48_0.csv'

h = pd.read_csv(title)
L = 10
hole_size = 1


#show_single_frame(h, L, hole_size, 100)

ani = get_animation(h, L, hole_size)


mp4 = title.replace("history_", "")
mp4 = mp4.replace("csv", "mp4")
ani.save(mp4)