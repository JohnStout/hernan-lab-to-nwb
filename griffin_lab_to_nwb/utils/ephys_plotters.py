# basic plotter
#
# Written by John Stout

import matplotlib.pyplot as plt
import numpy as np

# lets inherit the __init__ from process_signal
def multi_plotter(data: dict, fs: int, time_range: list = [0, 1], color: str = 'k'):
    """
    Generate plotting function that plots as many rows as there are signals

    Args:
        data: dictionary of csc data with csc names
        fs: sampling rate of csc data
    
    Optional:
        time_range: list telling the figure what to plot. Default is the first second.
        color: default is a single value, 'k'. This can take as many colors as there are data points.

    """
    #if type(data) is list:
        #data = list(data.values())

    if len(color) == 1:
        group_color = [color[0] for i in range(len(data))]

    fig, axes = plt.subplots(nrows=len(data),ncols=1)
    key_names = list(data.keys())
    idx = [int(time_range[0]*fs),int(time_range[1]*fs)]
    for i in range(len(data)):
        if i == len(data)-1:
            x_data = np.linspace(time_range[0],time_range[1],int(fs*(time_range[1]-time_range[0])))
            axes[i].plot(x_data, data[key_names[i]][idx[0]:idx[1]],color[i],linewidth=0.75)
            axes[i].set_xlabel("Time (sec)")
        else:
            axes[i].plot(data[key_names[i]][idx[0]:idx[1]],color[i],linewidth=0.75)
            #axes[i].xlabel("Time (sec)")
    fig.show()