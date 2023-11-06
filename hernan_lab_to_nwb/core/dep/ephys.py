# class ephys inherits base formatting __init__
from decode_lab_code.core.base import base
import numpy as np
import matplotlib.pyplot as plt

class ephys(base):

    # lets inherit the __init__ from process_signal
    def lfp_plotter(self, csc_attribute: str = 'csc_data', csc_names: list = [], time_range: list = [0, 1], color = None):

        """
        Generate plotting function that plots as many rows as there are signals

        Optional:
            csc_names: names of csc channels to plot as denoted by self.data.csc_data_names
            time_range: list telling the figure what to plot. Default is the first second.
            color: default is a single value, 'k'. This can take as many colors as there are data points.

        """

        # get csc_data
        csc_data = getattr(self,csc_attribute)

        if color is None:
            temp_color = 'k'
            color = [temp_color[0] for i in range(len(csc_data))]
        
        # get sampling rate - this method and class will be inherited by ioreaders
        fs = self.csc_data_fs

        fig, axes = plt.subplots(nrows=len(csc_data),ncols=1,)
        key_names = list(csc_data.keys())
        idx = [int(time_range[0]*fs),int(time_range[1]*fs)]
        for i in range(len(csc_data)):
            if i == len(csc_data)-1:
                x_data = np.linspace(time_range[0],time_range[1],int(fs*(time_range[1]-time_range[0])))
                axes[i].plot(x_data, csc_data[key_names[i]][idx[0]:idx[1]],color[i],linewidth=0.75)
                axes[i].set_xlabel("Time (sec)")
            else:
                axes[i].plot(csc_data[key_names[i]][idx[0]:idx[1]],color[i],linewidth=0.75)
                #axes[i].xlabel("Time (sec)")
            axes[i].yaxis.set_tick_params(labelsize=8)
            axes[i].xaxis.set_tick_params(labelsize=8)
            #var(axes[i])
        fig.show()
    