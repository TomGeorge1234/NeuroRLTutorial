"""Utils (mostly plotting) for NeuroRL tutorial"""

import numpy as np
import matplotlib.pyplot as plt
import ratinabox
ratinabox.stylize_plots() # sets some RC params to make plots look better
from matplotlib import rcParams
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.xmargin'] = 0.05
rcParams['axes.ymargin'] = 0.05
rcParams['legend.frameon'] = False


def format_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    current_xlims = ax.dataLim.bounds[0], ax.dataLim.bounds[2]
    current_ylims = ax.dataLim.bounds[1], ax.dataLim.bounds[3]
    ax.spines['left'].set_bounds(current_ylims[0], current_ylims[1])
    ax.spines['bottom'].set_bounds(current_xlims[0], current_xlims[1])
    return ax

class BaseRescorlaWagner:
    def __init__(self, n_stimuli=1):
        self.n_stimuli = n_stimuli

    def plot(self, ax=None):
        if self.n_stimuli == 1:
            ax = self.plot_1d(ax)
        else:
            ax = self.plot_2d(ax)
        return ax 
        
    def plot_1d(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2)) 
        ax.scatter(np.arange(len(self.V_history)), self.V_history, label='Predicted value', linewidth=0, s=5, alpha=0.7) 
        ax.scatter(np.arange(len(self.R_history)), self.R_history, label='Reward', color='red', linewidth=0, alpha=0.7)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Value')
        ax.legend()
        ax = format_axes(ax)
        return ax
    
    def plot_2d(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2)) 
        for i in range(self.n_stimuli):
            ax.scatter(np.arange(len(self.W_history)), self.W_history[:, i], label=f'Stimulus {i} weight', linewidth=0, s=5, alpha=0.7)
        ax.scatter(np.arange(len(self.V_history)), self.V_history, label='Total predicted value', color='green', linewidth=0, alpha=0.7)
        ax.scatter(np.arange(len(self.R_history)), self.R_history, label='Reward', color='red', linewidth=0, alpha=0.7)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Association weight')
        ax.legend()
        ax = format_axes(ax)
        return ax  


class BaseTDLearner:
    def plot(self, ax=None):
        S = np.zeros_like(self.V_history)
        S[np.arange(len(self.S_history)), self.S_history] = 1
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(3, 2), sharex=True) 

        ax[0].imshow(np.array(self.V_history).T, aspect='auto')
        ax[0].set_ylabel('State')
        ax[0].set_title('Value function')
        ax[1].imshow(S.T, aspect='auto')
        ax[1].set_xlabel('Trial')
        ax[1].set_ylabel('State')
        return ax