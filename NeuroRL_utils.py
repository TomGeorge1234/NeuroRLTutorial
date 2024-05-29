"""Utils (mostly plotting) for NeuroRL tutorial"""

import numpy as np
import matplotlib.pyplot as plt
import ratinabox
ratinabox.stylize_plots() # sets some RC params to make plots look better
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.xmargin'] = 0.05
rcParams['axes.ymargin'] = 0.05
rcParams['legend.frameon'] = True
rcParams['legend.scatterpoints'] = 4
rcParams["legend.framealpha"] = 0.4
rcParams["legend.edgecolor"] = [1,1,1,0]

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
    def __init__(self,gamma=0.5, alpha=0.1, n_states=10,):
        self.V = np.zeros(n_states)
        self.S_prev = None
        self.gamma = gamma
        self.alpha = alpha
        self.n_states = n_states

        #History data (for plotting) 
        self.V_history = []
        self.S_history = []
        self.R_history = []
        self.TD_history = []

    def plot(self, episode=0, axs=None):

        T = self.S_history[episode].shape[0]
        n_states = self.S_history[episode].shape[1] 
        max_V, min_V, max_TD, min_TD = 0,1,0,1
        for ep in range(len(self.V_history)):
            max_V = max(max_V, np.max(self.V_history[ep]))
            min_V = min(min_V, np.min(self.V_history[ep]))
            max_TD = max(max_TD, np.max(self.TD_history[ep]))
            min_TD = min(min_TD, np.min(self.TD_history[ep]))
        
        if axs is None:
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1], width_ratios=[3, 1])
            axs = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[2, 0])
            ]
        [ax1, ax2, ax3, ax4] = axs
        fig = ax1.get_figure()

        # Central big plot shows the states 
        ax1.imshow(self.S_history[episode], cmap='Greys', aspect='auto')
        ax1.set_ylabel("State, S")
        ax1.spines['left'].set_bounds(0, n_states-1)
        ax1.spines['bottom'].set_bounds(0, T-1)
        ax1.spines['left'].set_position(('outward', 3))
        ax1.spines['bottom'].set_position(('outward', 3))
        ax1.set_xticks(np.arange(T))
        ax1.set_yticks(np.arange(n_states, step=n_states//5))
        ax1.set_xticks([0, T-1])
        ax1.set_xticklabels([])
    

        # Right plot sharing y-axis with the central plot
        ax2.plot(self.V_history[episode][::-1], np.arange(len(self.V_history[episode])), 
        label='Value', color='C0')
        # fill between line and zero 
        ax2.fill_betweenx(np.arange(n_states), self.V_history[episode][::-1], 0, where=self.V_history[episode] > 0, color='C0', alpha=0.5)
        ax2.spines['left'].set_bounds(0, n_states-1)
        ax2.spines['bottom'].set_position(('outward', 3))
        ax2.spines['left'].set_position(('outward', 3))
        ax2.set_xlabel('Value, V(S)')
        ax2.set_yticks([0, n_states-1, n_states-1])
        ax2.set_yticklabels([])
        ax2.set_xticks([0, max_V])
        ax2.set_xlim([min_V, max_V])

        # First bottom plot sharing x-axis with the central plot
        ax3.plot(self.R_history[episode], label='Reward', color='red')
        ax3.fill_between(np.arange(T), self.R_history[episode], 0, where=self.R_history[episode] > 0, color='red', alpha=0.5)
        ax3.spines['bottom'].set_bounds(0, T-1)
        ax3.spines['left'].set_position(('outward', 3))
        ax3.spines['bottom'].set_position(('outward', 3))
        ax3.set_ylabel('Reward, R')
        ax3.set_xticks([0, T-1])
        ax3.set_xticklabels([])

        # Second bottom plot sharing x-axis with the central plot
        ax4.plot(self.TD_history[episode], label='TD error', color='green')
        ax4.fill_between(np.arange(T), self.TD_history[episode], 0, where=self.TD_history[episode] > 0, color='green', alpha=0.5)
        ax4.spines['bottom'].set_bounds(0, T-1)
        ax4.spines['left'].set_position(('outward', 3))
        ax4.spines['bottom'].set_position(('outward', 3))
        ax4.set_ylabel('TD error')
        ax4.set_xlabel('Time step, t')
        ax4.set_xticks(np.arange(T, step=T//5))
        ax4.set_yticks([0, max_TD])
        ax4.set_ylim([min_TD, max_TD])

        fig.suptitle(f'Episode {episode}')
        return axs
    
    def animate_plot(self,):
        n_episodes = len(self.S_history)
        episodes = np.arange(n_episodes)
        def update(episode, axs):
            for ax in axs:
                ax.clear()
            axs = self.plot(episode=episode, axs=axs)
            return axs
        
        axs = self.plot(episode=0, axs=None)
        anim = FuncAnimation(plt.gcf(), update, frames=episodes, fargs=(axs,), interval=20)        
        plt.close()
        return anim
        