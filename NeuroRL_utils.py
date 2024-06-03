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
rcParams["legend.framealpha"] = 0.1
rcParams["legend.edgecolor"] = [1,1,1,0]

def format_axes(ax, xlims=None, ylims=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xlims = (xlims or (ax.dataLim.bounds[0], ax.dataLim.bounds[2]))
    ylims = (ylims or (ax.dataLim.bounds[1], ax.dataLim.bounds[3]))
    ax.spines['left'].set_bounds(ylims[0], ylims[1])
    ax.spines['bottom'].set_bounds(xlims[0], xlims[1])
    return ax

class BaseRescorlaWagner:
    def __init__(self, n_stimuli=1, alpha=0.1):
        self.n_stimuli = n_stimuli
        if n_stimuli == 1:
            self.V_history = [self.V]
        elif n_stimuli > 1: 
            self.V_history = [0]
            self.W_history = [self.W.copy()]
            self.S_history = [np.zeros(n_stimuli)]
        self.alpha = alpha
        self.R_history = [0]

    def plot(self, ax=None):
        if self.n_stimuli == 1:
            ax = self.plot_1d(ax)
        else:
            ax = self.plot_2d(ax)
        return ax 
        
    def plot_1d(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2)) 
        ax.scatter(np.arange(len(self.V_history)), self.V_history, label='Predicted value', color='C0', linewidth=0, s=5, alpha=0.7) 
        ax.scatter(np.arange(len(self.R_history)), self.R_history, label='Reward', color='orange', linewidth=0, alpha=0.7)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Value')
        ax.legend()
        ax = format_axes(ax)
        return ax
    
    def plot_2d(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 4, 2])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        N_trial = len(self.V_history)
        S_history = np.array(self.S_history); maxS, minS = np.max(S_history), np.min(S_history)
        R_history = np.array(self.R_history); maxR, minR = np.max(R_history), np.min(R_history)   
        W_history = np.array(self.W_history); maxW, minW = np.max(W_history), np.min(W_history)
        V_history = np.array(self.V_history); maxV, minV = np.max(V_history), np.min(V_history)

        for t in range(N_trial):
            for i in range(self.n_stimuli):
                ax1.add_artist(plt.Rectangle((t-0.5, i-0.5), 1, 1, color=f'C{i+1}', alpha = (S_history[t,i] - minS) / max(1, maxS), linewidth=0))
                ax2.scatter(t, W_history[t,i], label=f'Stimulus {i+1} weight' if t == 0 else None, color=f'C{i+1}', linewidth=0, s=8, alpha=0.7)
            ax3.scatter(t, V_history[t], label='Predicted (s.w)' if t==0 else None, s=8, color='C0', linewidth=0, alpha=0.7)
            ax3.scatter(t, R_history[t], label='Actual (reward recieved)' if t==0 else None, color='orange', linewidth=0, alpha=0.7)
        
        # format axes 
        ax1 = format_axes(ax1, ylims=(0, self.n_stimuli-1), xlims=(0, N_trial-1))
        ax1.set_yticks(np.arange(self.n_stimuli))
        ax1.set_yticklabels([f'{i+1}' for i in range(self.n_stimuli)])
        ax1.set_ylim(-0.5, self.n_stimuli-0.5)
        ax2 = format_axes(ax2)
        ax3 = format_axes(ax3)

        # y axis spines 
        # ax1.spines['left'].set_visible(False)
        ax1.set_yticks(np.arange(self.n_stimuli))

        # labels and legends
        ax1.set_ylabel('Stimuli, s')
        ax2.set_ylabel('Weights, w')
        ax3.set_ylabel('Reward, R')
        ax3.set_xlabel('Trial, t')
        ax2.legend()
        ax3.legend()

        if minW < 0:
            ax2.add_line(plt.Line2D([0, N_trial-1], [0, 0], color='black', linewidth=0.2))
        if minV < 0 or minR < 0:
            ax3.add_line(plt.Line2D([0, N_trial-1], [0, 0], color='black', linewidth=0.2))

        return [ax1, ax2, ax3]

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

        # convert state history from integers to one-hot encoding
        S_history = np.array([np.eye(self.n_states)[s] if s is not None else np.zeros(self.n_states) for s in self.S_history[episode]]).T
        max_V, min_V, max_TD, min_TD, max_R, min_R = 0,1,0,1,0,1
        for ep in range(len(self.V_history)):
            max_V = max(max_V, np.max(self.V_history[ep]))
            min_V = min(min_V, np.min(self.V_history[ep]))
            max_TD = max(max_TD, np.max(self.TD_history[ep]))
            min_TD = min(min_TD, np.min(self.TD_history[ep]))
            max_R = max(max_R, np.max(self.R_history[ep]))
            min_R = min(min_R, np.min(self.R_history[ep]))
        
        
        if axs is None:
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(3, 2, height_ratios=[4, 1, 1], width_ratios=[4, 1], hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
            ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax4 = fig.add_subplot(gs[2, 0], sharex=ax1)
            axs = [ax1, ax2, ax3, ax4]

        [ax1, ax2, ax3, ax4] = axs
        fig = ax1.get_figure()

        for ax in axs: 
            ax.spines['left'].set_position(('outward', 3))
            ax.spines['bottom'].set_position(('outward', 3))

        # Central big plot shows the states 
        ax1.imshow(S_history, cmap='Greys', aspect='auto')
        ax1.set_ylabel("State, S", rotation=0, labelpad=20)
        ax1.spines['left'].set_bounds(0, self.n_states-1)
        ax1.spines['bottom'].set_bounds(0, T-1)
        ax1.xaxis.set_visible(False)

        # Right plot sharing y-axis with the central plot
        ax2.fill_betweenx(np.arange(self.n_states), self.V_history[episode], 0, color='C0', alpha=0.5, linewidth=0)
        ax2.spines['left'].set_bounds(0, self.n_states-1)
        ax2.set_xlabel('Value, V(S)')
        ax2.set_xlim([min_V, max_V])
        ax2.yaxis.set_visible(False)
        ax2.set_xticks([0])
        if min_V < 0: 
            ax2.add_line(plt.Line2D([0, 0], [0, self.n_states-1], color='black', linewidth=0.2))


        # First bottom plot sharing x-axis with the central plot
        # ax3.fill_between(np.arange(T), self.R_history[episode], 0, color='orange', alpha=0.5, linewidth=0)
        ax3.bar(np.arange(T), self.R_history[episode], color='orange', alpha=0.5, linewidth=0)
        ax3.spines['bottom'].set_bounds(0, T-1)
        ax3.set_ylabel('Reward, R', rotation=0, labelpad=20)
        ax3.set_ylim([min_R, max_R])
        ax3.xaxis.set_visible(False)
        if min_R < 0: 
            ax3.add_line(plt.Line2D([0, T-1], [0, 0], color='black', linewidth=0.2))
        
        # Second bottom plot sharing x-axis with the central plot
        x_new = np.linspace(0, T-1, 1000)
        
        y_new = np.interp(x_new, np.arange(T), self.TD_history[episode])
        ax4.fill_between(x_new, y_new, 0, where=y_new >= 0, color='green', alpha=0.5, linewidth=0)
        ax4.fill_between(x_new, y_new, 0, where=y_new < 0, color='red', alpha=0.5, linewidth=0)
        ax4.spines['bottom'].set_bounds(0, T-1)
        ax4.set_ylabel('TD error', rotation=0, labelpad=20)
        ax4.set_xlabel('Time step, t')
        ax4.set_ylim([min_TD, max_TD])
        if min_TD < 0: 
            ax4.add_line(plt.Line2D([0, T-1], [0, 0], color='black', linewidth=0.2))

        fig.suptitle(f'Episode {episode}')
        fig.subplots_adjust(left=0.3, bottom=0.2, right=0.9, top=0.9, wspace=None, hspace=None) #remove border
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
        anim = FuncAnimation(plt.gcf(), update, frames=episodes, fargs=(axs,), interval=150)        
        plt.close()
        return anim
        