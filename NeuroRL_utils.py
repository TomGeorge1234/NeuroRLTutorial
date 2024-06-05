"""Utils (mostly plotting) for NeuroRL tutorial"""

from typing import Any
import numpy as np
import xarray as xr 
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
            self.stim_names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

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
                ax2.scatter(t, W_history[t,i], label=f'Stimulus {self.stim_names[i]} weight' if t == 0 else None, color=f'C{i+1}', linewidth=0, s=8, alpha=0.7)
            ax3.scatter(t, V_history[t], label='Predicted (s.w)' if t==0 else None, s=8, color='C0', linewidth=0, alpha=0.7)
            ax3.scatter(t, R_history[t], label='Actual (reward recieved)' if t==0 else None, color='orange', linewidth=0, alpha=0.7)
        
        # format axes 
        ax1 = format_axes(ax1, ylims=(0, self.n_stimuli-1), xlims=(0, N_trial-1))
        ax1.set_yticks(np.arange(self.n_stimuli))
        ax1.set_yticklabels([f'{self.stim_names[i]}' for i in range(self.n_stimuli)])
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

        self.theoretical_value = None # if not None, this is will be plotted ontop 

    def learn_episode(self, 
                    states : np.ndarray, 
                    rewards : np.ndarray,):
            """
            States = the states visited in the episode, aka a list of integers
                    [S0, S1, S2, S3, ...]
            rewards = the rewards received after each state in the episode, aka a list of floats
                    [R1, R2, R3, R4, ...]
                    """
            states = np.array(states)
            rewards = np.array(rewards)
            assert len(states) == len(rewards), "States and rewards must be the same length"

            T_episode = len(states)+ 1 # get the length of the episode (including the initial None state)

            # Insert an unrewarded "None" state at the beginning to indicate the start of an episode
            rewards = np.insert(rewards, 0, 0)
            states = np.insert(states.astype(object), [0, len(states)], [None, None])

            # Loop over the states and learn from each transition
            TD_errors = np.empty(T_episode) # store the TD errors
            for i in range(T_episode):
                # Learn from this transition
                TD_errors[i] = self.learn(states[i], states[i+1], rewards[i])
            
            # Save to history 
            self.V_history.append(self.V.copy())
            self.S_history.append(states[:T_episode])
            self.TD_history.append(TD_errors)
            self.R_history.append(rewards)
    
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
            fig = plt.figure(figsize=(3, 3))
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
        ax2.set_xticks([0, max_V])
        ax2.set_xticklabels([f'{0.0:.1f}', f'{max_V:.1f}'])
        if min_V < 0: 
            ax2.add_line(plt.Line2D([0, 0], [0, self.n_states-1], color='black', linewidth=0.2))
        if self.theoretical_value is not None:
            ax2.plot(self.theoretical_value, np.arange(self.n_states), color='k', linestyle='--', linewidth=0.5, label='Theory')
            ax2.legend(loc="right")


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
        anim = FuncAnimation(plt.gcf(), update, frames=episodes, fargs=(axs,), interval=50)        
        plt.close()
        return anim
        
class BaseTDQLearner(BaseTDLearner):
    def __init__(self, gamma=0.5, alpha=0.1, n_states=10, n_actions=4):
        self.n_actions = n_actions
        # additional history arrays for storing Q values and actions
        self.Q_history = []
        self.A_history = []

        super(BaseTDQLearner, self).__init__(gamma=gamma, alpha=alpha, n_states=n_states)

    def learn_episode(
        self,
        states : np.ndarray, 
        actions : np.ndarray,
        rewards : np.ndarray,):

        T_episode = len(states) + 1 # get the length of the episode (including the initial None state)

        # Insert an unrewarded "None" state at the beginning to indicate the start of an episode
        rewards = np.insert(rewards, 0, 0)
        states = np.insert(states.astype(object), [0, len(states)], [None, None])
        actions = np.insert(actions.astype(object), [0, len(actions)], [None, None])

        # Loop over the states and learn from each transition
        TD_errors = np.empty(T_episode) # store the TD errors
        for i in range(T_episode):
            # Learn from this transition
            TD_errors[i] = self.learn(states[i], states[i+1], actions[i], actions[i+1], rewards[i])
        
        # Save to history 
        self.Q_history.append(self.Q.copy())
        self.S_history.append(states[:T_episode])
        self.A_history.append(states[:T_episode])
        self.TD_history.append(TD_errors)
        self.R_history.append(rewards)
        self.V_history.append(np.mean(self.Q, axis=1))

        return 




class MiniGrid():
    def __init__(
        self,
        grid : np.ndarray,
        reward_locations : list = [(5,5)],
        reward_values : list = [10],
        init_agent_pos : tuple = (10,10),
        max_steps = 100,
    ):
        self.grid = grid
        self.reward_locations = reward_locations
        self.reward_values = reward_values
        assert len(reward_locations) == len(reward_values), "Reward locations and values must be the same length"
        self.width = grid.shape[1]
        self.height = grid.shape[0]
        self.x_coords = np.arange(self.width)
        self.y_coords = np.arange(self.height)[::-1]
        self.agent_pos = init_agent_pos
        self.agent_direction = 0
        self.max_steps = max_steps

        self.episode_number = 0 
        self.recent_episode_length = self.max_steps
        self.episode_history = {}

        self.grid_shape = (self.height, self.width)
        self.n_states = self.width * self.height
        self.n_actions = 4
        self.state_action_index = np.arange(self.n_states).reshape(self.grid_shape)
        self.action_name_dict = {0: "Q(s,↑)", 1: "Q(s,→)", 2: "Q(s,↓)", 3: "Q(s,←)"}

    def step(self, action):
        raise NotImplementedError
    
    def policy(self, Q_values):
        raise NotImplementedError

    def reset(self):
        is_illegal_position = True
        while is_illegal_position:
            self.agent_pos = (np.random.randint(self.width), np.random.randint(self.height))
            is_illegal_position = self.is_wall(self.agent_pos)
        self.agent_direction = np.random.randint(4)

    def is_wall(self, pos : tuple): 
        (x, y) = pos
        return self.grid[self.height - 1 - y, x] # y is flipped because of the way the grid is plotted
        
    def get_reward(self, pos : tuple):
        if pos in self.reward_locations:
            return self.reward_values[self.reward_locations.index(pos)]
        else:
            return 0
    
    def plot_episode(self, episode_number=None, ax=None):
        if episode_number is None:
            episode_number = self.episode_number - 1
        else: 
            episode_number = np.arange(self.episode_number)[episode_number]
        episode_data = self.episode_history[episode_number]
        positions = episode_data["positions"]
        actions = episode_data["actions"]
        ax = self._plot_env(ax=ax)
        ax = self._plot_rewards(ax)
        for i in range(len(positions)):
            episode_frac = i/len(positions)
            ax = self._plot_agent(agent_pos=positions[i], agent_direction=actions[i], ax=ax, color = plt.cm.viridis(1-episode_frac), size_scaler=(1+episode_frac)/2, alpha = (1+episode_frac)/2)
        
        return

    def plot_Q(self, Q_values):
        assert (Q_values.shape == (self.n_states, self.n_actions)), f"Q_values must be of shape (n_states, n_actions) = ({self.n_states}, {self.n_actions}) but got {Q_values.shape}"
        Q_values = Q_values.reshape(self.grid_shape[0], self.grid_shape[1], self.n_actions)[::-1,:]
        min_Q = np.min(Q_values)
        max_Q = np.max(Q_values)

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(3, 3, hspace=0.15, wspace=0.15)
        axN = fig.add_subplot(gs[0, 1])
        axW = fig.add_subplot(gs[1, 0])
        axE = fig.add_subplot(gs[1, 2])
        axS = fig.add_subplot(gs[2, 1])
        axC = fig.add_subplot(gs[1, 1])
        axs = [axN, axE, axS, axW, axC]
        for i in range(4):
            axs[i] = self._plot_env(ax=axs[i])
            axs[i] = self._plot_rewards(ax=axs[i])
            axs[i].imshow(Q_values[:,:,i], cmap="viridis", extent=[-0.5, self.width-0.5, -0.5, self.height-0.5], alpha=0.5, vmax=max_Q, vmin=min_Q)
            axs[i].set_title(f"{self.action_name_dict[i]}")
            if i > 0:
                axs[i].set_yticks([])
                axs[i].set_ylabel("")
        axC = self._plot_env(ax=axC)
        axC = self._plot_rewards(ax=axC)
        axC = self._plot_policy(Q_values, ax=axC)
        axC.set_title("π(s)")
        # axC.imshow(Q_values.max(axis=2), cmap="viridis", extent=[-0.5, self.width-0.5, -0.5, self.height-0.5], alpha=0.5, vmax=max_Q, vmin=min_Q)
        return axs
    
    def plot_training(self):
        fig, ax = plt.subplots(figsize=(2, 1))
        episode_lengths = np.array([len(self.episode_history[i]["states"]) for i in range(self.episode_number)])
        n_eps = len(episode_lengths)
        smoothed_episode_lengths = [np.mean(episode_lengths[max(0, i-100):i+1]) for i in range(n_eps)]
        ax.scatter(np.arange(self.episode_number), episode_lengths, linewidth=0, alpha=0.5,c='C0', label="Episode length")
        ax.plot(np.arange(len(smoothed_episode_lengths)), smoothed_episode_lengths, color='k',linestyle="--",linewidth=0.5, label = "Smoothed")
        ax.set_xlabel("Episode")
        ax.legend()
        ax = format_axes(ax)

        return ax
    
    def render(self, ax=None):
        ax = self._plot_env(ax=ax)
        ax = self._plot_agent(ax=ax,agent_pos=self.agent_pos, agent_direction=self.agent_direction)
        ax = self._plot_rewards(ax)
        return ax 

    def plot_first_and_last_5_episodes(self):
        fig, ax = plt.subplots(2,5, figsize=(5,4))
        for i in range(5):
            self.plot_episode(i, ax=ax[0,i])
            self.plot_episode(-i-1, ax=ax[1,i])
            if i == 2: 
                ax[0,i].set_title("First 5 episodes")
                ax[1,i].set_title("Last 5 episodes")
        return ax
    
    def _plot_env(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(0.1*self.width, 0.1*self.height))
        ax.imshow(self.grid, cmap="Greys", zorder=20, alpha=self.grid.astype(float), extent=[-0.5, self.width-0.5, -0.5, self.height-0.5])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        return ax 

    def _plot_rewards(self, ax, **kwargs):
        for reward in self.reward_locations:
            reward_patch = plt.Circle(reward, 0.4, color='orange')
            ax.add_patch(reward_patch)
            ax.text(reward[0], reward[1], "R", color='white', fontsize=4, ha='center', va='center',alpha=0.5)
        return ax
    
    def _plot_arrow(self, pos, direction, ax, **kwargs):
        alpha = kwargs.get("alpha", 1)
        color = kwargs.get("color", plt.cm.viridis(0.5))
        size_scaler = kwargs.get("size_scaler", 1)
        # Define the base triangle vertices relative to origin
        base_triangle = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]) - np.array([0.5,0.5])
        scaled_triangle = base_triangle * size_scaler
        # Define the rotation matrix based on direction
        if direction==0: angle=0  # North
        elif direction==1: angle=3*np.pi/2 # East
        elif direction==2: angle=np.pi # South
        elif direction==3: angle=np.pi/2 # West
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                    [np.sin(angle), np.cos(angle)]])
        # Rotate the triangle
        rotated_triangle = np.dot(scaled_triangle, rotation_matrix.T)   
        # Translate the triangle to the agent position
        translated_triangle = rotated_triangle + np.array(pos)
        # Plot the triangle
        triangle = plt.Polygon(translated_triangle, color=color, alpha=alpha, linewidth=0)
        ax.add_patch(triangle)
        return ax
    
    def _plot_agent(self, ax, agent_pos, agent_direction,  **kwargs):
        ax = self._plot_arrow(agent_pos, agent_direction, ax, **kwargs)
        return ax

    def _plot_policy(self, Q_values, ax=None): 
        Q_values = Q_values.reshape(self.grid_shape[0], self.grid_shape[1], self.n_actions)
        optimal_actions = Q_values.argmax(axis=2)
        if ax is None:
            ax = self._plot_env()
            ax = self._plot_rewards(ax)
        for x in self.x_coords:
            for y in self.y_coords:
                if self.is_wall((x,y)):
                    continue
                action = optimal_actions[y,x]
                ax = self._plot_arrow(pos=(x,y), direction=action, ax=ax, alpha=0.5, color = plt.cm.viridis(1), size_scaler=0.5)
        
        return ax

    
    def pos_to_state(self, pos):
        return pos[1] * self.width + pos[0] 

    def state_to_pos(self, state):
        return (state % self.width, state // self.width)
    
