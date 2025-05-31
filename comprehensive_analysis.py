import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.animation as animation

import seaborn as sns  
from scipy.linalg import expm
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy import stats

import os
import pickle
import io

import logging, time
logging.getLogger().setLevel(logging.INFO)

import itertools
import warnings

class TrajectoryData:
    """Base class"""
    def __init__(self):
        pass

class TrajectoryData_Sim2VR_SIMULATION(TrajectoryData):
    
    SIM2VR_STUDY_PATH = "hit_data"  # Your path
    
    TASK_CONDITION_KEYWORDS = {"difficulty": ("easy", "medium", "hard"), "effort": ("low", "mid", "high"),
                              "difficulty_strategy": ("easy", "medium", "hard"), "effort_strategy": ("low", "mid", "high"),
                               "ALL": ("easy", "medium", "hard", "low", "mid", "high")}
    
    independent_joints = [
     'elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion', 'pro_sup', 'deviation', 'flexion']
    
    independent_joints_ids = [25, 26, 28, 29, 30, 31, 32]
    
    def __init__(self, USER_ID, TASK_CONDITION="difficulty"):
        self.USER_ID = USER_ID
        self.TASK_CONDITION = TASK_CONDITION
        
        super().__init__()
        
        # Load state files
        _state_files = [f_complete for subdir in os.listdir(f"{self.SIM2VR_STUDY_PATH}/{USER_ID}/") for f in os.listdir(f"{self.SIM2VR_STUDY_PATH}/{USER_ID}/{subdir}") if (TASK_CONDITION in subdir or any([_kw in subdir for _kw in self.TASK_CONDITION_KEYWORDS[TASK_CONDITION]])) and (os.path.isfile(f_complete := os.path.join(f"{self.SIM2VR_STUDY_PATH}/{USER_ID}/{subdir}", f))) and f.endswith("states.csv")]
        data_VR_STUDY = pd.DataFrame()
        if len(_state_files) == 0:
            raise FileNotFoundError(f"No state files found for USER_ID={USER_ID} and TASK_CONDITION={TASK_CONDITION}.")
        
        for _run_id, _file in enumerate(sorted(_state_files)):
            _df = pd.read_csv(_file, header=1)
            _df = _df.rename(columns = {cn: cn.strip() for cn in _df.columns})
            _df = _df.set_index("timestamp")
            
            # Compute velocities and accelerations
            _df = pd.concat((_df, _df[[cn for cn in _df.columns if "_pos_" in cn]].apply(lambda x: savgol_filter(x, 15, 3, deriv=1, delta = np.median(np.diff(_df.index)), axis=0)).rename(columns={k: k.replace("_pos_", "_vel_") for k in _df.columns})), axis=1)
            _df = pd.concat((_df, _df[[cn for cn in _df.columns if "_pos_" in cn]].apply(lambda x: savgol_filter(x, 15, 3, deriv=2, delta = np.median(np.diff(_df.index)), axis=0)).rename(columns={k: k.replace("_pos_", "_acc_") for k in _df.columns})), axis=1)
            
            _df["RUN_ID"] = _run_id
            _df["RUN_ID_INFO"] = os.path.dirname(_file).split("/")[-1]
            data_VR_STUDY = pd.concat((data_VR_STUDY, _df))
        
        self.static_optimization_loaded = False
        
        # Read action data if available
        _action_data_files = [f_complete for subdir in os.listdir(f"{self.SIM2VR_STUDY_PATH}/{USER_ID}/") for f in os.listdir(f"{self.SIM2VR_STUDY_PATH}/{USER_ID}/{subdir}") if (TASK_CONDITION in subdir or any([_kw in subdir for _kw in self.TASK_CONDITION_KEYWORDS[TASK_CONDITION]])) and (os.path.isfile(f_complete := os.path.join(f"{self.SIM2VR_STUDY_PATH}/{USER_ID}/{subdir}", f))) and f.endswith("action_log.pickle")]
        self._actions_loaded = len(_action_data_files) > 0
        
        # Combine data
        self.data = pd.concat((data_VR_STUDY,), axis=1)
        
        # Read event files
        _event_files = [f_complete for subdir in os.listdir(f"{self.SIM2VR_STUDY_PATH}/{USER_ID}/") for f in os.listdir(f"{self.SIM2VR_STUDY_PATH}/{USER_ID}/{subdir}") if (TASK_CONDITION in subdir or any([_kw in subdir for _kw in self.TASK_CONDITION_KEYWORDS[TASK_CONDITION]])) and (os.path.isfile(f_complete := os.path.join(f"{self.SIM2VR_STUDY_PATH}/{USER_ID}/{subdir}/", f))) and f.endswith("events.csv")]
        self._num_episodes = len(_event_files)
        
        # Process event files
        indices_VR_STUDY = pd.DataFrame()
        target_spawns = pd.DataFrame()
        target_hits = pd.DataFrame()
        target_contacts = pd.DataFrame()
        target_misses = pd.DataFrame()
        target_info = pd.DataFrame()
        
        for _run_id, _file in enumerate(sorted(_event_files)):
            with open(_file, "r") as f:
                _lines = f.read().split("\n")
            
            _firstrowindex = np.where(["grid mapping" in _l for _l in _lines])[0][-1] + 1
            _lastrowindex = np.where(["episode statistics" in _l for _l in _lines])[0][0] - 1
            _df_events = pd.DataFrame([_l.split(",") for _l in _lines[_firstrowindex:_lastrowindex+1]], 
                                    columns=["timestamp", "type", "target_ID", "grid_ID", "velocity"])
            
            _df_events["type"] = _df_events["type"].apply(lambda x: x.strip())
            _df_events = _df_events.replace({"": None})
            
            # Process event data
            _df_events["target_ID"] = _df_events["target_ID"].apply(lambda x: int(x.replace("target ID", "").strip()) if x else None)
            _df_events["grid_ID"] = _df_events["grid_ID"].apply(lambda x: int(x.replace("grid ID", "").strip()) if x is not None else x)
            _df_events["velocity"] = _df_events["velocity"].apply(lambda x: np.fromstring(x.replace("velocity", "").strip(), dtype=float, sep=' ') if x is not None else x)
            _df_events["timestamp"] = _df_events["timestamp"].astype(float)
            _df_events = _df_events.set_index("timestamp")
            _df_events["RUN_ID"] = _run_id
            _df_events["RUN_ID_INFO"] = os.path.dirname(_file).split("/")[-1]
            
            # Extract individual event types
            _target_spawns = _df_events.loc[_df_events["type"].apply(str.strip) == "target_spawn"]
            _target_hits = _df_events.loc[_df_events["type"].apply(str.strip) == "target_hit"]
            _target_contacts = _df_events.loc[_df_events["type"].apply(str.strip) == "target_contact"]
            _target_misses = _df_events.loc[_df_events["type"].apply(str.strip) == "target_miss"]

            # Grid mapping
            _target_info = pd.DataFrame(
                np.vstack([np.fromstring(_i.replace("[", "").strip(), dtype=float, sep=" ") 
                          for _i in _lines[_firstrowindex-1].split("grid mapping")[-1].split("]")[:-1]]), 
                columns=["id", "j", "i", "local_x", "local_y", "global_x", "global_y", "global_z"]
            ).astype(dict(zip(["id", "j", "i", "local_x", "local_y", "global_x", "global_y", "global_z"], 
                             [int, int, int, float, float, float, float, float])))
            _target_info["RUN_ID"] = _run_id
            _target_info["RUN_ID_INFO"] = os.path.dirname(_file).split("/")[-1]
            
            # Concatenate all data
            indices_VR_STUDY = pd.concat((indices_VR_STUDY, _df_events))
            target_spawns = pd.concat((target_spawns, _target_spawns))
            target_hits = pd.concat((target_hits, _target_hits))
            target_contacts = pd.concat((target_contacts, _target_contacts))
            target_misses = pd.concat((target_misses, _target_misses))
            target_info = pd.concat((target_info, _target_info))
        
        # Store processed data
        self._indices_VR_STUDY = indices_VR_STUDY
        self.STUDY_DIRECTION_NUMS = self._indices_VR_STUDY["RUN_ID"].max() + 1
        self._target_spawns = target_spawns
        self._target_hits = target_hits
        self._target_contacts = target_contacts
        self._target_misses = target_misses
        self._target_info = target_info.set_index("id")
        
        # Compute target summary statistics
        self._compute_target_stats()
        print(f"SIM2VR SIMULATION -- {self._num_episodes} episodes identified.")
    
    @staticmethod
    def target_labels_map(x): 
        return "Hits" if x.endswith("_hit") else "Contacts" if x.endswith("_contact") else "Misses" if x.endswith("_miss") else x
    
    @staticmethod
    def task_conditions_map(x): 
        return x.split('-')[-1].capitalize()
    
    def _compute_target_stats(self):
        _df_target_counts = self._indices_VR_STUDY.loc[self._indices_VR_STUDY["type"] != "target_spawn"].groupby(by="RUN_ID_INFO")["type"].value_counts().rename("counts").reset_index("type").pivot(columns="type").droplevel(level=0, axis=1)
        _df_target_counts = _df_target_counts.fillna(0).sort_index()[[cn for cn in ("target_hit", "target_contact", "target_miss") if cn in _df_target_counts.columns]]
        for cn in ("target_hit", "target_contact", "target_miss"):
            if cn not in _df_target_counts.columns:
                _df_target_counts[cn] = 0
        self.target_stats = _df_target_counts[["target_hit", "target_contact", "target_miss"]]
    
    def compute_hit_rates_by_position(self):
        """Compute hit rates by target position for heatmap visualization."""
        if not hasattr(self, 'target_stats'):
            self._compute_target_stats()
        
        target_events_with_positions = []
        
        for event_type in ['target_hit', 'target_contact', 'target_miss']:
            event_attr = f'_{event_type}' + ('s' if event_type != 'target_miss' else 'es')
            events_df = getattr(self, event_attr)
            
            for _, event in events_df.iterrows():
                grid_id = event['grid_ID']
                run_id = event['RUN_ID']
                
                target_pos = self._target_info[(self._target_info.index == grid_id) & 
                                            (self._target_info['RUN_ID'] == run_id)]
                
                if not target_pos.empty:
                    target_events_with_positions.append({
                        'event_type': event_type,
                        'grid_id': grid_id,
                        'target_x': target_pos.iloc[0]['global_x'],
                        'target_y': target_pos.iloc[0]['global_y'],
                        'local_x': target_pos.iloc[0]['local_x'],
                        'local_y': target_pos.iloc[0]['local_y'],
                        'j': target_pos.iloc[0]['j'],
                        'i': target_pos.iloc[0]['i']
                    })
        
        events_df = pd.DataFrame(target_events_with_positions)
        
        if events_df.empty:
            print("No target events found!")
            return None
        
        position_stats = events_df.groupby(['local_x', 'local_y', 'j', 'i'])['event_type'].value_counts().unstack(fill_value=0)
        position_stats['total'] = position_stats.sum(axis=1)
        position_stats['hit_rate'] = position_stats.get('target_hit', 0) / position_stats['total']
        position_stats['contact_rate'] = (position_stats.get('target_hit', 0) + 
                                        position_stats.get('target_contact', 0)) / position_stats['total']
        
        self.hit_rates_by_position = position_stats
        return position_stats

    def plot_hit_rate_heatmap_seaborn(self, use_contact_rate=False, figsize=(8, 6), title_suffix=""):
        """Plot hit rates heatmap using seaborn."""
        if not hasattr(self, 'hit_rates_by_position'):
            self.compute_hit_rates_by_position()
        
        if self.hit_rates_by_position is None:
            print("Cannot create heatmap: no position data available")
            return None, None
        
        rate_column = 'contact_rate' if use_contact_rate else 'hit_rate'
        
        df_reset = self.hit_rates_by_position.reset_index()
        pivot_data = df_reset.pivot_table(
            values=rate_column, 
            index='local_y',
            columns='local_x',
            aggfunc='mean'
        )
        
        pivot_data = pivot_data.sort_index(ascending=False)
        pivot_data = pivot_data.sort_index(axis=1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(pivot_data, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='RdYlBu_r',
                    vmin=0, 
                    vmax=1,
                    cbar_kws={'label': 'Rate'},
                    ax=ax,
                    annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        ax.set_xlabel('Target pos. (x)')
        ax.set_ylabel('Target pos. (y)')
        
        rate_type = 'Contact' if use_contact_rate else 'Hit'
        title = f'{self.USER_ID}\nTarget {rate_type} Rates{title_suffix}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, ax

    def plot_performance_timeline(self, figsize=(15, 6)):
        """Plot target events over time."""
        events = self._indices_VR_STUDY
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = {'target_hit': 'green', 'target_contact': 'orange', 'target_miss': 'red'}
        markers = {'target_hit': 'o', 'target_contact': 's', 'target_miss': 'x'}
        
        for event_type in ['target_hit', 'target_contact', 'target_miss']:
            event_data = events[events['type'] == event_type]
            if not event_data.empty:
                ax.scatter(event_data.index, [event_type]*len(event_data), 
                          label=self.target_labels_map(event_type), 
                          alpha=0.7, s=50, c=colors[event_type], marker=markers[event_type])
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Event Type')
        ax.legend()
        ax.set_title(f'{self.USER_ID} - Target Events Timeline')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax

    def plot_hit_rates_by_condition(self, figsize=(10, 6)):
        """Plot hit rates by task condition."""
        stats = self.target_stats.copy()
        stats['total'] = stats.sum(axis=1)
        stats['hit_rate'] = stats['target_hit'] / stats['total']
        stats['condition'] = stats.index.map(self.task_conditions_map)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if len(stats['condition'].unique()) > 1:
            sns.boxplot(data=stats.reset_index(), x='condition', y='hit_rate', ax=ax)
        else:
            sns.barplot(data=stats.reset_index(), x='condition', y='hit_rate', ax=ax)
        
        ax.set_title(f'{self.USER_ID} - Hit Rates by Task Condition')
        ax.set_ylabel('Hit Rate')
        ax.set_xlabel('Condition')
        
        plt.tight_layout()
        return fig, ax

    def plot_joint_trajectories(self, joint_names=None, figsize=(12, 10)):
        """Plot joint position trajectories."""
        if joint_names is None:
            # Use first 4 available joints
            available_joints = [j for j in self.independent_joints if f'{j}_pos' in self.data.columns]
            joint_names = available_joints[:4] if available_joints else []
        
        if not joint_names:
            print("No joint position data available")
            return None, None
        
        fig, axes = plt.subplots(len(joint_names), 1, figsize=figsize, sharex=True)
        if len(joint_names) == 1:
            axes = [axes]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.data['RUN_ID'].unique())))
        
        for i, joint in enumerate(joint_names):
            for run_id, color in zip(self.data['RUN_ID'].unique(), colors):
                run_data = self.data[self.data['RUN_ID'] == run_id]
                if f'{joint}_pos' in run_data.columns:
                    axes[i].plot(run_data.index, run_data[f'{joint}_pos'], 
                               alpha=0.7, label=f'Run {run_id}', color=color)
            
            axes[i].set_ylabel(f'{joint} (rad)')
            axes[i].set_title(f'{joint} Position Over Time')
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'{self.USER_ID} - Joint Trajectories', fontsize=16)
        plt.tight_layout()
        return fig, axes

    def plot_movement_profiles(self, joint='elbow_flexion', figsize=(12, 10)):
        """Plot position, velocity, and acceleration profiles for a joint."""
        if f'{joint}_pos' not in self.data.columns:
            available_joints = [j for j in self.independent_joints if f'{j}_pos' in self.data.columns]
            if available_joints:
                joint = available_joints[0]
                print(f"Joint {joint} not found, using {joint} instead")
            else:
                print("No joint data available")
                return None, None
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.data['RUN_ID'].unique())))
        
        for run_id, color in zip(self.data['RUN_ID'].unique(), colors):
            run_data = self.data[self.data['RUN_ID'] == run_id]
            
            axes[0].plot(run_data.index, run_data[f'{joint}_pos'], alpha=0.7, 
                        color=color, label=f'Run {run_id}')
            axes[1].plot(run_data.index, run_data[f'{joint}_vel'], alpha=0.7, color=color)
            axes[2].plot(run_data.index, run_data[f'{joint}_acc'], alpha=0.7, color=color)
        
        axes[0].set_ylabel('Position (rad)')
        axes[1].set_ylabel('Velocity (rad/s)')
        axes[2].set_ylabel('Acceleration (rad/s²)')
        axes[2].set_xlabel('Time (s)')
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
        
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.suptitle(f'{self.USER_ID} - {joint} Movement Profile', fontsize=16)
        
        plt.tight_layout()
        return fig, axes

    def plot_target_distribution(self, figsize=(10, 8)):
        """Plot spatial distribution of targets."""
        target_info = self._target_info.reset_index()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if len(target_info['RUN_ID'].unique()) > 1:
            scatter = ax.scatter(target_info['global_x'], target_info['global_y'], 
                            c=target_info['RUN_ID'], cmap='tab10', s=100, alpha=0.7)
            plt.colorbar(scatter, label='Run ID')
        else:
            ax.scatter(target_info['global_x'], target_info['global_y'], 
                      s=100, alpha=0.7, c='blue')
        
        ax.set_xlabel('Global X Position (m)')
        ax.set_ylabel('Global Y Position (m)')
        ax.set_title(f'{self.USER_ID} - Target Position Distribution')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax

    def create_performance_dashboard(self, figsize=(16, 12)):
        """Create comprehensive performance dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Hit rates by episode
        stats = self.target_stats.copy()
        stats['total'] = stats.sum(axis=1)
        stats['hit_rate'] = stats['target_hit'] / stats['total']
        
        axes[0,0].bar(range(len(stats)), stats['hit_rate'], alpha=0.7)
        axes[0,0].set_title('Hit Rate by Episode')
        axes[0,0].set_ylabel('Hit Rate')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].grid(True, alpha=0.3)
        
        # Event type distribution
        event_counts = stats[['target_hit', 'target_contact', 'target_miss']].sum()
        colors = ['green', 'orange', 'red']
        wedges, texts, autotexts = axes[0,1].pie(event_counts.values, 
                                                labels=[self.target_labels_map(f'{l}_hit') for l in event_counts.index], 
                                                autopct='%1.1f%%', colors=colors)
        axes[0,1].set_title('Overall Event Distribution')
        
        # Hit rates over time (trend)
        axes[0,2].plot(range(len(stats)), stats['hit_rate'], marker='o', linewidth=2, markersize=6)
        axes[0,2].set_title('Hit Rate Trend')
        axes[0,2].set_xlabel('Episode')
        axes[0,2].set_ylabel('Hit Rate')
        axes[0,2].grid(True, alpha=0.3)
        
        # Position-based hit rate distribution
        if hasattr(self, 'hit_rates_by_position') and self.hit_rates_by_position is not None:
            pos_stats = self.hit_rates_by_position
            axes[1,0].hist(pos_stats['hit_rate'], bins=10, alpha=0.7, edgecolor='black')
            axes[1,0].set_title('Hit Rate Distribution by Position')
            axes[1,0].set_xlabel('Hit Rate')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].grid(True, alpha=0.3)
        else:
            axes[1,0].text(0.5, 0.5, 'Position data\nnot available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Hit Rate Distribution by Position')
        
        # Response times (if available)
        try:
            reaction_times = self.analyze_reaction_times()
            if reaction_times:
                axes[1,1].hist(reaction_times, bins=15, alpha=0.7, edgecolor='black')
                axes[1,1].axvline(np.mean(reaction_times), color='red', linestyle='--', 
                                 label=f'Mean: {np.mean(reaction_times):.3f}s')
                axes[1,1].set_title('Reaction Time Distribution')
                axes[1,1].set_xlabel('Reaction Time (s)')
                axes[1,1].set_ylabel('Frequency')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            else:
                axes[1,1].text(0.5, 0.5, 'No reaction time\ndata available', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Reaction Time Distribution')
        except:
            axes[1,1].text(0.5, 0.5, 'Reaction time\nanalysis failed', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Reaction Time Distribution')
        
        # Movement efficiency (if hammer data available)
        hammer_cols = [col for col in self.data.columns if 'hammer' in col.lower() and '_pos_' in col]
        if len(hammer_cols) >= 2:
            # Calculate movement path length
            x_col, y_col = hammer_cols[0], hammer_cols[1]
            path_lengths = []
            for run_id in self.data['RUN_ID'].unique():
                run_data = self.data[self.data['RUN_ID'] == run_id]
                dx = np.diff(run_data[x_col])
                dy = np.diff(run_data[y_col])
                path_length = np.sum(np.sqrt(dx**2 + dy**2))
                path_lengths.append(path_length)
            
            axes[1,2].bar(range(len(path_lengths)), path_lengths, alpha=0.7)
            axes[1,2].set_title('Movement Path Length by Episode')
            axes[1,2].set_xlabel('Episode')
            axes[1,2].set_ylabel('Path Length (m)')
            axes[1,2].grid(True, alpha=0.3)
        else:
            axes[1,2].text(0.5, 0.5, 'Movement data\nnot available', 
                          ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Movement Path Length by Episode')
        
        fig.suptitle(f'{self.USER_ID} - Performance Dashboard', fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig, axes

    def analyze_reaction_times(self):
        """Analyze reaction times between target spawn and hit."""
        spawns = self._target_spawns
        hits = self._target_hits
        
        if spawns.empty or hits.empty:
            return []
        
        reaction_times = []
        for _, spawn in spawns.iterrows():
            # Find corresponding hit
            matching_hits = hits[(hits['target_ID'] == spawn['target_ID']) & 
                               (hits['RUN_ID'] == spawn['RUN_ID']) &
                               (hits.index > spawn.name)]
            if not matching_hits.empty:
                reaction_time = matching_hits.index.min() - spawn.name
                reaction_times.append(reaction_time)
        
        return reaction_times

    def plot_movement_efficiency(self, figsize=(15, 10)):
        """Plot movement efficiency analysis."""
        # Find hammer position columns
        hammer_cols = [col for col in self.data.columns if 'hammer' in col.lower() and '_pos_' in col]
        
        if len(hammer_cols) < 2:
            print("Insufficient hammer position data for movement analysis")
            return None, None
        
        x_col, y_col = hammer_cols[0], hammer_cols[1]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.data['RUN_ID'].unique())))
        
        # Movement paths for each run
        for run_id, color in zip(self.data['RUN_ID'].unique(), colors):
            run_data = self.data[self.data['RUN_ID'] == run_id]
            axes[0,0].plot(run_data[x_col], run_data[y_col], 
                        alpha=0.7, label=f'Run {run_id}', color=color)
        
        axes[0,0].set_title('Hammer Movement Paths')
        axes[0,0].set_xlabel('X Position (m)')
        axes[0,0].set_ylabel('Y Position (m)')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_aspect('equal')
        
        # Movement speed over time
        vel_cols = [col for col in self.data.columns if 'hammer' in col.lower() and '_vel_' in col]
        if len(vel_cols) >= 2:
            x_vel_col, y_vel_col = vel_cols[0], vel_cols[1]
            # Create speed column for the data
            speed_data = np.sqrt(self.data[x_vel_col]**2 + self.data[y_vel_col]**2)
            
            for run_id, color in zip(self.data['RUN_ID'].unique(), colors):
                run_data = self.data[self.data['RUN_ID'] == run_id]
                run_speed = np.sqrt(run_data[x_vel_col]**2 + run_data[y_vel_col]**2)
                axes[0,1].plot(run_data.index, run_speed, alpha=0.7, color=color)
            
            axes[0,1].set_title('Movement Speed Over Time')
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].set_ylabel('Speed (m/s)')
            axes[0,1].grid(True, alpha=0.3)
            
            # Store speed data for later use
            self.data['speed'] = speed_data
        else:
            axes[0,1].text(0.5, 0.5, 'Velocity data\nnot available', 
                        ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Movement Speed Over Time')
        
        # Path length by episode
        path_lengths = []
        for run_id in self.data['RUN_ID'].unique():
            run_data = self.data[self.data['RUN_ID'] == run_id]
            dx = np.diff(run_data[x_col])
            dy = np.diff(run_data[y_col])
            path_length = np.sum(np.sqrt(dx**2 + dy**2))
            path_lengths.append(path_length)
        
        axes[1,0].bar(range(len(path_lengths)), path_lengths, alpha=0.7)
        axes[1,0].set_title('Total Path Length by Episode')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Path Length (m)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Speed distribution
        if 'speed' in self.data.columns:
            speed_values = self.data['speed'].dropna()
            if len(speed_values) > 0:
                axes[1,1].hist(speed_values, bins=30, alpha=0.7, edgecolor='black')
                axes[1,1].axvline(speed_values.mean(), color='red', linestyle='--', 
                                label=f'Mean: {speed_values.mean():.3f} m/s')
                axes[1,1].set_title('Speed Distribution')
                axes[1,1].set_xlabel('Speed (m/s)')
                axes[1,1].set_ylabel('Frequency')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            else:
                axes[1,1].text(0.5, 0.5, 'No speed data\navailable', 
                            ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Speed Distribution')
        else:
            axes[1,1].text(0.5, 0.5, 'Speed data\nnot calculated', 
                        ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Speed Distribution')
        
        fig.suptitle(f'{self.USER_ID} - Movement Efficiency Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig, axes
    
    def preprocess(self):
        """Basic preprocessing method."""
        print("Preprocessing completed.")
        self.preprocessed = True

# ANALYSIS FUNCTIONS
def create_hit_rate_heatmap(user_id, task_condition="difficulty", use_seaborn=True):
    """Create a hit rate heatmap by target position."""
    
    trajectory_data = TrajectoryData_Sim2VR_SIMULATION(user_id, task_condition)
    trajectory_data.preprocess()
    
    position_stats = trajectory_data.compute_hit_rates_by_position()
    
    if position_stats is None:
        print("Could not compute position-based hit rates")
        return trajectory_data, None, None
    
    fig, ax = trajectory_data.plot_hit_rate_heatmap_seaborn()
    
    print(f"Hit Rate Heatmap for {user_id}")
    print("=" * 40)
    print(f"Positions analyzed: {len(position_stats)}")
    print(f"Mean hit rate: {position_stats['hit_rate'].mean():.3f}")
    print(f"Hit rate range: {position_stats['hit_rate'].min():.3f} - {position_stats['hit_rate'].max():.3f}")
    
    return trajectory_data, fig, ax

def comprehensive_analysis(user_id, task_condition="difficulty", save_plots=True):
    """Generate comprehensive analysis with multiple visualizations."""
    
    print(f"Loading data for {user_id}...")
    trajectory_data = TrajectoryData_Sim2VR_SIMULATION(user_id, task_condition)
    trajectory_data.preprocess()
    
    # Generate all plots
    plots = {}
    
    print("Generating visualizations...")
    
    # 1. Hit rate heatmap
    print("  - Hit rate heatmap")
    trajectory_data.compute_hit_rates_by_position()
    plots['heatmap'] = trajectory_data.plot_hit_rate_heatmap_seaborn()
    
    # 2. Performance timeline
    print("  - Performance timeline")
    plots['timeline'] = trajectory_data.plot_performance_timeline()
    
    # 3. Performance dashboard
    print("  - Performance dashboard")
    plots['dashboard'] = trajectory_data.create_performance_dashboard()
    
    # 4. Hit rates by condition
    print("  - Hit rates by condition")
    plots['conditions'] = trajectory_data.plot_hit_rates_by_condition()
    
    # 5. Joint trajectories
    print("  - Joint trajectories")
    plots['joints'] = trajectory_data.plot_joint_trajectories()
    
    # 6. Movement profiles
    print("  - Movement profiles")
    plots['profiles'] = trajectory_data.plot_movement_profiles()
    
    # 7. Target distribution
    print("  - Target distribution")
    plots['distribution'] = trajectory_data.plot_target_distribution()
    
    # 8. Movement efficiency
    print("  - Movement efficiency")
    plots['efficiency'] = trajectory_data.plot_movement_efficiency()
    
    # Save plots if requested
    if save_plots:
        print("Saving plots...")
        for plot_name, (fig, _) in plots.items():
            if fig is not None:
                filename = f'{user_id}_{plot_name}.png'
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  - Saved {filename}")
    
    # Display all plots
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print(f"ANALYSIS SUMMARY FOR {user_id}")
    print("="*60)
    
    # Basic stats
    stats = trajectory_data.target_stats
    total_targets = stats.sum().sum()
    total_hits = stats['target_hit'].sum()
    total_contacts = stats['target_contact'].sum()
    total_misses = stats['target_miss'].sum()
    
    print(f"Total targets: {total_targets}")
    print(f"Hits: {total_hits} ({total_hits/total_targets*100:.1f}%)")
    print(f"Contacts: {total_contacts} ({total_contacts/total_targets*100:.1f}%)")
    print(f"Misses: {total_misses} ({total_misses/total_targets*100:.1f}%)")
    print(f"Success rate (hits + contacts): {(total_hits + total_contacts)/total_targets*100:.1f}%")
    
    # Episode stats
    print(f"\nNumber of episodes: {len(stats)}")
    hit_rates = stats['target_hit'] / stats.sum(axis=1)
    print(f"Mean hit rate per episode: {hit_rates.mean():.3f} ± {hit_rates.std():.3f}")
    print(f"Best episode hit rate: {hit_rates.max():.3f}")
    print(f"Worst episode hit rate: {hit_rates.min():.3f}")
    
    # Reaction time stats
    try:
        reaction_times = trajectory_data.analyze_reaction_times()
        if reaction_times:
            print(f"\nReaction times:")
            print(f"Mean: {np.mean(reaction_times):.3f}s ± {np.std(reaction_times):.3f}s")
            print(f"Range: {np.min(reaction_times):.3f}s - {np.max(reaction_times):.3f}s")
    except:
        print("\nReaction time analysis not available")
    
    # Position-based stats
    if hasattr(trajectory_data, 'hit_rates_by_position') and trajectory_data.hit_rates_by_position is not None:
        pos_stats = trajectory_data.hit_rates_by_position
        print(f"\nPosition-based analysis:")
        print(f"Positions analyzed: {len(pos_stats)}")
        print(f"Best position hit rate: {pos_stats['hit_rate'].max():.3f}")
        print(f"Worst position hit rate: {pos_stats['hit_rate'].min():.3f}")
        print(f"Position hit rate std: {pos_stats['hit_rate'].std():.3f}")
    
    print("="*60)
    
    return trajectory_data, plots

def quick_analysis(user_id, task_condition="difficulty"):
    """Quick analysis with just the key visualizations."""
    
    print(f"Quick analysis for {user_id}...")
    trajectory_data = TrajectoryData_Sim2VR_SIMULATION(user_id, task_condition)
    trajectory_data.preprocess()
    
    # Create a 2x2 subplot with key visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Hit rate heatmap
    trajectory_data.compute_hit_rates_by_position()
    if hasattr(trajectory_data, 'hit_rates_by_position') and trajectory_data.hit_rates_by_position is not None:
        df_reset = trajectory_data.hit_rates_by_position.reset_index()
        pivot_data = df_reset.pivot_table(values='hit_rate', index='local_y', columns='local_x', aggfunc='mean')
        pivot_data = pivot_data.sort_index(ascending=False).sort_index(axis=1)
        
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', vmin=0, vmax=1,
                    cbar_kws={'label': 'Hit Rate'}, ax=axes[0,0])
        axes[0,0].set_title('Hit Rate Heatmap')
    else:
        axes[0,0].text(0.5, 0.5, 'No position data', ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title('Hit Rate Heatmap')
    
    # Performance over episodes
    stats = trajectory_data.target_stats
    stats['hit_rate'] = stats['target_hit'] / stats.sum(axis=1)
    axes[0,1].plot(range(len(stats)), stats['hit_rate'], marker='o', linewidth=2)
    axes[0,1].set_title('Hit Rate Trend')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Hit Rate')
    axes[0,1].grid(True, alpha=0.3)
    
    # Event distribution
    event_counts = stats[['target_hit', 'target_contact', 'target_miss']].sum()
    colors = ['green', 'orange', 'red']
    axes[1,0].pie(event_counts.values, labels=['Hits', 'Contacts', 'Misses'], 
                  autopct='%1.1f%%', colors=colors)
    axes[1,0].set_title('Event Distribution')
    
    # Joint trajectory (if available)
    available_joints = [j for j in trajectory_data.independent_joints if f'{j}_pos' in trajectory_data.data.columns]
    if available_joints:
        joint = available_joints[0]
        for run_id in trajectory_data.data['RUN_ID'].unique():
            run_data = trajectory_data.data[trajectory_data.data['RUN_ID'] == run_id]
            axes[1,1].plot(run_data.index, run_data[f'{joint}_pos'], alpha=0.7, label=f'Run {run_id}')
        axes[1,1].set_title(f'{joint} Trajectory')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Position (rad)')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'No joint data', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Joint Trajectory')
    
    fig.suptitle(f'{user_id} - Quick Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Quick stats
    total_targets = stats.sum().sum()
    total_hits = stats['target_hit'].sum()
    success_rate = total_hits / total_targets * 100
    
    print(f"\nQuick Stats for {user_id}:")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Episodes: {len(stats)}")
    print(f"Mean Hit Rate: {stats['hit_rate'].mean():.3f}")
    
    return trajectory_data, fig

# MAIN EXECUTION FUNCTIONS
def generate_heatmap_only():
    """Generate just the heatmap (original function)."""
    user_id = "SIM_m_mono"
    task_condition = "difficulty"
    
    try:
        print(f"Loading data for {user_id}...")
        trajectory_data, fig, ax = create_hit_rate_heatmap(user_id, task_condition, use_seaborn=True)
        
        if fig is not None:
            plt.show()
            print("Success! Heatmap displayed.")
            
            fig.savefig(f'{user_id}_hit_rate_heatmap.png', dpi=300, bbox_inches='tight')
            print("Heatmap saved as PNG file.")
            
        return trajectory_data
        
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_all_plots():
    """Generate all available plots."""
    user_id = "SIM_m"
    task_condition = "difficulty"
    
    try:
        trajectory_data, plots = comprehensive_analysis(user_id, task_condition, save_plots=True)
        print("Success! All plots generated and displayed.")
        return trajectory_data, plots
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_quick_plots():
    """Generate quick analysis plots."""
    user_id = "SIM_m"
    task_condition = "difficulty"
    
    try:
        trajectory_data, fig = quick_analysis(user_id, task_condition)
        fig.savefig(f'{user_id}_quick_analysis.png', dpi=300, bbox_inches='tight')
        print("Success! Quick analysis completed and saved.")
        return trajectory_data, fig
        
    except Exception as e:
        print(f"Error in quick analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def plot_reaction_time_distribution(self, figsize=(10, 6)):
    """Plot detailed reaction time distribution analysis."""
    reaction_times = self.analyze_reaction_times()
    
    if not reaction_times:
        print("No reaction time data available")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    n, bins, patches = ax.hist(reaction_times, bins=20, alpha=0.7, edgecolor='black', 
                              color='steelblue', density=False)
    
    # Add statistics
    mean_rt = np.mean(reaction_times)
    median_rt = np.median(reaction_times)
    std_rt = np.std(reaction_times)
    
    # Add vertical lines for statistics
    ax.axvline(mean_rt, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_rt:.3f}s')
    ax.axvline(median_rt, color='orange', linestyle='-.', linewidth=2, 
               label=f'Median: {median_rt:.3f}s')
    
    # Add text with detailed statistics
    stats_text = f"""Statistics:
Mean: {mean_rt:.3f}s
Median: {median_rt:.3f}s
Std Dev: {std_rt:.3f}s
Min: {np.min(reaction_times):.3f}s
Max: {np.max(reaction_times):.3f}s
N: {len(reaction_times)} responses"""
    
    ax.text(0.65, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, fontfamily='monospace')
    
    # Formatting
    ax.set_xlabel('Reaction Time (s)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{self.USER_ID} - Reaction Time Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add percentile lines
    p25 = np.percentile(reaction_times, 25)
    p75 = np.percentile(reaction_times, 75)
    
    ax.axvline(p25, color='green', linestyle=':', alpha=0.7, 
               label=f'25th percentile: {p25:.3f}s')
    ax.axvline(p75, color='green', linestyle=':', alpha=0.7, 
               label=f'75th percentile: {p75:.3f}s')
    
    plt.tight_layout()
    return fig, ax

def plot_reaction_time_analysis(self, figsize=(15, 10)):
    """Comprehensive reaction time analysis with multiple views."""
    reaction_times = self.analyze_reaction_times()
    
    if not reaction_times:
        print("No reaction time data available")
        return None, None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Distribution histogram
    axes[0,0].hist(reaction_times, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    mean_rt = np.mean(reaction_times)
    axes[0,0].axvline(mean_rt, color='red', linestyle='--', linewidth=2, 
                     label=f'Mean: {mean_rt:.3f}s')
    axes[0,0].set_xlabel('Reaction Time (s)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Reaction Time Distribution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Box plot
    axes[0,1].boxplot(reaction_times, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    axes[0,1].set_ylabel('Reaction Time (s)')
    axes[0,1].set_title('Reaction Time Box Plot')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add statistics as text
    q1, median, q3 = np.percentile(reaction_times, [25, 50, 75])
    iqr = q3 - q1
    stats_text = f'Q1: {q1:.3f}s\nMedian: {median:.3f}s\nQ3: {q3:.3f}s\nIQR: {iqr:.3f}s'
    axes[0,1].text(0.02, 0.98, stats_text, transform=axes[0,1].transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. Cumulative distribution
    sorted_rt = np.sort(reaction_times)
    cumulative = np.arange(1, len(sorted_rt) + 1) / len(sorted_rt)
    axes[1,0].plot(sorted_rt, cumulative, marker='.', linestyle='-', color='purple')
    axes[1,0].set_xlabel('Reaction Time (s)')
    axes[1,0].set_ylabel('Cumulative Probability')
    axes[1,0].set_title('Cumulative Distribution Function')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add percentile markers
    for p in [25, 50, 75, 90, 95]:
        rt_p = np.percentile(reaction_times, p)
        axes[1,0].axvline(rt_p, color='red', alpha=0.5, linestyle='--')
        axes[1,0].text(rt_p, p/100, f'P{p}', rotation=90, verticalalignment='bottom')
    
    # 4. Reaction time over trial sequence (if we can extract trial order)
    # Try to get reaction times in chronological order
    spawns = self._target_spawns
    hits = self._target_hits
    
    if not spawns.empty and not hits.empty:
        rt_chronological = []
        hit_times = []
        
        for _, spawn in spawns.iterrows():
            matching_hits = hits[(hits['target_ID'] == spawn['target_ID']) & 
                               (hits['RUN_ID'] == spawn['RUN_ID']) &
                               (hits.index > spawn.name)]
            if not matching_hits.empty:
                rt = matching_hits.index.min() - spawn.name
                rt_chronological.append(rt)
                hit_times.append(matching_hits.index.min())
        
        if rt_chronological:
            axes[1,1].plot(range(len(rt_chronological)), rt_chronological, 
                          marker='o', linestyle='-', alpha=0.7, color='darkgreen')
            axes[1,1].set_xlabel('Trial Number')
            axes[1,1].set_ylabel('Reaction Time (s)')
            axes[1,1].set_title('Reaction Time Over Trials')
            axes[1,1].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(range(len(rt_chronological)), rt_chronological, 1)
            p = np.poly1d(z)
            axes[1,1].plot(range(len(rt_chronological)), p(range(len(rt_chronological))), 
                          "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}s/trial')
            axes[1,1].legend()
        else:
            axes[1,1].text(0.5, 0.5, 'Cannot determine\ntrial sequence', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Reaction Time Over Trials')
    else:
        axes[1,1].text(0.5, 0.5, 'Trial sequence\ndata not available', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Reaction Time Over Trials')
    
    fig.suptitle(f'{self.USER_ID} - Comprehensive Reaction Time Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig, axes

TrajectoryData_Sim2VR_SIMULATION.plot_reaction_time_distribution = plot_reaction_time_distribution
TrajectoryData_Sim2VR_SIMULATION.plot_reaction_time_analysis = plot_reaction_time_analysis

def comprehensive_analysis_with_reaction_time(user_id, task_condition="difficulty", save_plots=True):
    """Generate comprehensive analysis with separate reaction time plot."""
    
    print(f"Loading data for {user_id}...")
    trajectory_data = TrajectoryData_Sim2VR_SIMULATION(user_id, task_condition)
    trajectory_data.preprocess()
    
    # Generate all plots
    plots = {}
    
    print("Generating visualizations...")
    
    # 1. Hit rate heatmap
    print("  - Hit rate heatmap")
    trajectory_data.compute_hit_rates_by_position()
    plots['heatmap'] = trajectory_data.plot_hit_rate_heatmap_seaborn()
    
    # 2. Performance timeline
    print("  - Performance timeline")
    plots['timeline'] = trajectory_data.plot_performance_timeline()
    
    # 3. Performance dashboard
    print("  - Performance dashboard")
    plots['dashboard'] = trajectory_data.create_performance_dashboard()
    
    # 4. Hit rates by condition
    print("  - Hit rates by condition")
    plots['conditions'] = trajectory_data.plot_hit_rates_by_condition()
    
    # 5. Joint trajectories
    print("  - Joint trajectories")
    plots['joints'] = trajectory_data.plot_joint_trajectories()
    
    # 6. Movement profiles
    print("  - Movement profiles")
    plots['profiles'] = trajectory_data.plot_movement_profiles()
    
    # 7. Target distribution
    print("  - Target distribution")
    plots['distribution'] = trajectory_data.plot_target_distribution()
    
    # 8. Movement efficiency
    print("  - Movement efficiency")
    plots['efficiency'] = trajectory_data.plot_movement_efficiency()
    
    # 9. Reaction time distribution (NEW)
    print("  - Reaction time distribution")
    plots['reaction_time'] = trajectory_data.plot_reaction_time_distribution()
    
    # 10. Comprehensive reaction time analysis (NEW)
    print("  - Comprehensive reaction time analysis")
    plots['reaction_time_analysis'] = trajectory_data.plot_reaction_time_analysis()
    
    # Save plots if requested
    if save_plots:
        print("Saving plots...")
        for plot_name, plot_result in plots.items():
            if plot_result[0] is not None:
                filename = f'{user_id}_{plot_name}.png'
                plot_result[0].savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  - Saved {filename}")
    
    # Display all plots
    plt.show()
    
    # Print summary statistics (same as before)
    print("\n" + "="*60)
    print(f"ANALYSIS SUMMARY FOR {user_id}")
    print("="*60)
    
    # Basic stats
    stats = trajectory_data.target_stats
    total_targets = stats.sum().sum()
    total_hits = stats['target_hit'].sum()
    total_contacts = stats['target_contact'].sum()
    total_misses = stats['target_miss'].sum()
    
    print(f"Total targets: {total_targets}")
    print(f"Hits: {total_hits} ({total_hits/total_targets*100:.1f}%)")
    print(f"Contacts: {total_contacts} ({total_contacts/total_targets*100:.1f}%)")
    print(f"Misses: {total_misses} ({total_misses/total_targets*100:.1f}%)")
    print(f"Success rate (hits + contacts): {(total_hits + total_contacts)/total_targets*100:.1f}%")
    
    # Episode stats
    print(f"\nNumber of episodes: {len(stats)}")
    hit_rates = stats['target_hit'] / stats.sum(axis=1)
    print(f"Mean hit rate per episode: {hit_rates.mean():.3f} ± {hit_rates.std():.3f}")
    print(f"Best episode hit rate: {hit_rates.max():.3f}")
    print(f"Worst episode hit rate: {hit_rates.min():.3f}")
    
    # Reaction time specific stats
    reaction_times = trajectory_data.analyze_reaction_times()
    if reaction_times:
        print(f"\nDETAILED REACTION TIME ANALYSIS:")
        print(f"Mean: {np.mean(reaction_times):.3f}s ± {np.std(reaction_times):.3f}s")
        print(f"Median: {np.median(reaction_times):.3f}s")
        print(f"Range: {np.min(reaction_times):.3f}s - {np.max(reaction_times):.3f}s")
        print(f"25th percentile: {np.percentile(reaction_times, 25):.3f}s")
        print(f"75th percentile: {np.percentile(reaction_times, 75):.3f}s")
        print(f"95th percentile: {np.percentile(reaction_times, 95):.3f}s")
        print(f"Sample size: {len(reaction_times)} responses")
    
    print("="*60)
    
    return trajectory_data, plots


# Run the script
if __name__ == "__main__":
    print("VR Simulation Analysis Tool")
    print("="*40)
    print("Available functions:")
    print("1. generate_heatmap_only() - Just the hit rate heatmap")
    print("2. generate_quick_plots() - Quick 2x2 analysis")
    print("3. generate_all_plots() - Comprehensive analysis with all plots")
    print("4. comprehensive_analysis_with_reaction_time() - All plots + detailed reaction time analysis")
    print()
    
    # Ask user what they want to run
    choice = input("Which analysis would you like to run? (1/2/3/4 or 'all' for comprehensive): ").strip().lower()
    
    if choice in ['1', 'heatmap']:
        print("Running heatmap analysis...")
        trajectory_data = generate_heatmap_only()
    elif choice in ['2', 'quick']:
        print("Running quick analysis...")
        trajectory_data, fig = generate_quick_plots()
    elif choice in ['3', 'all', 'comprehensive']:
        print("Running comprehensive analysis...")
        trajectory_data, plots = generate_all_plots()
    elif choice in ['4', 'reaction', 'rt']:
        print("Running comprehensive analysis with reaction time...")
        trajectory_data, plots = comprehensive_analysis_with_reaction_time("SIM_m", "difficulty")
    else:
        print("Invalid choice. Running comprehensive analysis by default...")
        trajectory_data, plots = generate_all_plots()

    # Example usage:
    # trajectory_data = TrajectoryData_Sim2VR_SIMULATION("SIM_m", "difficulty")
    # fig, axes = trajectory_data.plot_movement_efficiency()
    # plt.show()