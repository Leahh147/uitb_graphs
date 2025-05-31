import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import re
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')

# Import the existing class - make sure this file is in the same directory
# or adjust the import path as needed
from comprehensive_analysis import TrajectoryData_Sim2VR_SIMULATION

class ReactionTimeAnalyzer:
    """Reaction time analyzer that integrates with the existing framework."""
    
    def __init__(self, trajectory_data):
        """Initialize with a TrajectoryData_Sim2VR_SIMULATION instance."""
        self.trajectory_data = trajectory_data
        self.target_positions = {}
        self._extract_target_positions()
    
    def _extract_target_positions(self):
        """Extract target positions from the trajectory data."""
        try:
            # Get target info from the existing trajectory data
            target_info = self.trajectory_data._target_info
            
            for _, row in target_info.iterrows():
                grid_id = row.name  # index is the grid ID
                run_id = row['RUN_ID']
                x, y, z = row['global_x'], row['global_y'], row['global_z']
                self.target_positions[(run_id, grid_id)] = (x, y, z)
                
        except Exception as e:
            print(f"Warning: Could not extract target positions: {e}")
    
    def calculate_hand_velocity(self):
        """Calculate hand velocity from trajectory data."""
        # Use the existing velocity data if available, or calculate it
        data = self.trajectory_data.data.copy()
        
        if 'right_vel_x' not in data.columns:
            # Calculate velocity using the same method as the framework
            dt = data.index.to_series().diff()
            data['right_vel_x'] = data['right_pos_x'].diff() / dt
            data['right_vel_y'] = data['right_pos_y'].diff() / dt
            data['right_vel_z'] = data['right_pos_z'].diff() / dt
        
        # Calculate speed
        data['right_hand_speed'] = np.sqrt(
            data['right_vel_x']**2 + 
            data['right_vel_y']**2 + 
            data['right_vel_z']**2
        )
        
        # Fill NaN values
        velocity_cols = ['right_vel_x', 'right_vel_y', 'right_vel_z', 'right_hand_speed']
        for col in velocity_cols:
            if col in data.columns:
                data[col] = data[col].fillna(0)
        
        return data
    
    def find_movement_toward_target(self, segment, target_pos, spawn_time):
        """
        Find the first time when hand velocity points toward target.
        
        Uses dot product between velocity vector and direction-to-target vector.
        When dot product becomes positive and significant, hand is moving toward target.
        """
        if len(segment) < 2:
            return None
        
        target_x, target_y, target_z = target_pos
        
        # Calculate for each time point in segment
        for i, (timestamp, row) in enumerate(segment.iterrows()):
            hand_x = row.get('right_pos_x', 0)
            hand_y = row.get('right_pos_y', 0)
            hand_z = row.get('right_pos_z', 0)
            vel_x = row.get('right_vel_x', 0)
            vel_y = row.get('right_vel_y', 0)
            vel_z = row.get('right_vel_z', 0)
            
            # Vector from hand to target
            to_target_x = target_x - hand_x
            to_target_y = target_y - hand_y
            to_target_z = target_z - hand_z
            
            # Distance to target
            distance_to_target = np.sqrt(to_target_x**2 + to_target_y**2 + to_target_z**2)
            
            if distance_to_target < 0.001:  # Very close to target
                continue
            
            # Normalize direction vector
            to_target_x /= distance_to_target
            to_target_y /= distance_to_target
            to_target_z /= distance_to_target
            
            # Dot product: velocity · direction_to_target
            dot_product = vel_x * to_target_x + vel_y * to_target_y + vel_z * to_target_z
            
            # Speed in direction of target
            speed_toward_target = dot_product
            current_speed = row.get('right_hand_speed', 0)
            
            # Threshold: moving toward target with at least 10% of current speed
            speed_threshold = max(current_speed * 0.1, 0.05)  # At least 0.05 m/s toward target
            
            if speed_toward_target > speed_threshold and speed_toward_target > 0:
                return timestamp
        
        return None
    
    def calculate_reaction_times(self):
        """Calculate reaction times using directional movement toward target."""
        # Get the data with velocities
        data_with_velocity = self.calculate_hand_velocity()
        
        spawns = self.trajectory_data._target_spawns
        hits = self.trajectory_data._target_hits
        
        reaction_times = []
        grid_reaction_times = {i: [] for i in range(9)}
        
        for spawn_idx, spawn in spawns.iterrows():
            spawn_time = spawn_idx
            target_id = spawn['target_ID']
            grid_id = spawn['grid_ID']
            run_id = spawn['RUN_ID']
            
            # Get target position
            target_pos = self.target_positions.get((run_id, grid_id))
            if target_pos is None:
                continue
            
            # Find matching hit
            matching_hits = hits[
                (hits['target_ID'] == target_id) & 
                (hits['RUN_ID'] == run_id) & 
                (hits.index > spawn_time)
            ]
            
            if matching_hits.empty:
                continue
            
            hit_time = matching_hits.index.min()
            
            # Get trajectory segment from spawn to hit
            segment_mask = (
                (data_with_velocity.index >= spawn_time) & 
                (data_with_velocity.index <= hit_time) &
                (data_with_velocity['RUN_ID'] == run_id)
            )
            segment = data_with_velocity[segment_mask].copy()
            
            if len(segment) < 2:
                continue
            
            # Find when movement toward target starts
            time_that_move = self.find_movement_toward_target(segment, target_pos, spawn_time)
            
            if time_that_move is None:
                continue
            
            # Calculate reaction time
            reaction_time = time_that_move - spawn_time
            
            # Validate reaction time (must be positive and reasonable)
            if 0 < reaction_time < 2.0:  # Between 0 and 2 seconds
                reaction_times.append(reaction_time)
                grid_reaction_times[grid_id].append(reaction_time)
        
        return reaction_times, grid_reaction_times


class MultiUserReactionTimeComparison:
    """Multi-user reaction time comparison following the existing framework pattern."""
    
    def __init__(self, data_path="hit_data", task_condition="difficulty"):
        self.data_path = data_path
        self.task_condition = task_condition
        self.users_data = {}
        self.reaction_times_data = {}
        
        # Create output folder
        self.output_folder = "reaction-time-comparison"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")
    
    def load_all_users(self):
        """Load data for all available users."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")
        
        # Find all SIM_* folders
        user_folders = [f for f in os.listdir(self.data_path) 
                       if f.startswith('SIM_') and os.path.isdir(os.path.join(self.data_path, f))]
        
        print(f"Found {len(user_folders)} users: {user_folders}")
        
        for user_id in user_folders:
            try:
                print(f"Loading data for {user_id}...")
                trajectory_data = TrajectoryData_Sim2VR_SIMULATION(user_id, self.task_condition)
                trajectory_data.preprocess()
                self.users_data[user_id] = trajectory_data
                print(f"  ✓ Successfully loaded {user_id}")
            except Exception as e:
                print(f"  ✗ Failed to load {user_id}: {e}")
        
        print(f"Successfully loaded {len(self.users_data)} users")
        return self.users_data
    
    def calculate_all_reaction_times(self):
        """Calculate reaction times for all users."""
        print("\nCalculating reaction times for all users...")
        
        for user_id, trajectory_data in self.users_data.items():
            try:
                print(f"Processing {user_id}...")
                
                # Create reaction time analyzer
                rt_analyzer = ReactionTimeAnalyzer(trajectory_data)
                
                # Calculate reaction times
                reaction_times, grid_reaction_times = rt_analyzer.calculate_reaction_times()
                
                if reaction_times:
                    self.reaction_times_data[user_id] = {
                        'overall': reaction_times,
                        'by_grid': grid_reaction_times,
                        'mean': np.mean(reaction_times),
                        'std': np.std(reaction_times, ddof=1) if len(reaction_times) > 1 else 0,
                        'median': np.median(reaction_times),
                        'count': len(reaction_times)
                    }
                    print(f"  ✓ {user_id}: {len(reaction_times)} reaction times, mean = {np.mean(reaction_times):.3f}s")
                else:
                    print(f"  ✗ {user_id}: No valid reaction times")
                    
            except Exception as e:
                print(f"  ✗ Failed to process {user_id}: {e}")
        
        return self.reaction_times_data
    
    def aggregate_sim_v_data(self):
        """Aggregate SIM_8501-8512 into SIM_v."""
        sim_v_pattern = r"SIM_85(?:0[1-9]|1[0-2])"  # Match SIM_8501 to SIM_8512
        
        sim_v_reaction_times = []
        sim_v_grid_rts = {i: [] for i in range(9)}
        to_remove = []
        
        for user_id, data in self.reaction_times_data.items():
            if re.match(sim_v_pattern, user_id):
                sim_v_reaction_times.extend(data['overall'])
                
                # Aggregate grid data
                for grid_id in range(9):
                    sim_v_grid_rts[grid_id].extend(data['by_grid'][grid_id])
                
                to_remove.append(user_id)
        
        # Remove individual entries and add combined entry
        if sim_v_reaction_times:
            for user_id in to_remove:
                del self.reaction_times_data[user_id]
            
            self.reaction_times_data['SIM_v'] = {
                'overall': sim_v_reaction_times,
                'by_grid': sim_v_grid_rts,
                'mean': np.mean(sim_v_reaction_times),
                'std': np.std(sim_v_reaction_times, ddof=1) if len(sim_v_reaction_times) > 1 else 0,
                'median': np.median(sim_v_reaction_times),
                'count': len(sim_v_reaction_times)
            }
            
            print(f"Combined {len(to_remove)} users into SIM_v with {len(sim_v_reaction_times)} reaction time measurements")
    
    def create_reaction_time_box_plot(self, figsize=(12, 6)):
        """Create box plot comparing reaction times across users."""
        if not self.reaction_times_data:
            print("No reaction time data available!")
            return None
        
        # Models to highlight (following the existing pattern)
        models_to_highlight = [
            'SIM_m', 'SIM_m_mono', 'SIM_ms',  # m group
            'SIM_audio', 'SIM_audio_mono', 'SIM_audio_sparse',  # audio group
            'SIM_v'  # v group
        ]
        
        # Define color groups for highlighting
        highlight_colors = {
            'SIM_v': '#FFCCBC',  # Peach color for SIM_v
            'SIM_audio': '#81D4FA',  # Light blue for audio models
            'SIM_audio_mono': '#81D4FA',
            'SIM_audio_sparse': '#81D4FA',
            'SIM_m': '#F9E79F',  # Yellow for m models
            'SIM_m_mono': '#F9E79F',
            'SIM_ms': '#F9E79F'
        }
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Prepare data for plotting
        all_rts = []
        all_users = []
        
        # Calculate median reaction time for each user for sorting
        user_medians = {user_id: data['median'] for user_id, data in self.reaction_times_data.items()}
        
        # Sort users by their median reaction time
        sorted_users = sorted(user_medians.keys(), key=lambda x: user_medians[x])
        
        # Add background highlighting
        for i, user_id in enumerate(sorted_users):
            if user_id in models_to_highlight:
                ax.axvspan(i-0.5, i+0.5, 
                          color=highlight_colors.get(user_id, '#FFFDE7'),
                          alpha=0.4, zorder=0)
        
        # Add data in sorted order
        for user_id in sorted_users:
            rts = self.reaction_times_data[user_id]['overall']
            all_rts.extend(rts)
            all_users.extend([user_id] * len(rts))
        
        rt_df = pd.DataFrame({'user': all_users, 'reaction_time': all_rts})
        rt_df['user'] = pd.Categorical(rt_df['user'], categories=sorted_users, ordered=True)
        
        # Create box plot
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        sns.boxplot(data=rt_df, x='user', y='reaction_time', ax=ax, 
                   palette=colors[:len(self.reaction_times_data)],
                   showfliers=False)
        
        # Customize plot
        ax.set_title('Reaction Times\n Simulated vs. Real Users', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Reaction Time (s)', fontsize=12)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_folder, 'reaction_times_directional_comparison.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Reaction time plot saved to: {output_path}")
        
        return fig, ax
    
    def create_grid_analysis_plot(self, figsize=(15, 10)):
        """Create detailed grid-wise analysis plot."""
        if not self.reaction_times_data:
            return None
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()
        
        # For each grid position
        for grid_id in range(9):
            ax = axes[grid_id]
            
            grid_data = []
            grid_users = []
            
            for user_id, data in self.reaction_times_data.items():
                if data['by_grid'][grid_id]:  # If this user has data for this grid
                    grid_rts = data['by_grid'][grid_id]
                    grid_data.extend(grid_rts)
                    grid_users.extend([user_id] * len(grid_rts))
            
            if grid_data:
                grid_df = pd.DataFrame({'user': grid_users, 'reaction_time': grid_data})
                
                # Create box plot for this grid
                sns.boxplot(data=grid_df, x='user', y='reaction_time', ax=ax)
                ax.set_title(f'Grid {grid_id}', fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('Reaction Time (s)' if grid_id % 3 == 0 else '')
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Grid {grid_id}\nNo data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle('Reaction Times by Grid Position\nDirectional Movement Detection', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_folder, 'reaction_times_by_grid.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Grid analysis plot saved to: {output_path}")
        
        return fig, axes
    
    def print_summary_statistics(self):
        """Print detailed summary statistics."""
        print("\n" + "="*80)
        print("REACTION TIME ANALYSIS SUMMARY")
        print("="*80)
        
        if not self.reaction_times_data:
            print("No reaction time data available!")
            return
        
        print(f"\nMethod: Directional Movement Detection")
        print(f"Algorithm: reaction_time = time_that_move - spawn_time")
        print(f"Where time_that_move = first time hand moves toward target")
        
        print(f"\nUSER STATISTICS:")
        print("-" * 50)
        
        # Sort by mean reaction time
        sorted_users = sorted(self.reaction_times_data.items(), 
                            key=lambda x: x[1]['mean'])
        
        for user_id, data in sorted_users:
            mean_rt = data['mean']
            std_rt = data['std']
            median_rt = data['median']
            count = data['count']
            
            # Calculate grid coverage
            grids_with_data = sum(1 for grid_rts in data['by_grid'].values() if grid_rts)
            
            print(f"{user_id:15s}: {mean_rt:.3f}s ± {std_rt:.3f}s "
                  f"(median: {median_rt:.3f}s, n={count}, grids: {grids_with_data}/9)")
        
        # Overall statistics
        all_reaction_times = []
        for data in self.reaction_times_data.values():
            all_reaction_times.extend(data['overall'])
        
        if all_reaction_times:
            overall_mean = np.mean(all_reaction_times)
            overall_std = np.std(all_reaction_times, ddof=1)
            overall_median = np.median(all_reaction_times)
            
            print(f"\nOVERALL STATISTICS:")
            print(f"Total measurements: {len(all_reaction_times)}")
            print(f"Overall mean: {overall_mean:.3f}s ± {overall_std:.3f}s")
            print(f"Overall median: {overall_median:.3f}s")
            print(f"Range: {np.min(all_reaction_times):.3f}s - {np.max(all_reaction_times):.3f}s")
        
        print("="*80)
    
    def export_data_to_csv(self):
        """Export reaction time data to CSV files."""
        # Export individual measurements
        all_data = []
        for user_id, data in self.reaction_times_data.items():
            for i, rt in enumerate(data['overall']):
                all_data.append({
                    'user_id': user_id,
                    'trial_number': i + 1,
                    'reaction_time': rt
                })
        
        if all_data:
            detailed_df = pd.DataFrame(all_data)
            detailed_path = os.path.join(self.output_folder, 'reaction_times_detailed.csv')
            detailed_df.to_csv(detailed_path, index=False)
            print(f"Detailed data exported to: {detailed_path}")
        
        # Export summary statistics
        summary_data = []
        for user_id, data in self.reaction_times_data.items():
            grids_with_data = sum(1 for grid_rts in data['by_grid'].values() if grid_rts)
            summary_data.append({
                'user_id': user_id,
                'mean_rt': data['mean'],
                'std_rt': data['std'],
                'median_rt': data['median'],
                'count': data['count'],
                'grids_with_data': grids_with_data
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.output_folder, 'reaction_times_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary statistics exported to: {summary_path}")


def main():
    """Main function to run the multi-user reaction time analysis."""
    print("Multi-User Reaction Time Analysis")
    print("Using Directional Movement Detection Algorithm")
    print("="*50)
    
    # Initialize comparison
    comparison = MultiUserReactionTimeComparison(data_path="hit_data", task_condition="difficulty")
    
    # Load all users
    print("Loading user data...")
    comparison.load_all_users()
    
    if not comparison.users_data:
        print("No user data found! Please check your data path and file structure.")
        return None
    
    # Calculate reaction times for all users
    print("Calculating reaction times...")
    comparison.calculate_all_reaction_times()
    
    # Aggregate SIM_v data
    print("Aggregating SIM_v data...")
    comparison.aggregate_sim_v_data()
    
    # Create visualizations
    print("Creating visualizations...")
    fig1, ax1 = comparison.create_reaction_time_box_plot()
    fig2, axes2 = comparison.create_grid_analysis_plot()
    
    # Print summary statistics
    comparison.print_summary_statistics()
    
    # Export data
    print("Exporting data...")
    comparison.export_data_to_csv()
    
    # Show plots
    plt.show()
    
    print(f"\nAnalysis complete! Results saved in '{comparison.output_folder}' folder:")
    print("- reaction_times_directional_comparison.png")
    print("- reaction_times_by_grid.png") 
    print("- reaction_times_detailed.csv")
    print("- reaction_times_summary.csv")
    
    return comparison


if __name__ == "__main__":
    comparison = main()