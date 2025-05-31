import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import re
warnings.filterwarnings('ignore')

# Import the existing class - make sure this file is in the same directory
# or adjust the import path as needed
from comprehensive_analysis import TrajectoryData_Sim2VR_SIMULATION

class SimpleUserComparison:
    """Simple class for comparing reaction times and hitting speeds across users."""
    
    def __init__(self, data_path="hit_data", task_condition="difficulty"):
        self.data_path = data_path
        self.task_condition = task_condition
        self.users_data = {}
        
        # Create output folder
        self.output_folder = "multi-user-comparison"
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
    
    def extract_reaction_times(self):
        """Extract reaction times for all users."""
        reaction_times_data = {}
        
        for user_id, trajectory_data in self.users_data.items():
            reaction_times = trajectory_data.analyze_reaction_times()
            if reaction_times:
                reaction_times_data[user_id] = reaction_times
                print(f"  {user_id}: {len(reaction_times)} reaction time measurements")
            else:
                print(f"  {user_id}: No reaction time data")
        
        return reaction_times_data
    
    def extract_hitting_speeds(self):
        """Extract hitting speeds for all users."""
        hitting_speeds_data = {}
        
        for user_id, trajectory_data in self.users_data.items():
            speeds = []
            
            # Get hitting events
            hits = trajectory_data._target_hits
            contacts = trajectory_data._target_contacts
            
            # Extract velocities from hits
            for _, hit in hits.iterrows():
                if hit['velocity'] is not None and len(hit['velocity']) >= 3:
                    # Calculate speed magnitude from velocity vector
                    speed = np.linalg.norm(hit['velocity'])
                    speeds.append(speed)
            
            # Extract velocities from contacts
            for _, contact in contacts.iterrows():
                if contact['velocity'] is not None and len(contact['velocity']) >= 3:
                    speed = np.linalg.norm(contact['velocity'])
                    speeds.append(speed)
            
            if speeds:
                hitting_speeds_data[user_id] = speeds
                print(f"  {user_id}: {len(speeds)} hitting speed measurements")
            else:
                print(f"  {user_id}: No hitting speed data")
        
        return hitting_speeds_data
    
    def create_separate_box_plots(self, figsize=(10, 5)):
        """Create separate box plots for reaction times and hitting speeds."""
        print("\nExtracting data...")
        reaction_times_data = self.extract_reaction_times()
        hitting_speeds_data = self.extract_hitting_speeds()
        
        # Combine SIM_8501-SIM_8512 into SIM_v
        sim_v_pattern = r"SIM_85(?:0[1-9]|1[0-2])"  # Regex pattern to match SIM_8501 to SIM_8512
        
        # For reaction times
        sim_v_reaction_times = []
        to_remove_rt = []
        
        for user_id in reaction_times_data.keys():
            if re.match(sim_v_pattern, user_id):
                sim_v_reaction_times.extend(reaction_times_data[user_id])
                to_remove_rt.append(user_id)
        
        # Remove individual entries and add combined entry if data exists
        if sim_v_reaction_times:
            for user_id in to_remove_rt:
                del reaction_times_data[user_id]
            reaction_times_data['SIM_v'] = sim_v_reaction_times
            print(f"Combined {len(to_remove_rt)} users into SIM_v with {len(sim_v_reaction_times)} reaction time measurements")
        
        # For hitting speeds
        sim_v_hitting_speeds = []
        to_remove_hs = []
        
        for user_id in hitting_speeds_data.keys():
            if re.match(sim_v_pattern, user_id):
                sim_v_hitting_speeds.extend(hitting_speeds_data[user_id])
                to_remove_hs.append(user_id)
        
        # Remove individual entries and add combined entry if data exists
        if sim_v_hitting_speeds:
            for user_id in to_remove_hs:
                del hitting_speeds_data[user_id]
            hitting_speeds_data['SIM_v'] = sim_v_hitting_speeds
            print(f"Combined {len(to_remove_hs)} users into SIM_v with {len(sim_v_hitting_speeds)} hitting speed measurements")
        
        # Models to highlight (with grouping)
        models_to_highlight = [
            'SIM_m', 'SIM_m_mono', 'SIM_ms',  # m group
            'SIM_audio', 'SIM_audio_mono', 'SIM_audio_sparse',  # audio group
            'SIM_v'  # v group
        ]
        
        # Define colors for different users
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Define color groups for highlighting
        highlight_colors = {}
        
        # Group 1: SIM_v group
        v_color = '#FFCCBC'  # Peach color for SIM_v
        highlight_colors['SIM_v'] = v_color
        
        # Group 2: Audio group
        audio_color = '#81D4FA'  # Light blue for all audio models
        highlight_colors['SIM_audio'] = audio_color
        highlight_colors['SIM_audio_mono'] = audio_color
        highlight_colors['SIM_audio_sparse'] = audio_color
        
        # Group 3: M group
        m_color = '#F9E79F'  # Yellow for all m models
        highlight_colors['SIM_m'] = m_color
        highlight_colors['SIM_m_mono'] = m_color
        highlight_colors['SIM_ms'] = m_color
        
        figures = []
        
        # 1. Reaction Time Box Plot
        if reaction_times_data:
            fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
            
            # Prepare data for plotting
            all_rts = []
            all_users = []
            
            # Calculate median reaction time for each user for sorting
            user_medians_rt = {user_id: np.median(rts) for user_id, rts in reaction_times_data.items()}
            
            # Sort users by their median reaction time (increasing order)
            sorted_users_rt = sorted(user_medians_rt.keys(), key=lambda x: user_medians_rt[x])
            
            # First add background highlighting for the entire columns
            for i, user_id in enumerate(sorted_users_rt):
                # Check if this is one of our specified models to highlight
                if user_id in models_to_highlight:
                    # Add a colored background rectangle for the entire column
                    ax1.axvspan(i-0.5, i+0.5, 
                               color=highlight_colors.get(user_id, '#FFFDE7'),
                               alpha=0.4, zorder=0)  # Increased alpha for better visibility
            
            # Add data in the sorted order
            for user_id in sorted_users_rt:
                rts = reaction_times_data[user_id]
                all_rts.extend(rts)
                all_users.extend([user_id] * len(rts))
            
            rt_df = pd.DataFrame({'user': all_users, 'reaction_time': all_rts})
            
            # Set the order for the categorical x-axis
            rt_df['user'] = pd.Categorical(rt_df['user'], categories=sorted_users_rt, ordered=True)
            
            # Create box plot
            box_plot = sns.boxplot(data=rt_df, x='user', y='reaction_time', ax=ax1, 
                                 palette=colors[:len(reaction_times_data)],
                                 showfliers=False)
                
            # Customize the box plot
            ax1.set_title('Reaction Times\nSimulated vs. Real Users', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Reaction Time (s)', fontsize=11)
            ax1.set_xlabel('')
            ax1.tick_params(axis='x', rotation=45, labelsize=10)
            ax1.tick_params(axis='y', labelsize=10)
            ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            # Save reaction time plot
            rt_output_path = os.path.join(self.output_folder, 'reaction_times_comparison.png')
            fig1.savefig(rt_output_path, dpi=300, bbox_inches='tight')
            print(f"Reaction time plot saved to: {rt_output_path}")
            
            figures.append((fig1, ax1, 'reaction_times'))
            print(f"Reaction time box plot created with {len(reaction_times_data)} users")
        
        # 2. Hitting Speed Box Plot (separate figure)
        if hitting_speeds_data:
            fig2, ax2 = plt.subplots(1, 1, figsize=figsize)
            
            # Prepare data for plotting
            all_speeds = []
            all_users = []
            
            # Calculate median hitting speed for each user for sorting
            user_medians_speed = {user_id: np.median(speeds) for user_id, speeds in hitting_speeds_data.items()}
            
            # Sort users by their median hitting speed (increasing order)
            sorted_users_speed = sorted(user_medians_speed.keys(), key=lambda x: user_medians_speed[x])
            
            # First add background highlighting for the entire columns
            for i, user_id in enumerate(sorted_users_speed):
                # Check if this is one of our specified models to highlight
                if user_id in models_to_highlight:
                    # Add a colored background rectangle for the entire column
                    ax2.axvspan(i-0.5, i+0.5, 
                               color=highlight_colors.get(user_id, '#FFFDE7'),
                               alpha=0.3, zorder=0)
            
            # Add data in the sorted order
            for user_id in sorted_users_speed:
                speeds = hitting_speeds_data[user_id]
                all_speeds.extend(speeds)
                all_users.extend([user_id] * len(speeds))
            
            speed_df = pd.DataFrame({'user': all_users, 'hitting_speed': all_speeds})
            
            # Set the order for the categorical x-axis
            speed_df['user'] = pd.Categorical(speed_df['user'], categories=sorted_users_speed, ordered=True)
            
            # Create box plot
            box_plot = sns.boxplot(data=speed_df, x='user', y='hitting_speed', ax=ax2, 
                                 palette=colors[:len(hitting_speeds_data)],
                                 showfliers=False)
                
            # Customize the box plot
            ax2.set_title('Hitting Speeds\nSimulated vs. Real Users', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Hitting Speed (m/s)', fontsize=11)
            ax2.set_xlabel('')
            ax2.tick_params(axis='x', rotation=45, labelsize=10)
            ax2.tick_params(axis='y', labelsize=10)
            ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            # Save hitting speed plot
            speed_output_path = os.path.join(self.output_folder, 'hitting_speeds_comparison.png')
            fig2.savefig(speed_output_path, dpi=300, bbox_inches='tight')
            print(f"Hitting speed plot saved to: {speed_output_path}")
            
            figures.append((fig2, ax2, 'hitting_speeds'))
            print(f"Hitting speed box plot created with {len(hitting_speeds_data)} users")
        
        return figures
    
    def print_summary_stats(self):
        """Print summary statistics for all users."""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        reaction_times_data = self.extract_reaction_times()
        hitting_speeds_data = self.extract_hitting_speeds()
        
        # Reaction time summary
        if reaction_times_data:
            print("\nREACTION TIMES:")
            print("-" * 30)
            for user_id, rts in reaction_times_data.items():
                mean_rt = np.mean(rts)
                std_rt = np.std(rts)
                median_rt = np.median(rts)
                print(f"{user_id}: {mean_rt:.3f}s ± {std_rt:.3f}s (median: {median_rt:.3f}s, n={len(rts)})")
        
        # Hitting speed summary
        if hitting_speeds_data:
            print("\nHITTING SPEEDS:")
            print("-" * 30)
            for user_id, speeds in hitting_speeds_data.items():
                mean_speed = np.mean(speeds)
                std_speed = np.std(speeds)
                median_speed = np.median(speeds)
                print(f"{user_id}: {mean_speed:.3f} m/s ± {std_speed:.3f} m/s (median: {median_speed:.3f} m/s, n={len(speeds)})")
        
        print("="*60)


def main():
    """Main function to create the separate box plots."""
    print("Simple Multi-User Comparison Tool")
    print("="*40)
    
    # Initialize comparison
    comparison = SimpleUserComparison(data_path="hit_data", task_condition="difficulty")
    
    # Load all available users
    print("Loading user data...")
    comparison.load_all_users()
    
    if not comparison.users_data:
        print("No user data found! Please check your data path and file structure.")
        return None
    
    # Create separate box plots
    print("\nCreating separate box plots...")
    figures = comparison.create_separate_box_plots()
    
    # Print summary statistics
    comparison.print_summary_stats()
    
    # Show the plots
    plt.show()
    
    print(f"\nAnalysis complete! Plots saved in the '{comparison.output_folder}' folder.")
    print("Files created:")
    print("- reaction_times_comparison.png")
    print("- hitting_speeds_comparison.png")
    
    return comparison


if __name__ == "__main__":
    comparison = main()