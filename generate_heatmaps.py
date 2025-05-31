import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from comprehensive_analysis import TrajectoryData_Sim2VR_SIMULATION

def generate_multiple_heatmaps():
    """
    Generate hit rate heatmaps for specific models and store them in a folder.
    
    This function generates heatmaps for:
    1. SIM_m, SIM_m_mono, SIM_ms (m variants)
    2. SIM_audio, SIM_audio_mono, SIM_audio_sparse (audio variants)
    3. SIM_v (combined data from SIM_8501-SIM_8512)
    """
    # Create output directory
    output_dir = "heatmaps"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Models to process
    models = [
        'SIM_m',
        'SIM_m_mono',
        'SIM_ms',
        'SIM_audio',
        'SIM_audio_mono',
        'SIM_audio_sparse',
    ]
    
    # Load and combine data for SIM_v (SIM_8501-8512)
    print("\nCreating SIM_v combined model data...")
    
    # Pattern to match SIM_8501 through SIM_8512
    sim_v_pattern = r"SIM_85(?:0[1-9]|1[0-2])"
    
    # Find all matching user directories
    sim_v_users = []
    for item in os.listdir("hit_data"):
        if re.match(sim_v_pattern, item) and os.path.isdir(os.path.join("hit_data", item)):
            sim_v_users.append(item)
    
    if sim_v_users:
        print(f"Found {len(sim_v_users)} users for SIM_v: {', '.join(sim_v_users)}")
        
        # Load all target data from SIM_v users
        all_target_events = []
        for user_id in sim_v_users:
            try:
                data = TrajectoryData_Sim2VR_SIMULATION(user_id, "difficulty")
                
                # Extract target events with positions
                target_events_with_positions = []
                for event_type in ['target_hit', 'target_contact', 'target_miss']:
                    event_attr = f'_{event_type}' + ('s' if event_type != 'target_miss' else 'es')
                    events_df = getattr(data, event_attr)
                    
                    for _, event in events_df.iterrows():
                        grid_id = event['grid_ID']
                        run_id = event['RUN_ID']
                        
                        target_pos = data._target_info[(data._target_info.index == grid_id) & 
                                                    (data._target_info['RUN_ID'] == run_id)]
                        
                        if not target_pos.empty:
                            target_events_with_positions.append({
                                'event_type': event_type,
                                'grid_id': grid_id,
                                'target_x': target_pos.iloc[0]['global_x'],
                                'target_y': target_pos.iloc[0]['global_y'],
                                'local_x': target_pos.iloc[0]['local_x'],
                                'local_y': target_pos.iloc[0]['local_y'],
                                'j': target_pos.iloc[0]['j'],
                                'i': target_pos.iloc[0]['i'],
                                'user_id': user_id
                            })
                
                all_target_events.extend(target_events_with_positions)
                print(f"  - Added {len(target_events_with_positions)} events from {user_id}")
                
            except Exception as e:
                print(f"  - Error processing {user_id}: {str(e)}")
        
        # Create combined dataframe
        sim_v_events_df = pd.DataFrame(all_target_events)
        
        print(f"Combined {len(sim_v_events_df)} events for SIM_v")
        
        # Add SIM_v to the list of models
        models.append('SIM_v')
    else:
        print("No SIM_85xx models found")
    
    # Generate heatmaps for all models
    for user_id in models:
        print(f"\nGenerating heatmap for {user_id}...")
        
        # Special handling for SIM_v
        if user_id == 'SIM_v' and 'sim_v_events_df' in locals():
            # Process the combined events dataframe for SIM_v
            if sim_v_events_df.empty:
                print("  - No data available for SIM_v")
                continue
            
            position_stats = sim_v_events_df.groupby(['local_x', 'local_y', 'j', 'i'])['event_type'].value_counts().unstack(fill_value=0)
            position_stats['total'] = position_stats.sum(axis=1)
            position_stats['hit_rate'] = position_stats.get('target_hit', 0) / position_stats['total']
            
            # Create the pivot table for the heatmap
            df_reset = position_stats.reset_index()
            pivot_data = df_reset.pivot_table(
                values='hit_rate', 
                index='local_y',
                columns='local_x',
                aggfunc='mean'
            )
            
            pivot_data = pivot_data.sort_index(ascending=False)
            pivot_data = pivot_data.sort_index(axis=1)
            
        else:
            # Normal processing for individual models
            try:
                # Load data for this model
                trajectory_data = TrajectoryData_Sim2VR_SIMULATION(user_id, "difficulty")
                trajectory_data.preprocess()
                
                # Compute hit rates by position
                position_stats = trajectory_data.compute_hit_rates_by_position()
                
                if position_stats is None:
                    print(f"  - Could not compute position-based hit rates for {user_id}")
                    continue
                
                # Create the pivot table for the heatmap
                df_reset = position_stats.reset_index()
                pivot_data = df_reset.pivot_table(
                    values='hit_rate', 
                    index='local_y',
                    columns='local_x',
                    aggfunc='mean'
                )
                
                pivot_data = pivot_data.sort_index(ascending=False)
                pivot_data = pivot_data.sort_index(axis=1)
                
            except Exception as e:
                print(f"  - Error processing {user_id}: {str(e)}")
                continue
        
        # Create and save the heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        import seaborn as sns
        sns.heatmap(pivot_data, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='RdYlBu_r',
                    vmin=0, 
                    vmax=1,
                    cbar_kws={'label': 'Hit Rate'},
                    ax=ax,
                    annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        ax.set_xlabel('Target pos. (x)')
        ax.set_ylabel('Target pos. (y)')
        
        title = f'{user_id}\nTarget Hit Rates'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{user_id}_hit_rate_heatmap.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory
        
        print(f"  - Saved heatmap to {output_path}")
    
    print("\nHeatmap generation complete!")
    print(f"All heatmaps saved in the '{output_dir}' folder.")

def calculate_sim_v_hit_rate():
    """Calculate and print detailed hit rate statistics for SIM_v model."""
    print("\nCalculating SIM_v Hit Rate Statistics...")
    
    # Pattern to match SIM_8501 through SIM_8512
    sim_v_pattern = r"SIM_85(?:0[1-9]|1[0-2])"
    
    # Find all matching user directories
    sim_v_users = []
    for item in os.listdir("hit_data"):
        if re.match(sim_v_pattern, item) and os.path.isdir(os.path.join("hit_data", item)):
            sim_v_users.append(item)
    
    if not sim_v_users:
        print("No SIM_85xx models found")
        return
    
    print(f"Found {len(sim_v_users)} users for SIM_v: {', '.join(sim_v_users)}")
    
    # Track totals across all users
    total_hits = 0
    total_contacts = 0
    total_misses = 0
    total_targets = 0
    
    # Get individual hit rates per user
    user_hit_rates = {}
    
    for user_id in sim_v_users:
        try:
            # Load data
            data = TrajectoryData_Sim2VR_SIMULATION(user_id, "difficulty")
            data.preprocess()
            
            # Get target counts
            if hasattr(data, 'target_stats'):
                stats = data.target_stats
            else:
                data._compute_target_stats()
                stats = data.target_stats
            
            # Extract counts
            hits = stats['target_hit'].sum() if 'target_hit' in stats.columns else 0
            contacts = stats['target_contact'].sum() if 'target_contact' in stats.columns else 0
            misses = stats['target_miss'].sum() if 'target_miss' in stats.columns else 0
            
            user_targets = hits + contacts + misses
            user_hit_rate = hits / user_targets if user_targets > 0 else 0
            
            # Add to totals
            total_hits += hits
            total_contacts += contacts
            total_misses += misses
            total_targets += user_targets
            
            # Store hit rate for this user
            user_hit_rates[user_id] = user_hit_rate
            
            print(f"  - {user_id}: Hit rate {user_hit_rate:.3f} ({hits}/{user_targets} targets)")
            
        except Exception as e:
            print(f"  - Error processing {user_id}: {str(e)}")
    
    # Calculate overall statistics
    if total_targets > 0:
        overall_hit_rate = total_hits / total_targets
        overall_success_rate = (total_hits + total_contacts) / total_targets
        
        print("\nSIM_v OVERALL HIT RATE STATISTICS")
        print("=" * 40)
        print(f"Total targets: {total_targets}")
        print(f"Hits: {total_hits} ({total_hits/total_targets*100:.1f}%)")
        print(f"Contacts: {total_contacts} ({total_contacts/total_targets*100:.1f}%)")
        print(f"Misses: {total_misses} ({total_misses/total_targets*100:.1f}%)")
        print(f"Overall hit rate: {overall_hit_rate:.3f}")
        print(f"Success rate (hits + contacts): {overall_success_rate:.3f}")
        
        # Calculate hit rate statistics across users
        if user_hit_rates:
            hit_rates = list(user_hit_rates.values())
            print("\nHit Rate Variation Across Users")
            print(f"Mean hit rate: {np.mean(hit_rates):.3f} ± {np.std(hit_rates):.3f}")
            print(f"Median hit rate: {np.median(hit_rates):.3f}")
            print(f"Range: {np.min(hit_rates):.3f} - {np.max(hit_rates):.3f}")
    else:
        print("No target data available for SIM_v")

if __name__ == "__main__":
    generate_multiple_heatmaps()
    calculate_sim_v_hit_rate()