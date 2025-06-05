# Visualization Tool for Hitting Performance Analysis

This tool provides visual insights into player performance by generating three key types of plots:

- **Hit Rate Heat Map**
- **Hitting Speed Box Plot**
- **Reaction Time Box Plot**

These visualizations help coaches, analysts, and players better understand hitting behavior and reaction dynamics in Whac-A-Mole.

---

## 📊 Features

### 1. Hit Rate Heat Map
- Visualizes spatial distribution of hit success rates.

### 2. Hitting Speed Box Plot
- Displays the distribution of hitting speeds across different players or sessions.

### 3. Reaction Time Box Plot
- Summarizes how fast players respond to stimuli.
- Highlights reaction variability and possible outliers.

---

## 📁 Project Structure

```
visualization-tool/
│
├── hit_data/                      # Input datasets (e.g., states.csv and events.csv)
│   └── SIM_{}
│       └── {}_easy
│           ├── events.csv        # Event logs (e.g., hit results, timings)
│           └── states.csv        # State logs (e.g., positions, timestamps)
│
├── output/                        # Generated plot images
├── comprehensive_analysis.py
├── multi_user_comparison.py
├── reaction_time_analysis_3.py
└── README.md                      # This file
```
To generate the plots, keep the files in order as above.

---

## 🙋‍♀️ Contact
If you have any questions, please contact Jiahao He (leah12577@gmail.com).


