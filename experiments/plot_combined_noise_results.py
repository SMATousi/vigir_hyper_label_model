import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load both result files
traditional_df = pd.read_csv("results/avg_noise_performance_exp_acc.csv")
transformer_df = pd.read_csv("results/avg_noise_performance_exp_transformer_acc.csv")

# Create the figure with larger size for better visibility
plt.figure(figsize=(14, 10))

# Get method names from traditional methods (all columns except noise_power)
traditional_methods = [col for col in traditional_df.columns if col != 'noise_power']

# Plot traditional methods with thinner lines and smaller markers
for method in traditional_methods:
    plt.plot(traditional_df['noise_power'], traditional_df[method], 
             marker='o', linewidth=1.5, markersize=4, label=method, alpha=0.7)

# Plot transformer method with thicker line and larger markers to make it stand out
transformer_method = 'LELA-Transformer-Bag'
plt.plot(transformer_df['noise_power'], transformer_df[transformer_method], 
         marker='s', linewidth=3, markersize=8, label=transformer_method, 
         color='red', alpha=0.9, linestyle='-')

# Customize the plot
plt.xlabel('Noise Power (Label Flipping Probability)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.title('Method Performance vs Label Flipping Noise\n(Traditional Methods vs Transformer - Average across 14 datasets)', 
          fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks to show all noise levels
plt.xticks(traditional_df['noise_power'], fontsize=12)
plt.yticks(fontsize=12)

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Create a custom legend with two groups
traditional_handles = []
transformer_handles = []

for line in plt.gca().get_lines():
    if line.get_label() == transformer_method:
        transformer_handles.append(line)
    else:
        traditional_handles.append(line)

# Create two separate legends in different positions
legend1 = plt.legend(traditional_handles, [h.get_label() for h in traditional_handles], 
                    title="Traditional Methods", loc='upper right', 
                    bbox_to_anchor=(1.02, 0.5), fontsize=10)
legend2 = plt.legend(transformer_handles, [h.get_label() for h in transformer_handles], 
                    title="Transformer Method", loc='upper left', 
                    bbox_to_anchor=(0.02, 0.98), fontsize=11, 
                    fancybox=True, shadow=True, framealpha=0.9)

# Add the first legend back (matplotlib removes it when creating the second)
plt.gca().add_artist(legend1)
# plt.gca().add_artist(legend2)

# Set y-axis limits to better show the differences
all_methods_data = []
for method in traditional_methods:
    all_methods_data.extend(traditional_df[method].values)
all_methods_data.extend(transformer_df[transformer_method].values)

y_min = min(all_methods_data) - 0.02
y_max = max(all_methods_data) + 0.02
plt.ylim(y_min, y_max)

# Tight layout to prevent legend cutoff
plt.tight_layout()

# Save the plot
plt.savefig('results/combined_noise_performance_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('results/combined_noise_performance_plot.pdf', bbox_inches='tight')

# Show the plot
plt.show()

print("Combined plot saved as:")
print("- results/combined_noise_performance_plot.png")
print("- results/combined_noise_performance_plot.pdf")

# Print comprehensive statistics
print("\nPerformance Summary - All Methods:")
print("="*70)
print(f"{'Method':<25} {'No Noise':<10} {'Max Noise':<10} {'Degradation':<12} {'Type':<12}")
print("-"*70)

# Traditional methods statistics
for method in traditional_methods:
    no_noise = traditional_df[traditional_df['noise_power'] == 0.0][method].iloc[0]
    max_noise = traditional_df[traditional_df['noise_power'] == 0.5][method].iloc[0]
    degradation = no_noise - max_noise
    print(f"{method:<25} {no_noise:<10.3f} {max_noise:<10.3f} {degradation:<12.3f} {'Traditional':<12}")

# Transformer method statistics
no_noise_transformer = transformer_df[transformer_df['noise_power'] == 0.0][transformer_method].iloc[0]
max_noise_transformer = transformer_df[transformer_df['noise_power'] == 0.5][transformer_method].iloc[0]
degradation_transformer = no_noise_transformer - max_noise_transformer
print(f"{transformer_method:<25} {no_noise_transformer:<10.3f} {max_noise_transformer:<10.3f} {degradation_transformer:<12.3f} {'Transformer':<12}")

# Find the best performing methods
print(f"\nBest Performance Analysis:")
print("="*50)

# Best at no noise
all_no_noise = {}
for method in traditional_methods:
    all_no_noise[method] = traditional_df[traditional_df['noise_power'] == 0.0][method].iloc[0]
all_no_noise[transformer_method] = no_noise_transformer

best_no_noise = max(all_no_noise, key=all_no_noise.get)
print(f"Best at no noise: {best_no_noise} ({all_no_noise[best_no_noise]:.3f})")

# Best at maximum noise
all_max_noise = {}
for method in traditional_methods:
    all_max_noise[method] = traditional_df[traditional_df['noise_power'] == 0.5][method].iloc[0]
all_max_noise[transformer_method] = max_noise_transformer

best_max_noise = max(all_max_noise, key=all_max_noise.get)
print(f"Best at max noise: {best_max_noise} ({all_max_noise[best_max_noise]:.3f})")

# Most robust (smallest degradation)
all_degradations = {}
for method in traditional_methods:
    no_noise = traditional_df[traditional_df['noise_power'] == 0.0][method].iloc[0]
    max_noise = traditional_df[traditional_df['noise_power'] == 0.5][method].iloc[0]
    all_degradations[method] = no_noise - max_noise
all_degradations[transformer_method] = degradation_transformer

most_robust = min(all_degradations, key=all_degradations.get)
least_robust = max(all_degradations, key=all_degradations.get)

print(f"Most robust: {most_robust} (degradation: {all_degradations[most_robust]:.3f})")
print(f"Least robust: {least_robust} (degradation: {all_degradations[least_robust]:.3f})")

# Transformer ranking
transformer_rank_no_noise = sorted(all_no_noise.items(), key=lambda x: x[1], reverse=True)
transformer_position_no_noise = [i for i, (method, _) in enumerate(transformer_rank_no_noise, 1) if method == transformer_method][0]

transformer_rank_max_noise = sorted(all_max_noise.items(), key=lambda x: x[1], reverse=True)
transformer_position_max_noise = [i for i, (method, _) in enumerate(transformer_rank_max_noise, 1) if method == transformer_method][0]

transformer_rank_robustness = sorted(all_degradations.items(), key=lambda x: x[1])
transformer_position_robustness = [i for i, (method, _) in enumerate(transformer_rank_robustness, 1) if method == transformer_method][0]

print(f"\nTransformer Rankings:")
print(f"- No noise performance: {transformer_position_no_noise}/{len(all_no_noise)} methods")
print(f"- Max noise performance: {transformer_position_max_noise}/{len(all_max_noise)} methods") 
print(f"- Robustness (lower degradation is better): {transformer_position_robustness}/{len(all_degradations)} methods")
