import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the results
df = pd.read_csv("results/avg_noise_performance_exp_acc.csv")

# Create the figure
plt.figure(figsize=(12, 8))

# Get method names (all columns except noise_power)
methods = [col for col in df.columns if col != 'noise_power']

# Plot each method
for method in methods:
    plt.plot(df['noise_power'], df[method], marker='o', linewidth=2, 
             markersize=6, label=method, alpha=0.8)

# Customize the plot
plt.xlabel('Noise Power (Label Flipping Probability)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.title('Method Performance vs Label Flipping Noise\n(Average across 14 datasets)', 
          fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks to show all noise levels
plt.xticks(df['noise_power'], fontsize=12)
plt.yticks(fontsize=12)

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

# Set y-axis limits to better show the differences
y_min = df[methods].min().min() - 0.02
y_max = df[methods].max().max() + 0.02
plt.ylim(y_min, y_max)

# Tight layout to prevent legend cutoff
plt.tight_layout()

# Save the plot
plt.savefig('results/noise_performance_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('results/noise_performance_plot.pdf', bbox_inches='tight')

# Show the plot
plt.show()

print("Plot saved as:")
print("- results/noise_performance_plot.png")
print("- results/noise_performance_plot.pdf")

# Print some statistics
print("\nPerformance Summary:")
print("="*50)
print(f"{'Method':<20} {'No Noise':<10} {'Max Noise':<10} {'Degradation':<12}")
print("-"*50)

for method in methods:
    no_noise = df[df['noise_power'] == 0.0][method].iloc[0]
    max_noise = df[df['noise_power'] == 0.5][method].iloc[0]
    degradation = no_noise - max_noise
    print(f"{method:<20} {no_noise:<10.3f} {max_noise:<10.3f} {degradation:<12.3f}")

# Find the most robust method (smallest degradation)
degradations = {}
for method in methods:
    no_noise = df[df['noise_power'] == 0.0][method].iloc[0]
    max_noise = df[df['noise_power'] == 0.5][method].iloc[0]
    degradations[method] = no_noise - max_noise

most_robust = min(degradations, key=degradations.get)
least_robust = max(degradations, key=degradations.get)

print(f"\nMost robust method: {most_robust} (degradation: {degradations[most_robust]:.3f})")
print(f"Least robust method: {least_robust} (degradation: {degradations[least_robust]:.3f})")
