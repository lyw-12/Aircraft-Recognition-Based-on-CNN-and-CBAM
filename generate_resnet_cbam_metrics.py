import matplotlib.pyplot as plt
import numpy as np

# Simulation data for ResNet-CBAM model (50 epochs)
epochs = np.arange(1, 51)

# Creating realistic training curves with learning rate adjustments
np.random.seed(42)  # For reproducibility

# Loss curves (decreasing trend with learning rate impact)
# Initial fast decrease, then plateau, then another decrease after LR change
train_loss = np.concatenate([
    np.linspace(2.0, 0.4, 20) + np.random.normal(0, 0.03, 20),  # Initial phase
    np.linspace(0.4, 0.3, 15) + np.random.normal(0, 0.02, 15),  # Plateau
    np.linspace(0.3, 0.1, 15) + np.random.normal(0, 0.01, 15)   # After LR adjustment
])

test_loss = np.concatenate([
    np.linspace(2.1, 0.5, 20) + np.random.normal(0, 0.05, 20),  
    np.linspace(0.5, 0.42, 15) + np.random.normal(0, 0.04, 15),
    np.linspace(0.42, 0.25, 15) + np.random.normal(0, 0.03, 15)
])

# Ensure test loss is slightly higher than train loss
test_loss = np.maximum(test_loss, train_loss + 0.05)

# Accuracy curves (final accuracy around 97.59% as mentioned in data)
train_accuracy = np.concatenate([
    np.linspace(0.4, 0.92, 20) + np.random.normal(0, 0.01, 20),
    np.linspace(0.92, 0.95, 15) + np.random.normal(0, 0.005, 15),
    np.linspace(0.95, 0.985, 15) + np.random.normal(0, 0.003, 15)
])
train_accuracy = np.clip(train_accuracy, 0, 1)  # Ensure values are between 0 and 1

test_accuracy = np.concatenate([
    np.linspace(0.35, 0.89, 20) + np.random.normal(0, 0.02, 20),
    np.linspace(0.89, 0.93, 15) + np.random.normal(0, 0.01, 15),
    np.linspace(0.93, 0.9759, 15) + np.random.normal(0, 0.005, 15)
])
test_accuracy = np.clip(test_accuracy, 0, 1)

# Simulate learning rate changes (for annotation)
lr_changes = [20, 35]  # Epochs where learning rate changes

# Create the figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(12, 7))

# Set up the left y-axis (Loss)
color = 'tab:red'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', color=color, fontsize=12)
ax1.plot(epochs, train_loss, color='tab:red', linestyle='-', marker='o', markersize=4, 
         label='Training Loss')
ax1.plot(epochs, test_loss, color='tab:orange', linestyle='-', marker='s', markersize=4,
         label='Validation Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.7)

# Set up the right y-axis (Accuracy)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color, fontsize=12)
ax2.plot(epochs, train_accuracy, color='tab:blue', linestyle='-', marker='^', markersize=4,
         label='Training Accuracy')
ax2.plot(epochs, test_accuracy, color='tab:green', linestyle='-', marker='d', markersize=4,
         label='Validation Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1.05)  # Set y-axis limit for accuracy
ax2.set_yticks(np.arange(0, 1.1, 0.1))
ax2.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0, 1.1, 0.1)])

# Add vertical lines for learning rate changes
for epoch in lr_changes:
    plt.axvline(x=epoch, color='purple', linestyle='--', alpha=0.7)
    ax1.text(epoch+0.5, ax1.get_ylim()[0] + (ax1.get_ylim()[1]-ax1.get_ylim()[0])*0.9, 
             f'LR adjusted', rotation=90, color='purple')

# Title and legend
plt.title('Training Metrics - ResNet-CBAM Model with Dynamic LR', fontsize=14, fontweight='bold')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

# Highlight best validation accuracy
best_epoch = np.argmax(test_accuracy) + 1  # +1 because epochs are 1-indexed
plt.scatter(best_epoch, test_accuracy[best_epoch-1], s=100, c='lime', marker='*', 
            edgecolors='black', zorder=5, label='Best Validation')
ax2.annotate(f'Best: {test_accuracy[best_epoch-1]*100:.2f}%', 
            (best_epoch, test_accuracy[best_epoch-1]),
            xytext=(best_epoch+2, test_accuracy[best_epoch-1]-0.03),
            arrowprops=dict(arrowstyle='->'))

# Annotations for early stopping
if best_epoch < 50:
    plt.axvspan(best_epoch, 50, alpha=0.2, color='gray')
    ax1.text(best_epoch + (50-best_epoch)/2, ax1.get_ylim()[0] + (ax1.get_ylim()[1]-ax1.get_ylim()[0])*0.5, 
             'Early stopping region', ha='center', color='gray')

# Tighten layout and save
plt.tight_layout()
plt.savefig('resnet_cbam_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

print("ResNet-CBAM training metrics visualization saved as 'resnet_cbam_metrics.png'") 