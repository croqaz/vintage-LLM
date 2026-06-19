import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_trainer_metrics(json_path):
    # Load the trainer state
    with open(json_path, 'r') as f:
        data = json.load(f)

    steps = []
    losses = []
    lrs = []

    # Extract the metrics from log_history
    for entry in data.get('log_history', []):
        if 'loss' in entry and 'learning_rate' in entry and 'step' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
            lrs.append(entry['learning_rate'])

    if not steps:
        print(f'No valid loss/learning_rate data found in {json_path}')
        return

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot loss on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Steps')
    # ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(steps, losses, color=color, linewidth=2, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot learning rate on the secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Learning Rate', color=color)
    # 50% transparency (alpha=0.5) for the learning rate, making it faint
    ax2.plot(steps, lrs, color=color, linewidth=2, alpha=0.25, label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and layout adjustments
    plt.title('Training Loss and Learning Rate over Steps')
    fig.tight_layout()

    # Add legends (combining both axes)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    # Save to file or show
    output_path = Path(json_path).parent / 'training_metrics_plot.png'
    plt.savefig(output_path, dpi=300)
    print(f'Plot saved successfully to: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot loss and learning rate from a trainer_state.json file')
    parser.add_argument(
        '--input', type=str, default='training/checkpoints/checkpoint-10150/trainer_state2.json', help='Path to the trainer_state.json file'
    )
    args = parser.parse_args()

    plot_trainer_metrics(args.input)
