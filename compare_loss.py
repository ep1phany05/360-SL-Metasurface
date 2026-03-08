import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List


def plot_loss_comparison(run_log_paths: List[str]):
    """
    Plots a comparison of training/validation loss curves from multiple experiment runs.

    Args:
        run_log_paths (List[str]): A list of paths to the log directories of different runs.
                                   e.g., ['runs/exp1/log', 'runs/exp2/log']
    """
    # --- Configuration ---
    # Define the four .npy files to be plotted and their corresponding titles.
    loss_filenames = ["train_loss_history.npy", "train_l1_loss_history.npy", "valid_loss_history.npy", "valid_l1_loss_history.npy"]

    plot_titles = ["Training Loss Comparison", "Training L1 Loss Comparison", "Validation Loss Comparison", "Validation L1 Loss Comparison"]

    # --- Create Figure and Subplots ---
    # Create a 2x2 grid of subplots. `figsize` can be adjusted.
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Flatten the 2x2 array of axes into a 1D array for easier iteration.
    axes = axes.flatten()

    # --- Plotting Loop ---
    # Iterate over each subplot axis, the corresponding filename, and title.
    for ax, filename, title in zip(axes, loss_filenames, plot_titles):

        # --- Loop through each provided experiment path ---
        for log_path in run_log_paths:
            # Construct the full path to the specific .npy file.
            full_file_path = os.path.join(log_path, filename)

            # --- File Loading and Validation ---
            # Check if the file exists before trying to load it.
            if not os.path.exists(full_file_path):
                print(f"Warning: File not found, skipping: {full_file_path}")
                continue

            # Load the loss history data from the .npy file.
            loss_history = np.load(full_file_path)

            # Generate the x-axis values (epochs). We start epochs from 1 for plotting.
            epochs = range(1, len(loss_history) + 1)

            # --- Generate a meaningful label for the legend ---
            # We assume the path is like '.../runs/250917-090424/log'.
            # This code will extract the '250917-090424' part.
            # `strip` handles potential trailing slashes.
            parent_dir = os.path.dirname(log_path.strip('/\\'))
            run_name = os.path.basename(parent_dir)

            # Plot the data on the current subplot.
            ax.plot(epochs, loss_history, label=run_name, lw=2)

        # --- Configure the subplot ---
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Only add a legend if lines were actually plotted on the axis.
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    # --- Final Layout Adjustment and Display ---
    # Adjust layout to prevent titles and labels from overlapping.
    plt.suptitle("Training and Validation Loss Comparison", fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust rect to make space for suptitle

    # Show the plot.
    plt.show()


# --- Example Usage ---
if __name__ == '__main__':
    # This block demonstrates how to use the function.
    # It will create dummy directories and loss files for a runnable example.

    print("Creating dummy experiment logs for demonstration...")

    # --- Create dummy data for two hypothetical runs ---
    run_dirs = ['runs/250917-134108', 'runs/250917-090424']
    log_paths_for_demo = []

    for run_dir in run_dirs:
        log_dir = os.path.join(run_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        log_paths_for_demo.append(log_dir)

        # Generate some plausible-looking loss curves (decaying exponential + noise)
        num_epochs = 400
        epochs_arr = np.arange(num_epochs)

        # Base decay and noise levels for this run
        base_decay = 0.95 + np.random.rand() * 0.04  # e.g., 0.95 to 0.99
        noise_level = 0.05 + np.random.rand() * 0.1

        # Create and save the four loss files
        train_loss = 2.0 * (base_decay ** epochs_arr) + np.random.rand(num_epochs) * noise_level
        train_l1 = 1.0 * (base_decay ** epochs_arr) + np.random.rand(num_epochs) * noise_level * 0.5
        valid_loss = 2.2 * (base_decay ** epochs_arr) + np.random.rand(num_epochs) * noise_level
        valid_l1 = 1.1 * (base_decay ** epochs_arr) + np.random.rand(num_epochs) * noise_level * 0.5

        np.save(os.path.join(log_dir, "train_loss_history.npy"), train_loss)
        np.save(os.path.join(log_dir, "train_l1_loss_history.npy"), train_l1)
        np.save(os.path.join(log_dir, "valid_loss_history.npy"), valid_loss)
        np.save(os.path.join(log_dir, "valid_l1_loss_history.npy"), valid_l1)

    print("Dummy logs created.")
    print(f"Log directories to compare: {log_paths_for_demo}")
    print("\nNow plotting the comparison...")

    # --- Call the function with the paths to the log directories ---
    plot_loss_comparison(log_paths_for_demo)
