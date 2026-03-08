import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List


def visualize_phase_maps_auto_range(file_paths: List[str]):
    """
    Visualize one, two, or three phase maps from .npy files in a single figure.
    Each phase map's colormap is scaled to its own data range (min/max).

    Args:
        file_paths (List[str]): A list of strings containing the paths to the .npy files.
                                The list should contain 1, 2, or 3 file paths.
    """
    # --- Input Validation ---
    # Check if the number of paths is within the acceptable range (1 to 3).
    num_plots = len(file_paths)
    if num_plots == 0 or num_plots > 3:
        print(f"Error: Expected 1 to 3 file paths, but got {num_plots}.")
        return

    # --- Create Figure and Subplots ---
    # Create a figure with a subplot layout of 1 row and `num_plots` columns.
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))

    # If there's only one plot, `subplots` returns a single Axes object, not an array.
    # To keep the code consistent, we wrap it in a numpy array.
    if num_plots == 1:
        axes = np.array([axes])

    # --- Plotting Loop ---
    # Iterate over each file path and its corresponding subplot axis.
    for i, file_path in enumerate(file_paths):
        ax = axes[i]

        # --- File Loading and Validation ---
        # Check if the file exists before trying to load it.
        if not os.path.exists(file_path):
            print(f"Warning: File not found at '{file_path}'. Skipping.")
            ax.set_title(f"File not found:\n{os.path.basename(file_path)}")
            ax.axis('off')  # Turn off the axis for the empty plot.
            continue

        # Load the phase map data from the .npy file.
        phase_map = np.load(file_path)

        # --- Image Display with Auto-scaling Colormap ---
        # Display the phase map using imshow.
        # By NOT specifying `vmin` and `vmax`, matplotlib automatically scales
        # the colormap to the min and max values of the `phase_map` data.
        im = ax.imshow(phase_map, cmap='hsv')

        # --- Title and Axis Configuration ---
        # Extract the filename from the path to use as a title.
        base_name = os.path.basename(file_path)
        title = os.path.splitext(base_name)[0]
        ax.set_title(title, fontsize=12)
        ax.axis('off')  # Hide the x and y axis ticks and labels.

        # --- Colorbar Configuration ---
        # Add a colorbar to the subplot. It will automatically reflect the data's range.
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Phase (radians)', fontsize=10)

        # We no longer set fixed ticks like [-pi, 0, pi] because the range is now dynamic.
        # Matplotlib will generate appropriate ticks automatically.

    # --- Final Layout Adjustment and Display ---
    # Adjust the layout to prevent titles and labels from overlapping.
    plt.tight_layout()
    # Show the plot.
    plt.show()


# --- Example Usage ---
if __name__ == '__main__':
    # To use this script:
    # 1. Make sure you have numpy and matplotlib installed:
    #    pip install numpy matplotlib
    # 2. Create some dummy .npy files for testing.
    # 3. Replace the placeholder paths in `phase_map_files` with your actual file paths.

    # --- Test Case : Visualize two phase maps ---
    print("\n--- Visualizing 2 phase maps ---")
    file_list = ['runs/250917-134108/phase_best_epoch_818.npy', 'runs/250917-090424/phase_best_epoch_417.npy']
    visualize_phase_maps_auto_range(file_list)
