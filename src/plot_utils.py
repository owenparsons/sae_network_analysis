# For visualization
import matplotlib.pyplot as plt


def plot_heatmap_single(corr_matrix, filename, title='Feature Correlation Heatmap', cbar_range=(0, 1), figsize=(12, 10)):
    """
    Plot a heatmap of the given correlation matrix and save it to a file.

    Parameters:
    ----------
    corr_matrix : pd.DataFrame or np.ndarray
        The correlation matrix to be visualized as a heatmap. It should be a square matrix
        representing the correlation coefficients between features.

    filename : str
        The path and name of the file where the heatmap image will be saved. If set to
        None or an empty string, the heatmap will not be saved to a file.

    title : str, optional
        The title of the heatmap. Default is 'Feature Correlation Heatmap'.

    cbar_range : tuple, optional
        A tuple specifying the range of values for the colorbar. Default is (0, 1).

    figsize : tuple, optional
        A tuple defining the figure size (width, height) in inches. Default is (12, 10).

    Returns:
    -------
    None
        This function does not return any value. It will display the heatmap and save it
        to the specified file if a filename is provided.
    """
    (vmin, vmax) = cbar_range

    # Create the heatmap with a colorbar title
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        center=0,
        vmin=vmin,  # Set minimum value for colorbar
        vmax=vmax,  # Set maximum value for colorbar
        cbar_kws={'label': 'Correlation Coefficient', 'orientation': 'vertical'}  # Add colorbar label
    )

    # Customize the colorbar label orientation
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.set_label_position('left')  # Position the label on the left
    colorbar.ax.set_ylabel(colorbar.ax.get_ylabel(), rotation=270, labelpad=20)  # Rotate the label

    plt.xticks([])
    plt.yticks([])

    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save the heatmap to a file
    plt.show()  # Display the heatmap
