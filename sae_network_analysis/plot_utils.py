# For visualization
import matplotlib.pyplot as plt
import seaborn as sns


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

def plot_cluster_heatmap_single(corr_matrix, cluster_dict, filename, title='Feature Correlation (Absolute Values) Heatmap Grouped by Clusters', cbar_range=(0, 1)):
    """
    Plots and saves a heatmap of a correlation matrix, grouping features by clusters.

    This function visualizes a correlation matrix using a heatmap where features are grouped
    by predefined clusters. It includes custom tick labels and positions, separating lines
    between clusters, and a color bar to represent the correlation coefficients.

    Parameters:
    ----------
    corr_matrix : pandas.DataFrame
        A correlation matrix where both rows and columns are features.
    cluster_dict : dict
        A dictionary where keys are cluster names and values are lists of feature indices that belong
        to each cluster. This dictionary is used to group the features on the heatmap.
    filename : str
        The file path where the heatmap will be saved (as an image file).
    title : str, optional
        The title of the heatmap. Defaults to 'Feature Correlation (Absolute Values) Heatmap Grouped by Clusters'.
    cbar_range : tuple, optional
        A tuple representing the range for the color bar. Defaults to (0, 1), which maps the color scale
        from no correlation (0) to perfect correlation (1).

    Returns:
    -------
    None
        The function saves the heatmap to the specified filename and displays the plot.

    Example:
    --------
    >>> import pandas as pd
    >>> corr_matrix = pd.DataFrame({
    >>>    'a': [1.0, 0.2, 0.3],
    >>>    'b': [0.2, 1.0, 0.4],
    >>>    'c': [0.3, 0.4, 1.0]
    >>> }, index=['a', 'b', 'c'])
    >>> clusters = {'cluster_1': [0, 1], 'cluster_2': [2]}
    >>> plot_cluster_heatmap_single(corr_matrix, clusters, 'heatmap.png')

    Notes:
    ------
    - The heatmap groups features by cluster, with lines separating clusters.
    - Custom tick labels are placed at the center of each cluster along both axes.
    - The heatmap image is saved to the specified filename with a resolution of 300 DPI.
    """
    # Create a list of labels for the heatmap
    labels = []
    tick_positions = []
    current_position = 0

    for cluster, indices in cluster_dict.items():
        labels.append(cluster)
        tick_positions.append(current_position + len(indices) / 2)
        current_position += len(indices)

    (vmin, vmax) = cbar_range

    # Create the heatmap with a colorbar title
    plt.figure(figsize=(12, 10))
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

    # Set the tick positions and labels
    plt.xticks(tick_positions, labels, rotation=45, ha='right')
    plt.yticks(tick_positions, labels, rotation=0)

    # Add separating lines between clusters
    for position in range(1, len(corr_matrix)):
        if position in [sum(len(indices) for indices in list(cluster_dict.values())[:i+1]) for i in range(len(cluster_dict))]:
            plt.axhline(y=position, color='black', linewidth=1)
            plt.axvline(x=position, color='black', linewidth=1)

    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
