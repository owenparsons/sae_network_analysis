# For data manipulation and analysis
import pandas as pd  # Library for data manipulation and analysis
import numpy as np   # Library for numerical computations

# For network analysis
import networkx as nx  # Library for creating and manipulating complex networks
from networkx.algorithms.community import greedy_modularity_communities  # Algorithm for community detection

# For visualization
import matplotlib.pyplot as plt  # Library for creating static, animated, and interactive visualizations in Python
import seaborn as sns  # Library for statistical data visualization based on Matplotlib
from matplotlib import cm  # For colormaps in visualizations

def cluster_features_with_network_analysis(corr_matrix, threshold=0.5, plot_graph=False, show_edge_weights=False, show_legend=False):
    """
    Clusters features using network analysis by creating a graph from the correlation matrix.

    Parameters:
        corr_matrix (pd.DataFrame): A pandas DataFrame representing the correlation matrix.
        threshold (float): The correlation threshold for creating edges between nodes.
        plot_graph (bool): If True, plots the network graph.
        show_edge_weights (bool): If True, the edges' thickness represents the correlation value.

    Returns:
        dict: A dictionary where keys are cluster numbers and values are lists of feature names in each cluster.
    """
    # Create a graph
    G = nx.Graph()

    # Add nodes (features)
    G.add_nodes_from(corr_matrix.columns)

    # Add edges based on the correlation threshold, and store the weights
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr_value)

    # Detect communities using the Greedy Modularity algorithm
    communities = list(greedy_modularity_communities(G))

    # Create a cluster dictionary
    clusters = {f'Cluster {i+1}': list(community) for i, community in enumerate(communities)}

    # Optional: Plot the graph
    if plot_graph:
        pos = nx.spring_layout(G, seed=42)

        # Assign a color to each community
        cmap = cm.get_cmap('tab10', len(communities))  # Use 'tab10' color map with enough distinct colors
        community_color_map = {}

        for i, community in enumerate(communities):
            for node in community:
                community_color_map[node] = cmap(i)  # Assign color from the color map

        # Get node colors in the correct order
        node_colors = [community_color_map[node] for node in G.nodes]

        # Edge thickness based on the weight (correlation values)
        if show_edge_weights:
            edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
            # Normalize weights for better visualization (scale the line width)
            edge_widths = [abs(weight) * 5 for weight in weights]  # Scale for better visibility
        else:
            edges = G.edges()
            edge_widths = [1] * len(edges)  # Default width

        # Plot the graph with node colors and edge weights
        plt.figure(figsize=(10, 8))
        nx.draw(
            G, pos, with_labels=True, node_size=500, node_color=node_colors,
            edge_color='gray', edgelist=edges, width=edge_widths, cmap=cmap
        )

        # Create legend (cluster colors)
        for i, community in enumerate(communities):
            plt.scatter([], [], color=cmap(i), label=f'Cluster {i+1}')

        if show_legend:
          plt.legend(title="Clusters")
        plt.title("Feature Correlation Network with Clusters and Edge Weights" if show_edge_weights else "Feature Correlation Network with Clusters")
        plt.show()

    return clusters


def compute_centrality_measures(corr_matrix, threshold=0.5):
    """
    Creates a graph from a correlation matrix and computes several centrality measures.

    Parameters:
        corr_matrix (pd.DataFrame): A pandas DataFrame representing the correlation matrix.
        threshold (float): The correlation threshold for creating edges between nodes.

    Returns:
        dict: A dictionary containing centrality measures (degree, betweenness, closeness, eigenvector).
    """
    # Create an undirected graph
    G = nx.Graph()

    # Add nodes (features)
    G.add_nodes_from(corr_matrix.columns)

    # Add edges based on the correlation threshold
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr_value)

    # Compute centrality measures
    centrality_measures = {}

    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    centrality_measures['degree_centrality'] = degree_centrality

    # Betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    centrality_measures['betweenness_centrality'] = betweenness_centrality

    # Closeness centrality
    closeness_centrality = nx.closeness_centrality(G)
    centrality_measures['closeness_centrality'] = closeness_centrality

    # Eigenvector centrality
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        centrality_measures['eigenvector_centrality'] = eigenvector_centrality
    except nx.NetworkXError as e:
        print(f"Eigenvector centrality calculation failed: {e}")

    # Convert centrality measures to DataFrame for easier visualization
    centrality_df = pd.DataFrame(centrality_measures)

    return centrality_df


def filter_for_cluster_size_threshold(clusters, threshold=5):
    """
    Filter clusters based on a specified size threshold.

    Parameters:
    ----------
    clusters : dict
        A dictionary where the keys are cluster names and the values are lists of features
        belonging to those clusters.

    threshold : int, optional
        The minimum number of features required for a cluster to be considered a main cluster.
        Default is 5.

    Returns:
    -------
    dict
        A dictionary containing only the clusters that have more features than the specified
        threshold. The keys are the cluster names and the values are the lists of features.
    """
    main_clusters = {}

    for cluster in clusters.keys():
        if len(clusters[cluster]) > threshold:
            main_clusters[cluster] = clusters[cluster]

    return main_clusters


def filter_for_specific_clusters(corr_matrix, clusters):
    """
    Filter a correlation matrix to retain only the specified clusters.

    Parameters:
    ----------
    corr_matrix : pd.DataFrame
        A DataFrame representing the correlation matrix, with features as both rows and columns.

    clusters : dict
        A dictionary where the keys are cluster names and the values are lists of features
        belonging to those clusters.

    Returns:
    -------
    pd.DataFrame
        A filtered correlation matrix containing only the features specified in the clusters.
        The returned DataFrame includes rows and columns for the features that belong to the
        specified clusters.
    """
    all_features = [feature for cluster in clusters.keys() for feature in clusters[cluster]]
    return corr_matrix.loc[all_features, all_features]
