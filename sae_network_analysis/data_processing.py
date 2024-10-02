import json  # For working with JSON files
import pandas as pd  # For data manipulation and analysis
import random  # For random sampling of negative targets
import numpy as np  # For numerical operations
import torch  # For working with PyTorch tensors

def save_dict_to_json(dictionary, filename):
    """
    Save a dictionary to a JSON file.

    :param dictionary: The dictionary to save
    :param filename: The name of the file to save to (including .json extension)
    """
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4)

def load_dict_from_json(filename):
    """
    Load a dictionary from a JSON file.

    :param filename: The name of the file to load from (including .json extension)
    :return: The loaded dictionary
    """
    with open(filename, 'r') as f:
        return json.load(f)

def create_prompt_dataframe(file_path, context_col, target_col, prompt):
    """
    Creates a new DataFrame with input, pos_target, and neg_target columns based on the given context and target columns.

    :param file_path: str, path to the CSV file containing the dataframe
    :param context_col: str, name of the column containing context words
    :param target_col: str, name of the column containing target words
    :param prompt: str, an f-string containing {context_word} to be replaced by the context word from each row
    :return: pd.DataFrame, new dataframe with input, pos_target, and neg_target columns
    """
    # Read the DataFrame from the specified file path
    df = pd.read_csv(file_path)

    # Extract all unique target words to use for negative sampling
    all_target_words = df[target_col].unique().tolist()

    # Function to sample a negative target word
    def get_neg_target(pos_target):
        # Remove the positive target word from the list of potential negative targets
        possible_neg_targets = [word for word in all_target_words if word != pos_target]
        # Randomly select a negative target word
        return random.choice(possible_neg_targets)

    # Create the new DataFrame
    output_df = pd.DataFrame({
        'input': [prompt.format(context_word=row[context_col]) for _, row in df.iterrows()],
        'pos_target': df[target_col],
        'neg_target': [get_neg_target(row[target_col]) for _, row in df.iterrows()]
    })

    return output_df

def process_target_tokens(df, model, set_to_lower=True, add_space=True):
    """
    Processes the pos_target and neg_target columns of the input DataFrame based on the specified conditions.

    :param df: pd.DataFrame, the input DataFrame with 'pos_target' and 'neg_target' columns
    :param model: an object with a tokenizer that has an encode method, e.g., model.tokenizer.encode()
    :param set_to_lower: bool, if True, convert pos_target and neg_target to lowercase
    :param add_space: bool, if True, adds a space before encoding the target words
    :return: pd.DataFrame, the updated DataFrame with processed tokens
    """

    # Apply lowercase transformation if set_to_lower is True
    if set_to_lower:
        df['pos_target'] = df['pos_target'].str.lower()
        df['neg_target'] = df['neg_target'].str.lower()

    # Define the function to encode targets using the model's tokenizer
    def encode_target(target, add_space):
        # Encode with or without a leading space based on the add_space flag
        text = f" {target}" if add_space else f"{target}"
        return model.tokenizer.encode(text)

    # Apply encoding to the pos_target and neg_target columns
    df['pos_token'] = df['pos_target'].apply(lambda x: encode_target(x, add_space))
    df['neg_token'] = df['neg_target'].apply(lambda x: encode_target(x, add_space))

    return df

def convert_sparse_feature_to_long_df(sparse_tensor: torch.Tensor) -> pd.DataFrame:
    """
    Convert a sparse tensor to a long format pandas DataFrame.
    """
    df = pd.DataFrame(sparse_tensor.detach().cpu().numpy())
    df_long = df.melt(ignore_index=False, var_name='column', value_name='value')
    df_long.columns = ["feature", "attribution"]
    df_long_nonzero = df_long[df_long['attribution'] != 0]
    df_long_nonzero = df_long_nonzero.reset_index().rename(columns={'index': 'position'})
    return df_long_nonzero

def convert_feature_attribution_dict_to_long(attribution_dict, sae):
    """
    Convert a dictionary of feature attributions to a long format pandas DataFrame.
    """
    long_dict = {}

    for key, df in attribution_dict.items():
        long_dict[key] = convert_sparse_feature_to_long_df(df.sae_feature_attributions[sae.cfg.hook_name][0])
    return long_dict

def merge_dataframes(df_dict, merge_method='union', agg_method='max'):
    """
    Merges a dictionary of DataFrames into a single DataFrame based on specified criteria.

    Parameters:
        df_dict (dict): A dictionary where keys are strings and values are DataFrames with columns ['position', 'feature', 'attribution'].
        merge_method (str): Specifies how to merge columns - 'union' (default) or 'intersection'.
        agg_method (str): Specifies how to aggregate multiple attributions for the same feature:
            - 'max': Select the value with the largest absolute attribution.
            - 'first': Select the value with the lowest position.
            - 'last': Select the value with the highest position.

    Returns:
        pd.DataFrame: A merged DataFrame with the specified features and aggregated attribution values.
    """

    # Helper function to aggregate values based on the agg_method
    def aggregate_attributions(group, method):
        if method == 'max':
            # Select value with the largest absolute attribution
            return group.loc[group['attribution'].abs().idxmax(), 'attribution']
        elif method == 'first':
            # Select value with the lowest position
            return group.loc[group['position'].idxmin(), 'attribution']
        elif method == 'last':
            # Select value with the highest position
            return group.loc[group['position'].idxmax(), 'attribution']
        else:
            raise ValueError("Invalid aggregation method. Choose 'max', 'first', or 'last'.")

    # Get all features based on merge_method
    if merge_method == 'union':
        all_features = set(feature for df in df_dict.values() for feature in df['feature'])
    elif merge_method == 'intersection':
        all_features = set(df_dict[next(iter(df_dict))]['feature'])
        for df in df_dict.values():
            all_features &= set(df['feature'])
    else:
        raise ValueError("Invalid merge method. Choose 'union' or 'intersection'.")

    # Initialize an empty list to collect the results
    merged_data = []

    # Iterate through the dictionary and process each DataFrame
    for key, df in df_dict.items():
        # Aggregate attributions based on the agg_method
        aggregated = df.groupby('feature').apply(aggregate_attributions, method=agg_method).reset_index()
        aggregated.columns = ['feature', 'attribution']

        # Create a row with the feature attributions
        row = {feature: aggregated.loc[aggregated['feature'] == feature, 'attribution'].values[0] if feature in aggregated['feature'].values else np.nan for feature in all_features}
        merged_data.append(pd.Series(row, name=key))

    # Convert the list of Series into a DataFrame
    merged_df = pd.DataFrame(merged_data)

    # Replace NaN values with 0.0
    merged_df = merged_df.fillna(0.0)

    return merged_df


def filter_for_cluster_size_threshold(clusters, threshold=5):
    """
    Filters clusters based on a minimum size threshold.

    This function takes a dictionary of clusters and returns only the clusters
    that contain more members than the specified threshold.

    Parameters:
    ----------
    clusters : dict
        A dictionary where keys are cluster identifiers and values are lists of cluster members.
    threshold : int, optional
        The minimum number of members a cluster must have to be included in the result (default is 5).

    Returns:
    -------
    dict
        A dictionary containing only the clusters with more than `threshold` members.
lusters, threshold=3)
    {'cluster_2': ['c', 'd', 'e', 'f']}
    """
    main_clusters = {}

    for cluster in clusters.keys():
        if len(clusters[cluster]) > threshold:

            features_list = clusters[cluster]

            main_clusters[cluster] = features_list

    return main_clusters


def filter_for_specific_clusters(corr_matrix, clusters):
    """
    Filters a correlation matrix for a set of specific clusters.

    This function extracts the features from the provided clusters and returns
    a sub-matrix from the input correlation matrix that contains only the
    correlation values between these features.

    Parameters:
    ----------
    corr_matrix : pandas.DataFrame
        A correlation matrix where both rows and columns are feature names.
    clusters : dict
        A dictionary where keys are cluster identifiers and values are lists of features
        (members of each cluster).

    Returns:
    -------
    pandas.DataFrame
    A filtered correlation matrix containing only the rows and columns corresponding
    to the features in the provided clusters.
    """
    all_features = [feature for cluster in clusters.keys() for feature in clusters[cluster]]
    return corr_matrix.loc[all_features, all_features]
