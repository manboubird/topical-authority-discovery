import pandas as pd
from kedro_datasets.networkx import GMLDataset
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def preprocess_followers(followers_data: pd.DataFrame) -> nx.DiGraph:
    """Preprocesses the data for followers to create a directed graph.
    
    Creates a directed graph where:
    - Leaders (topical authorities) are target nodes
    - Followers are source nodes
    - Edges represent follower relationships (follower -> leader)
    
    Args:
        followers_data: DataFrame containing follower relationships with columns:
            - follower_id: ID of the follower (source node)
            - leader_id: ID of the topical authority (target node)
            - Optional edge attributes can be included in additional columns
    
    Returns:
        nx.DiGraph: Directed graph representing follower relationships
    """
    # Create an empty directed graph
    graph = nx.DiGraph()
    
    # Add edges from the followers data
    # Each row represents a follower->leader relationship
    for _, row in followers_data.iterrows():
        follower_id = str(row['follower_id'])
        leader_id = str(row['leader_id'])
        
        # Add edge attributes if they exist in the dataframe
        edge_attrs = {k: v for k, v in row.items() 
                     if k not in ['follower_id', 'leader_id']}
        
        # Add the edge to the graph
        graph.add_edge(follower_id, leader_id, **edge_attrs)
    
    return graph


def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies["iata_approved"] = _is_true(companies["iata_approved"])
    companies["company_rating"] = _parse_percentage(companies["company_rating"])
    return companies


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    rated_shuttles = rated_shuttles.drop("id", axis=1)
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table


def extract_keyword_from_bio(users: pd.DataFrame) -> pd.DataFrame:
    """Extract keywords from user bios using TF-IDF.

    Args:
        users: DataFrame containing user_id, bio, and topics_of_interest columns

    Returns:
        DataFrame with added extracted_keywords column
    """
    # Create a copy to avoid modifying the original
    users_with_keywords = users.copy()
    
    # Preprocess bios
    def preprocess_text(text):
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    # Preprocess all bios
    processed_bios = users_with_keywords['bio'].apply(preprocess_text)
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10,  # Extract top 10 keywords
        stop_words='english',  # Use built-in English stop words
        ngram_range=(1, 2)  # Consider both unigrams and bigrams
    )
    
    # Fit and transform the bios
    tfidf_matrix = vectorizer.fit_transform(processed_bios)
    
    # Get feature names (keywords)
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract top keywords for each bio
    def get_top_keywords(tfidf_vector, feature_names):
        # Get indices of non-zero elements
        indices = tfidf_vector.indices
        # Get corresponding scores
        scores = tfidf_vector.data
        # Create pairs of (score, keyword)
        keyword_scores = [(scores[i], feature_names[indices[i]]) for i in range(len(indices))]
        # Sort by score in descending order
        keyword_scores.sort(reverse=True)
        # Get top keywords
        top_keywords = [kw for _, kw in keyword_scores[:5]]  # Get top 5 keywords
        return '||'.join(top_keywords)
    
    # Apply keyword extraction to each bio
    users_with_keywords['extracted_keywords'] = [
        get_top_keywords(tfidf_matrix[i], feature_names)
        for i in range(len(users_with_keywords))
    ]
    
    return users_with_keywords


def compute_authority_score(
    graph: nx.DiGraph,
    users_with_keywords: pd.DataFrame
) -> pd.DataFrame:
    """Compute authority scores using the Fast Algorithm for Interest Propagation.
    
    This algorithm computes authority scores in three passes:
    1. Compute explainable authority scores (Fe) based on known interests
    2. Infer broader interests (Si) for all users
    3. Compute broader authority scores (Fi) based on inferred interests
    
    Args:
        graph: Directed graph representing follower relationships
        users_with_keywords: DataFrame containing user profiles with extracted keywords
    
    Returns:
        DataFrame containing authority scores for each user and topic
    """
    import numpy as np
    
    # Get unique topics from all users
    all_topics = set()
    for topics in users_with_keywords['topics_of_interest'].dropna():
        all_topics.update(topics.split('||'))
    topics_list = sorted(list(all_topics))
    
    # Initialize matrices
    n_users = len(graph.nodes())
    n_topics = len(topics_list)
    
    # Create topic to index mapping
    topic_to_idx = {topic: idx for idx, topic in enumerate(topics_list)}
    
    # Initialize Sc (clamped interests) matrix
    Sc = np.zeros((n_topics, n_users))
    user_to_idx = {user: idx for idx, user in enumerate(graph.nodes())}
    
    # Fill Sc matrix with known interests
    for _, row in users_with_keywords.iterrows():
        if pd.notna(row['topics_of_interest']):
            user_idx = user_to_idx[row['user_id']]
            for topic in row['topics_of_interest'].split('||'):
                if topic in topic_to_idx:
                    Sc[topic_to_idx[topic], user_idx] = 1
    
    # PASS 1: Compute explainable authority scores (Fe)
    Fe = np.zeros((n_topics, n_users))
    for v in graph.nodes():
        v_idx = user_to_idx[v]
        followers_with_interests = [u for u in graph.predecessors(v) 
                                 if any(Sc[:, user_to_idx[u]] > 0)]
        
        if followers_with_interests:
            min_v = len(followers_with_interests)
            for u in followers_with_interests:
                u_idx = user_to_idx[u]
                Fe[:, v_idx] += Sc[:, u_idx]
            Fe[:, v_idx] /= min_v
    
    # PASS 2: Compute broader interests (Si)
    Si = np.zeros((n_topics, n_users))
    for u in graph.nodes():
        u_idx = user_to_idx[u]
        following = list(graph.successors(u))
        if following:
            nout_u = len(following)
            for v in following:
                v_idx = user_to_idx[v]
                Si[:, u_idx] += Fe[:, v_idx]
            Si[:, u_idx] /= nout_u
    
    # PASS 3: Compute broader authority scores (Fi)
    Fi = np.zeros((n_topics, n_users))
    for v in graph.nodes():
        v_idx = user_to_idx[v]
        followers = list(graph.predecessors(v))
        if followers:
            nin_v = len(followers)
            for u in followers:
                u_idx = user_to_idx[u]
                Fi[:, v_idx] += Si[:, u_idx]
            Fi[:, v_idx] /= nin_v
    
    # Final authority scores
    F = Fe + Fi
    
    # Convert to DataFrame
    authority_scores = pd.DataFrame(
        F.T,
        index=graph.nodes(),
        columns=topics_list
    )
    
    return authority_scores
