import pandas as pd
from kedro_datasets.networkx import GMLDataset
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import spacy
from spacy.kb import InMemoryLookupKB
from spacy.tokens import Span
import numpy as np
import pytextrank
import os
import ibis
import pyarrow as pa
from google.cloud import bigquery
import logging
import time
from pathlib import Path
import google.auth
from google.api_core import exceptions
import duckdb
from ibis.expr.types import Table
from google.oauth2 import service_account
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from topical_authority_discovery.utils import get_bigquery_client, verify_bigquery_permissions, create_bigquery_query_job
from typing import Tuple, Dict, List

def load_bigquery_to_duckdb(
    table_id: str,
    limit: int = 100
) -> ibis.expr.types.Table:
    """
    Load data from BigQuery to DuckDB using Ibis.
    Gets parameters and credentials from Kedro context.
    
    Args:
        table_id: Full BigQuery table ID (e.g., "project.dataset.table")
        limit: Maximum number of rows to return
    
    Returns:
        ibis.expr.types.Table: Ibis table object pointing to the DuckDB table
    """
    try:
        # Get project path and bootstrap Kedro
        project_path = Path.cwd()
        bootstrap_project(project_path)
        
        # Create a session to access parameters and credentials
        with KedroSession.create(project_path=project_path) as session:
            context = session.load_context()
            
            # Get parameters and credentials
            billing_project_id = context.params["bigquery"]["project_id"]
            duckdb_path = context.params["duckdb_path"]
            credentials = context.config_loader.get("credentials")
            service_account_path = credentials["gcp"]["service_account_path"]
            
            # Parse table_id into components
            project_id, dataset_id, table_name = table_id.split('.')
            
            # Initialize BigQuery client
            bq_client = get_bigquery_client(billing_project_id, service_account_path)
            
            # Verify permissions
            verify_bigquery_permissions(bq_client, project_id, dataset_id, table_name)
            
            # Create DuckDB connection
            con = duckdb.connect(duckdb_path)
            
            try:
                # Execute query and measure time
                start_time = time.time()
                query_job = create_bigquery_query_job(
                    bq_client,
                    project_id,
                    dataset_id,
                    table_name,
                    limit
                )
                
                # Convert results to PyArrow table
                pa_table = query_job.to_arrow()
                elapsed_time = time.time() - start_time
                logging.info(f"BigQuery query executed and data loaded in {elapsed_time:.2f} seconds")
                
                # Create a more intuitive table name based on the data source
                if "accuweather" in project_id:
                    duckdb_table_name = "accuweather_weather"
                elif "google_trends" in dataset_id:
                    if "rising" in table_name:
                        duckdb_table_name = "google_trends_rising_terms"
                    else:
                        duckdb_table_name = "google_trends_top_terms"
                else:
                    # Fallback to a sanitized version of the original name
                    duckdb_table_name = f"{dataset_id}_{table_name}".replace('-', '_')
                
                # Drop the table if it exists to ensure clean state
                con.execute(f"DROP TABLE IF EXISTS {duckdb_table_name}")
                
                # Create table in DuckDB
                con.execute(f"CREATE TABLE {duckdb_table_name} AS SELECT * FROM pa_table")
                logging.info(f"Data successfully loaded into DuckDB table: {duckdb_table_name}")
                
                # Return Ibis table object
                duckdb_ibis = ibis.duckdb.connect(duckdb_path)
                return duckdb_ibis.table(duckdb_table_name)
                
            except exceptions.GoogleAPIError as e:
                logging.error(f"Error executing BigQuery query: {str(e)}")
                raise
                
    except Exception as e:
        logging.error(f"Error loading data from BigQuery to DuckDB: {str(e)}")
        raise
    finally:
        if 'con' in locals():
            con.close()

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


def construct_fashion_knowledge_base(fashion_entities: pd.DataFrame, kb_path: str = None) -> InMemoryLookupKB:
    """Construct a knowledge base for fashion terminology.
    
    Args:
        fashion_entities: DataFrame containing fashion entities with columns:
            - entity_id: Unique identifier for the entity (e.g., "Q1")
            - name: Canonical name of the entity (e.g., "Minimalist Style")
            - aliases: Pipe-separated list of aliases (e.g., "minimalist style||minimalist fashion")
            - vector: Entity vector representation (optional)
        kb_path: Optional path to load/save knowledge base file
    
    Returns:
        InMemoryLookupKB: A knowledge base containing fashion entities and their aliases.
    """
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Initialize knowledge base
    kb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    
    # Try to load existing knowledge base if path is provided
    if kb_path and os.path.exists(kb_path):
        try:
            kb.from_disk(kb_path)
            return kb
        except Exception as e:
            print(f"Failed to load existing knowledge base: {e}")
    
    # Add entities and aliases to knowledge base
    for _, row in fashion_entities.iterrows():
        # Add entity
        kb.add_entity(
            entity=row['entity_id'],
            entity_vector=[1.0],  # Default vector if not provided
            freq=1
        )
        
        # Add aliases
        aliases = row['aliases'].split('||')
        for alias in aliases:
            kb.add_alias(
                alias=alias.strip(),
                entities=[row['entity_id']],
                probabilities=[1.0]
            )
    
    # Save the knowledge base if path is provided
    if kb_path:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(kb_path), exist_ok=True)
            kb.to_disk(kb_path)
        except Exception as e:
            print(f"Failed to save knowledge base: {e}")
    
    return kb

def link_fashion_entities(text: str, kb: InMemoryLookupKB) -> list:
    """Link fashion-related phrases in text to entities in the knowledge base.
    
    Args:
        text: Input text to process
        kb: Knowledge base containing fashion entities
    
    Returns:
        list: List of linked fashion entities found in the text
    """
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Process text
    doc = nlp(text)
    
    # Find potential fashion mentions
    fashion_mentions = []
    
    # Split text into potential phrases (2-3 words)
    words = text.lower().split()
    spans_to_check = []
    
    # Collect all spans to check
    for i in range(len(words)):
        # Try 2-word phrases
        if i + 1 < len(words):
            spans_to_check.append(doc[i:i+2])
        
        # Try 3-word phrases
        if i + 2 < len(words):
            spans_to_check.append(doc[i:i+3])
    
    # Get candidates for all spans in batch
    candidates_batch = kb.get_candidates_batch(spans_to_check)
    
    # Process results
    for span, candidates in zip(spans_to_check, candidates_batch):
        if candidates:
            best_candidate = max(candidates, key=lambda x: x.prior_prob)
            fashion_mentions.append({
                "text": span.text,
                "entity_id": best_candidate.entity_,
                "confidence": best_candidate.prior_prob
            })
    
    # Remove duplicates while preserving order
    seen = set()
    unique_mentions = []
    for mention in fashion_mentions:
        if mention["entity_id"] not in seen:
            seen.add(mention["entity_id"])
            unique_mentions.append(mention)
    
    return unique_mentions

def combine_user_datasets(*user_datasets: pd.DataFrame) -> pd.DataFrame:
    """Combine multiple user datasets into a single DataFrame.
    
    Args:
        *user_datasets: Variable number of DataFrames containing user data
        
    Returns:
        pd.DataFrame: Combined user dataset with unique user_ids
    """
    # Combine all datasets
    combined_df = pd.concat(user_datasets, ignore_index=True)
    
    # Remove duplicates based on user_id, keeping the first occurrence
    combined_df = combined_df.drop_duplicates(subset=['user_id'], keep='first')
    
    return combined_df

def extract_keyword_from_bio(users: pd.DataFrame, fashion_entities: pd.DataFrame, kb_path: str = None) -> pd.DataFrame:
    """Extract key phrases from user bios and link them to fashion terminology.
    
    Args:
        users: DataFrame containing user profiles with 'bio' column.
               Can be a single dataset or combined dataset from multiple sources.
        fashion_entities: DataFrame containing fashion entities and their aliases.
        kb_path: Optional path to load/save knowledge base file
    
    Returns:
        DataFrame with additional 'extracted_keywords' and 'fashion_entities' columns
    """
    import spacy
    import pytextrank
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Add PyTextRank to the pipeline
    nlp.add_pipe("textrank", last=True)
    
    # Construct fashion knowledge base
    fashion_kb = construct_fashion_knowledge_base(fashion_entities, kb_path)
    
    def extract_and_link_phrases(text):
        """Extract key phrases and link them to fashion entities."""
        if pd.isna(text):
            return pd.Series(["", ""])
            
        # Extract key phrases using PyTextRank
        doc = nlp(text)
        phrases = []
        for phrase in doc._.phrases:
            phrase_text = phrase.text.lower()
            if len(phrase_text.split()) >= 2 and phrase.rank > 0.1:
                phrases.append(phrase_text)
        
        # Link phrases to fashion entities
        fashion_entities = []
        for phrase in phrases:
            linked_entities = link_fashion_entities(phrase, fashion_kb)
            if linked_entities:
                fashion_entities.extend(linked_entities)
        
        # Format results
        keywords = "||".join(phrases) if phrases else ""
        fashion_entity_ids = "||".join(set(e["entity_id"] for e in fashion_entities)) if fashion_entities else ""
        
        # Print debug information
        print(f"\nProcessing text: {text}")
        print(f"Extracted phrases: {phrases}")
        print(f"Linked entities: {fashion_entities}")
        print(f"Formatted keywords: {keywords}")
        print(f"Formatted entity IDs: {fashion_entity_ids}")
        
        return pd.Series([keywords, fashion_entity_ids])
    
    # Create a copy of the input DataFrame
    users_with_keywords = users.copy()
    
    # Extract keywords and link fashion entities
    users_with_keywords[['extracted_keywords', 'fashion_entities']] = users_with_keywords['bio'].apply(extract_and_link_phrases)
    
    # Add source information if not present
    if 'source_dataset' not in users_with_keywords.columns:
        users_with_keywords['source_dataset'] = 'combined'
    
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

def create_graph_and_initial_sc(
    raw_follower_data: pd.DataFrame,
    raw_user_interests: pd.DataFrame
) -> Tuple[nx.DiGraph, np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Create NetworkX graph and initial topic interest matrix from raw data.
    
    Args:
        raw_follower_data: DataFrame with columns (follower_id, followed_id)
        raw_user_interests: DataFrame with columns (user_id, topic, score)
    
    Returns:
        Tuple containing:
        - nx.DiGraph: Directed graph representing follower relationships
        - np.ndarray: Initial topic interest matrix Sc of shape (|T| x |V|)
        - Dict[str, int]: Topic to index mapping
        - Dict[str, int]: User to index mapping
    """
    # Create directed graph from follower data
    graph = nx.DiGraph()
    
    # Add edges from follower data
    for _, row in raw_follower_data.iterrows():
        follower_id = str(row['follower_id'])
        followed_id = str(row['followed_id'])
        graph.add_edge(follower_id, followed_id)
    
    # Get unique topics and users
    topics = sorted(raw_user_interests['topic'].unique())
    users = sorted(graph.nodes())
    
    # Create mappings
    topic_to_idx = {topic: idx for idx, topic in enumerate(topics)}
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    
    # Initialize Sc matrix: |T| x |V| (topics x users)
    n_topics = len(topics)
    n_users = len(users)
    Sc = np.zeros((n_topics, n_users))
    
    # Fill Sc matrix based on user interests
    for _, row in raw_user_interests.iterrows():
        user_id = str(row['user_id'])
        topic = row['topic']
        score = row['score']
        
        # Only include users that are in the graph
        if user_id in user_to_idx and topic in topic_to_idx:
            user_idx = user_to_idx[user_id]
            topic_idx = topic_to_idx[topic]
            
            # Set to 1 if user has interest in this topic (score > 0)
            if score > 0:
                Sc[topic_idx, user_idx] = 1
    
    logging.info(f"Created graph with {len(graph.nodes())} users and {len(graph.edges())} edges")
    logging.info(f"Created Sc matrix with shape {Sc.shape} ({n_topics} topics x {n_users} users)")
    logging.info(f"Number of non-zero entries in Sc: {np.count_nonzero(Sc)}")
    
    return graph, Sc, topic_to_idx, user_to_idx

def propagate_interests(
    graph: nx.DiGraph,
    Sc: np.ndarray,
    topic_to_idx: Dict[str, int],
    user_to_idx: Dict[str, int],
    alg_params: Dict[str, float]
) -> np.ndarray:
    """
    Implement Algorithm 1: Fast Algorithm for Interest Propagation.
    
    This function implements the three-pass algorithm described in the ALF paper:
    - PASS 1: Compute explainable authority scores (Fe)
    - PASS 2: Re-estimate interests (Si) for all users
    - PASS 3: Compute inferred authority scores (Fi)
    - Final: Combine scores as F = alpha * Fe + beta * Fi
    
    Args:
        graph: NetworkX directed graph representing follower relationships
        Sc: Initial topic interest matrix of shape (|T| x |V|)
        topic_to_idx: Mapping from topic names to matrix row indices
        user_to_idx: Mapping from user IDs to matrix column indices
        alg_params: Dictionary containing algorithm parameters:
            - alpha: Weight parameter for explainable authority (default: 0.1)
            - beta: Weight parameter for inferred authority (default: 0.01)
            - gamma: Weight parameter for combining scores (default: 1.0)
    
    Returns:
        np.ndarray: Final authority matrix F of shape (|T| x |V|)
    """
    # Extract parameters with defaults
    alpha = alg_params.get('alpha', 0.1)
    beta = alg_params.get('beta', 0.01)
    gamma = alg_params.get('gamma', 1.0)
    
    n_topics, n_users = Sc.shape
    
    # Early return for empty matrices
    if n_topics == 0 or n_users == 0:
        logging.info("Empty input matrices, returning zero matrix")
        return np.zeros((n_topics, n_users))
    
    # Initialize matrices
    Fe = np.zeros((n_topics, n_users))  # Explainable authority scores
    Si = np.zeros((n_topics, n_users))  # Re-estimated interests
    Fi = np.zeros((n_topics, n_users))  # Inferred authority scores
    
    # Precompute user indices for efficiency
    user_indices = {user: idx for user, idx in user_to_idx.items()}
    
    # PASS 1: Compute explainable authority scores (Fe)
    # Fe_v = (1 / min_v) * sum(Sc_u for u -> v where I(u) is not empty)
    for v in graph.nodes():
        v_idx = user_indices[v]
        followers = list(graph.predecessors(v))
        
        if not followers:
            continue
            
        # Find followers with non-empty interests efficiently
        followers_with_interests = []
        for u in followers:
            u_idx = user_indices[u]
            if np.any(Sc[:, u_idx] > 0):
                followers_with_interests.append(u_idx)
        
        if followers_with_interests:
            min_v = len(followers_with_interests)
            # Vectorized sum of Sc matrices for followers with interests
            Fe[:, v_idx] = np.sum(Sc[:, followers_with_interests], axis=1) / min_v
    
    # PASS 2: Re-estimate interests (Si)
    # Si_u = (1 / nout_u) * sum(Fe_v for u -> v)
    for u in graph.nodes():
        u_idx = user_indices[u]
        following = list(graph.successors(u))
        
        if following:
            nout_u = len(following)
            following_indices = [user_indices[v] for v in following]
            # Vectorized sum of Fe matrices for users that u follows
            Si[:, u_idx] = np.sum(Fe[:, following_indices], axis=1) / nout_u
    
    # PASS 3: Compute inferred authority scores (Fi)
    # Fi_v = (1 / nin_v) * sum(Si_u for u -> v)
    for v in graph.nodes():
        v_idx = user_indices[v]
        followers = list(graph.predecessors(v))
        
        if followers:
            nin_v = len(followers)
            follower_indices = [user_indices[u] for u in followers]
            # Vectorized sum of Si matrices for followers of v
            Fi[:, v_idx] = np.sum(Si[:, follower_indices], axis=1) / nin_v
    
    # Final authority scores: F = alpha * Fe + beta * Fi
    F = alpha * Fe + beta * Fi
    
    # Log only essential information
    non_zero_fe = np.count_nonzero(Fe)
    non_zero_fi = np.count_nonzero(Fi)
    non_zero_f = np.count_nonzero(F)
    
    logging.info(f"ALF Algorithm 1 completed: {non_zero_fe} Fe, {non_zero_fi} Fi, {non_zero_f} F non-zero elements")
    
    return F

def compute_authority_scores(
    F: np.ndarray,
    graph: nx.DiGraph,
    topic_to_idx: Dict[str, int],
    user_to_idx: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized Z-scores and weighted Z-scores for authority scores.
    
    This function implements sections 4.3.1 and 4.3.2 of the ALF paper:
    - Section 4.3.1: Normalize authority scores using Z-scores
    - Section 4.3.2: Compute weighted Z-scores using follower counts
    
    Args:
        F: Propagated authority matrix of shape (|T| x |V|)
        graph: NetworkX directed graph representing follower relationships
        topic_to_idx: Mapping from topic names to matrix row indices
        user_to_idx: Mapping from user IDs to matrix column indices
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (wZF, ZF) where:
            - wZF: Weighted Z-score matrix of shape (|T| x |V|)
            - ZF: Z-score matrix of shape (|T| x |V|)
    """
    n_topics, n_users = F.shape
    
    # Early return for empty matrices
    if n_topics == 0 or n_users == 0:
        logging.info("Empty input matrix, returning zero matrices")
        return np.zeros((n_topics, n_users)), np.zeros((n_topics, n_users))
    
    # Initialize ZF and wZF matrices
    ZF = np.zeros((n_topics, n_users))
    wZF = np.zeros((n_topics, n_users))
    
    # Precompute follower counts for all users
    follower_counts = {user_id: graph.in_degree(user_id) for user_id in user_to_idx.keys()}
    
    # Section 4.3.1: Compute Z-scores for each topic (row)
    for topic_idx in range(n_topics):
        topic_scores = F[topic_idx, :]
        non_zero_mask = topic_scores > 0
        non_zero_scores = topic_scores[non_zero_mask]
        
        if len(non_zero_scores) > 0:
            log_scores = np.log(non_zero_scores)
            mu = np.mean(log_scores)
            sigma = np.std(log_scores)
            
            if sigma > 0:
                # Vectorized Z-score computation
                ZF[topic_idx, non_zero_mask] = (log_scores - mu) / sigma
                ZF[topic_idx, ~non_zero_mask] = -3.0  # 3 standard deviations below mean
    
    # Section 4.3.2: Compute weighted Z-scores efficiently
    # Vectorized computation of weight factors
    for user_id, user_idx in user_to_idx.items():
        nin_u = follower_counts[user_id]
        if nin_u > 0:
            # Get all topics for this user
            user_f_scores = F[:, user_idx]
            user_zf_scores = ZF[:, user_idx]
            
            # Vectorized weight factor computation
            valid_scores = user_f_scores > 0
            if np.any(valid_scores):
                weight_factors = np.log(nin_u * user_f_scores[valid_scores])
                wZF[valid_scores, user_idx] = user_zf_scores[valid_scores] * weight_factors
    
    # Log only essential summary information
    non_zero_zf = np.count_nonzero(ZF)
    non_zero_wzf = np.count_nonzero(wZF)
    logging.info(f"Authority scores computed: {non_zero_zf} ZF, {non_zero_wzf} wZF non-zero elements")
    
    return wZF, ZF

def assign_topical_authorities(
    wzf: np.ndarray,
    F: np.ndarray,
    ZF: np.ndarray,
    topic_to_idx: Dict[str, int],
    user_to_idx: Dict[str, int],
    alg_params: Dict[str, float]
) -> pd.DataFrame:
    """
    Assign final topical authorities and remove false positives.
    
    This function implements section 4.3.3 of the ALF paper:
    - Assign highest scoring topic to each user
    - FP1 removal: Filter low-scoring users using popularity-based thresholds
    - FP2 removal: Filter using voting mechanism with top-k scores
    
    Args:
        wzf: Weighted Z-score matrix of shape (|T| x |V|)
        F: Original authority matrix of shape (|T| x |V|)
        ZF: Z-score matrix of shape (|T| x |V|)
        topic_to_idx: Mapping from topic names to matrix row indices
        user_to_idx: Mapping from user IDs to matrix column indices
        alg_params: Dictionary containing algorithm parameters:
            - rho_mid: Mid-point parameter for popularity threshold (default: 0.5)
            - tau: Threshold parameter for filtering (default: 0.1)
            - k_top: Number of top scores for voting mechanism (default: 10)
    
    Returns:
        pd.DataFrame: Final authority list with columns (user_id, assigned_topic, authority_score)
    """
    # Extract parameters with defaults
    rho_mid = alg_params.get('rho_mid', 0.5)
    tau = alg_params.get('tau', 0.1)
    k_top = int(alg_params.get('k_top', 10))
    
    n_topics, n_users = wzf.shape
    
    # Early return for empty matrices
    if n_topics == 0 or n_users == 0:
        logging.info("Empty input matrices, returning empty DataFrame")
        return pd.DataFrame(columns=['user_id', 'assigned_topic', 'authority_score'])
    
    # Create reverse mappings
    idx_to_topic = {idx: topic for topic, idx in topic_to_idx.items()}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    
    # Step 1: Assign highest scoring topic to each user (vectorized)
    best_topic_indices = np.argmax(wzf, axis=0)
    best_scores = np.max(wzf, axis=0)
    
    # Create initial assignments for users with positive scores
    assigned_topics = []
    for user_idx in range(n_users):
        if best_scores[user_idx] > 0:
            assigned_topics.append({
                'user_idx': user_idx,
                'topic_idx': best_topic_indices[user_idx],
                'wzf_score': best_scores[user_idx],
                'user_id': idx_to_user[user_idx],
                'topic_name': idx_to_topic[best_topic_indices[user_idx]]
            })
    
    # Step 2: FP1 removal - Filter low-scoring users using popularity-based thresholds
    filtered_assignments = []
    
    for assignment in assigned_topics:
        topic_idx = assignment['topic_idx']
        wzf_score = assignment['wzf_score']
        
        # Calculate popularity-based threshold θt
        topic_scores = wzf[topic_idx, :]
        non_zero_scores = topic_scores[topic_scores > 0]
        
        if len(non_zero_scores) > 0:
            sorted_scores = np.sort(non_zero_scores)
            mid_idx = int(len(sorted_scores) * rho_mid)
            theta_t = sorted_scores[mid_idx] + tau if mid_idx < len(sorted_scores) else sorted_scores[-1] + tau
            
            if wzf_score >= theta_t:
                filtered_assignments.append(assignment)
    
    # Step 3: FP2 removal - Voting mechanism with top-k scores (vectorized)
    final_assignments = []
    
    for assignment in filtered_assignments:
        topic_idx = assignment['topic_idx']
        user_idx = assignment['user_idx']
        
        # Get top-k users for this topic based on F and ZF scores
        topic_f_scores = F[topic_idx, :]
        topic_zf_scores = ZF[topic_idx, :]
        
        top_k_f_indices = np.argsort(topic_f_scores)[-k_top:]
        top_k_zf_indices = np.argsort(topic_zf_scores)[-k_top:]
        
        # Check if user is in top-k for at least one metric
        if user_idx in top_k_f_indices or user_idx in top_k_zf_indices:
            final_assignments.append(assignment)
    
    # Step 4: Create final DataFrame
    if final_assignments:
        final_df = pd.DataFrame([
            {
                'user_id': assignment['user_id'],
                'assigned_topic': assignment['topic_name'],
                'authority_score': assignment['wzf_score']
            }
            for assignment in final_assignments
        ])
        
        # Sort by authority score in descending order
        final_df = final_df.sort_values('authority_score', ascending=False).reset_index(drop=True)
    else:
        final_df = pd.DataFrame(columns=['user_id', 'assigned_topic', 'authority_score'])
    
    # Log only essential summary information
    logging.info(f"Authority assignment: {len(assigned_topics)} → {len(filtered_assignments)} → {len(final_assignments)} final")
    
    return final_df