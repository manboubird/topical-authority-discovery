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
            service_account_path = credentials["bigquery"]["service_account_path"]
            
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
                
                # Create table in DuckDB using a sanitized table name
                duckdb_table_name = f"{dataset_id}_{table_name}".replace('-', '_')
                con.execute(f"CREATE TABLE IF NOT EXISTS {duckdb_table_name} AS SELECT * FROM pa_table")
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