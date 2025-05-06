import pandas as pd
import networkx as nx
import pytest
from topical_authority_discovery.pipelines.data_processing.nodes import (
    compute_authority_score,
    preprocess_followers,
    extract_keyword_from_bio,
)


@pytest.fixture
def sample_followers_data():
    """Create sample followers data for testing."""
    return pd.DataFrame({
        'follower_id': ['user1', 'user2', 'user3', 'user4', 'user1'],
        'leader_id': ['authority1', 'authority1', 'authority2', 'authority2', 'authority2']
    })


@pytest.fixture
def sample_users_data():
    """Create sample users data with bios and topics."""
    return pd.DataFrame({
        'user_id': ['user1', 'user2', 'user3', 'user4', 'authority1', 'authority2'],
        'bio': [
            'AI enthusiast and machine learning practitioner',
            'Data scientist working on big data projects',
            'Software engineer interested in AI and ML',
            'Research scientist in data analytics',
            'AI researcher with 10+ years experience in deep learning',
            'Data Science expert specializing in big data analytics'
        ],
        'topics_of_interest': [
            'AI||MachineLearning',
            'DataScience||BigData',
            'AI||MachineLearning',
            'DataScience||Statistics',
            'AI||MachineLearning||DeepLearning',
            'DataScience||Statistics||BigData'
        ]
    })


@pytest.fixture
def real_world_sns_data():
    """Create a small real-world SNS community dataset for testing."""
    return {
        'followers': pd.DataFrame({
            'follower_id': [
                'data_scientist1', 'ml_engineer1', 'ai_researcher1',
                'data_scientist2', 'ml_engineer2', 'ai_researcher2',
                'data_scientist3', 'ml_engineer3', 'ai_researcher3',
                'data_scientist1', 'ml_engineer1', 'ai_researcher1'
            ],
            'leader_id': [
                'ai_expert', 'ai_expert', 'ai_expert',
                'data_science_expert', 'data_science_expert', 'data_science_expert',
                'ml_expert', 'ml_expert', 'ml_expert',
                'data_science_expert', 'ml_expert', 'ai_expert'
            ]
        }),
        'users': pd.DataFrame({
            'user_id': [
                'data_scientist1', 'ml_engineer1', 'ai_researcher1',
                'data_scientist2', 'ml_engineer2', 'ai_researcher2',
                'data_scientist3', 'ml_engineer3', 'ai_researcher3',
                'ai_expert', 'data_science_expert', 'ml_expert'
            ],
            'bio': [
                'Data scientist working on predictive modeling and analytics',
                'Machine Learning Engineer specializing in deep learning systems',
                'AI Researcher focusing on natural language processing',
                'Senior Data Scientist with expertise in big data analytics',
                'ML Engineer building production-grade AI systems',
                'AI Researcher working on computer vision applications',
                'Data Scientist specializing in statistical analysis',
                'ML Engineer with focus on model deployment',
                'AI Researcher in reinforcement learning',
                'Leading AI expert with 15+ years of research experience',
                'Data Science expert with PhD in Statistics',
                'Machine Learning expert specializing in deep learning'
            ],
            'topics_of_interest': [
                'DataScience||Statistics||BigData',
                'MachineLearning||DeepLearning||AI',
                'AI||NLP||MachineLearning',
                'DataScience||BigData||Analytics',
                'MachineLearning||AI||MLOps',
                'AI||ComputerVision||DeepLearning',
                'DataScience||Statistics||Analytics',
                'MachineLearning||MLOps||DeepLearning',
                'AI||ReinforcementLearning||MachineLearning',
                'AI||DeepLearning||MachineLearning||NLP||ComputerVision',
                'DataScience||Statistics||BigData||Analytics',
                'MachineLearning||DeepLearning||AI||MLOps'
            ]
        })
    }


def test_preprocess_followers(sample_followers_data):
    """Test the follower graph creation."""
    graph = preprocess_followers(sample_followers_data)
    
    # Check if graph is directed
    assert isinstance(graph, nx.DiGraph)
    
    # Check if all edges are present
    expected_edges = [
        ('user1', 'authority1'),
        ('user2', 'authority1'),
        ('user3', 'authority2'),
        ('user4', 'authority2'),
        ('user1', 'authority2')
    ]
    assert set(graph.edges()) == set(expected_edges)
    
    # Check if all nodes are present
    expected_nodes = {'user1', 'user2', 'user3', 'user4', 'authority1', 'authority2'}
    assert set(graph.nodes()) == expected_nodes


def test_preprocess_followers_with_duplicate_edges(sample_followers_data):
    """Test graph creation with duplicate edges."""
    # Add duplicate edge
    duplicate_data = sample_followers_data.copy()
    duplicate_data = pd.concat([
        duplicate_data,
        pd.DataFrame({
            'follower_id': ['user1'],
            'leader_id': ['authority1']
        })
    ])
    
    graph = preprocess_followers(duplicate_data)
    
    # Check if duplicate edges are handled correctly
    assert len(graph.edges()) == len(set(duplicate_data.apply(
        lambda x: (x['follower_id'], x['leader_id']), axis=1
    )))


def test_extract_keyword_from_bio(sample_users_data):
    """Test the keyword extraction from bios."""
    users_with_keywords = extract_keyword_from_bio(sample_users_data)
    
    # Check if new column is added
    assert 'extracted_keywords' in users_with_keywords.columns
    
    # Check if keywords are extracted (non-empty)
    assert all(users_with_keywords['extracted_keywords'].str.len() > 0)
    
    # Check if separator is correct
    assert all('||' in keywords for keywords in users_with_keywords['extracted_keywords'] if len(keywords) > 0)


def test_extract_keyword_from_bio_with_special_characters():
    """Test keyword extraction with special characters in bios."""
    special_chars_data = pd.DataFrame({
        'user_id': ['user1'],
        'bio': ['AI & ML expert (10+ years) - working on #DeepLearning, @NLP projects'],
        'topics_of_interest': ['AI||MachineLearning']
    })
    
    users_with_keywords = extract_keyword_from_bio(special_chars_data)
    
    # Check if special characters are handled
    assert len(users_with_keywords['extracted_keywords'].iloc[0]) > 0
    assert '||' in users_with_keywords['extracted_keywords'].iloc[0]


def test_compute_authority_score(sample_followers_data, sample_users_data):
    """Test the authority score computation."""
    # Create graph and process users
    graph = preprocess_followers(sample_followers_data)
    users_with_keywords = extract_keyword_from_bio(sample_users_data)
    
    # Compute authority scores
    authority_scores = compute_authority_score(graph, users_with_keywords)
    
    # Check if output is a DataFrame
    assert isinstance(authority_scores, pd.DataFrame)
    
    # Check if all users are present
    assert set(authority_scores.index) == set(graph.nodes())
    
    # Check if all topics are present
    expected_topics = {'AI', 'MachineLearning', 'DataScience', 'BigData', 'Statistics', 'DeepLearning'}
    assert set(authority_scores.columns) == expected_topics
    
    # Check if scores are between 0 and 1
    assert authority_scores.min().min() >= 0
    assert authority_scores.max().max() <= 1
    
    # Check specific authority relationships
    # authority1 should have high AI and MachineLearning scores
    assert authority_scores.loc['authority1', 'AI'] > 0.5
    assert authority_scores.loc['authority1', 'MachineLearning'] > 0.5
    
    # authority2 should have high DataScience and BigData scores
    assert authority_scores.loc['authority2', 'DataScience'] > 0.5
    assert authority_scores.loc['authority2', 'BigData'] > 0.5


def test_compute_authority_score_edge_cases():
    """Test authority score computation with edge cases."""
    # Test with empty graph
    empty_graph = nx.DiGraph()
    empty_users = pd.DataFrame({
        'user_id': [],
        'bio': [],
        'topics_of_interest': []
    })
    empty_scores = compute_authority_score(empty_graph, empty_users)
    assert empty_scores.empty
    
    # Test with single user
    single_user_graph = nx.DiGraph()
    single_user_graph.add_node('user1')
    single_user_data = pd.DataFrame({
        'user_id': ['user1'],
        'bio': ['AI enthusiast'],
        'topics_of_interest': ['AI||MachineLearning']
    })
    single_user_scores = compute_authority_score(single_user_graph, single_user_data)
    assert len(single_user_scores) == 1
    assert 'user1' in single_user_scores.index


def test_compute_authority_score_with_real_world_data(real_world_sns_data):
    """Test authority score computation with real-world SNS community data."""
    # Create graph and process users
    graph = preprocess_followers(real_world_sns_data['followers'])
    users_with_keywords = extract_keyword_from_bio(real_world_sns_data['users'])
    
    # Compute authority scores
    authority_scores = compute_authority_score(graph, users_with_keywords)
    
    # Check expert authority scores
    # AI expert should have high scores in AI-related topics
    assert authority_scores.loc['ai_expert', 'AI'] > 0.6
    assert authority_scores.loc['ai_expert', 'DeepLearning'] > 0.6
    
    # Data Science expert should have high scores in Data Science topics
    assert authority_scores.loc['data_science_expert', 'DataScience'] > 0.6
    assert authority_scores.loc['data_science_expert', 'Statistics'] > 0.6
    
    # ML expert should have high scores in ML topics
    assert authority_scores.loc['ml_expert', 'MachineLearning'] > 0.6
    assert authority_scores.loc['ml_expert', 'MLOps'] > 0.6
    
    # Check follower influence
    # Data scientists should have some authority in Data Science
    data_scientist_scores = authority_scores.loc[
        ['data_scientist1', 'data_scientist2', 'data_scientist3'],
        'DataScience'
    ]
    assert all(score > 0.3 for score in data_scientist_scores)
    
    # ML engineers should have some authority in Machine Learning
    ml_engineer_scores = authority_scores.loc[
        ['ml_engineer1', 'ml_engineer2', 'ml_engineer3'],
        'MachineLearning'
    ]
    assert all(score > 0.3 for score in ml_engineer_scores)


def test_compute_authority_score_with_topic_overlap(real_world_sns_data):
    """Test authority scores for overlapping topics."""
    graph = preprocess_followers(real_world_sns_data['followers'])
    users_with_keywords = extract_keyword_from_bio(real_world_sns_data['users'])
    authority_scores = compute_authority_score(graph, users_with_keywords)
    
    # Check overlapping topics
    # AI and Machine Learning should have some correlation
    ai_scores = authority_scores['AI']
    ml_scores = authority_scores['MachineLearning']
    correlation = ai_scores.corr(ml_scores)
    assert correlation > 0.5  # Should have positive correlation
    
    # Data Science and Statistics should have some correlation
    ds_scores = authority_scores['DataScience']
    stats_scores = authority_scores['Statistics']
    correlation = ds_scores.corr(stats_scores)
    assert correlation > 0.5  # Should have positive correlation 