import pytest
import pandas as pd
import networkx as nx


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