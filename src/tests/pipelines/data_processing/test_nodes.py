import pandas as pd
import networkx as nx
import pytest
from topical_authority_discovery.pipelines.data_processing.nodes import (
    compute_authority_score,
    preprocess_followers,
    extract_keyword_from_bio,
    construct_fashion_knowledge_base,
    link_fashion_entities,
)
import os
import tempfile
from pathlib import Path


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
        'user_id': ['user1', 'user2', 'user3', 'user4', 'authority1', 'authority2', 'fashion1', 'fashion2', 'fashion3'],
        'bio': [
            'AI enthusiast and machine learning practitioner',
            'Data scientist working on big data projects',
            'Software engineer interested in AI and ML',
            'Research scientist in data analytics',
            'AI researcher with 10+ years experience in deep learning',
            'Data Science expert specializing in big data analytics',
            'Fashion designer specializing in minimalist style and sustainable fashion',
            'Streetwear enthusiast and urban style influencer',
            'Vintage fashion collector and retro style expert'
        ],
        'topics_of_interest': [
            'AI||MachineLearning',
            'DataScience||BigData',
            'AI||MachineLearning',
            'DataScience||Statistics',
            'AI||MachineLearning||DeepLearning',
            'DataScience||Statistics||BigData',
            'Fashion||Minimalism||Sustainable',
            'Fashion||Streetwear||Urban',
            'Fashion||Vintage||Retro'
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


@pytest.fixture
def fashion_test_data():
    """Create sample data with fashion-related bios."""
    return pd.DataFrame({
        'user_id': ['fashion1', 'fashion2', 'fashion3'],
        'bio': [
            'Fashion designer specializing in minimalist style and sustainable fashion',
            'Streetwear enthusiast and urban style influencer',
            'Vintage fashion collector and retro style expert'
        ],
        'topics_of_interest': [
            'Fashion||Minimalism',
            'Fashion||Streetwear',
            'Fashion||Vintage'
        ]
    })


@pytest.fixture
def fashion_entities_data():
    """Create sample fashion entities data."""
    return pd.DataFrame({
        'entity_id': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        'name': [
            'Minimalist Style',
            'Streetwear',
            'Vintage Fashion',
            'Sustainable Fashion',
            'High Fashion'
        ],
        'aliases': [
            'minimalist style||minimalist fashion||minimalist look||minimalist aesthetic',
            'streetwear||street fashion||urban style||street style',
            'vintage fashion||retro style||vintage clothing||retro fashion',
            'sustainable fashion||eco fashion||ethical fashion||slow fashion',
            'high fashion||haute couture||luxury fashion||designer fashion'
        ]
    })


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


def test_extract_keyword_from_bio(sample_users_data, fashion_entities_data):
    """Test the keyword extraction from bios."""
    users_with_keywords = extract_keyword_from_bio(sample_users_data, fashion_entities_data)
    
    # Print debug information
    print("\nExtracted keywords:")
    for idx, row in users_with_keywords.iterrows():
        print(f"User {row['user_id']}:")
        print(f"  Bio: {row['bio']}")
        print(f"  Keywords: {row['extracted_keywords']}")
        print(f"  Fashion entities: {row['fashion_entities']}")
    
    # Check if new columns are added
    assert 'extracted_keywords' in users_with_keywords.columns
    assert 'fashion_entities' in users_with_keywords.columns
    
    # Check if keywords are extracted (non-empty)
    assert all(users_with_keywords['extracted_keywords'].str.len() > 0)
    
    # Check if keywords are properly formatted
    for keywords in users_with_keywords['extracted_keywords']:
        # Check if it's a single phrase or multiple phrases
        if '||' in keywords:
            # Multiple phrases should be properly separated
            phrases = keywords.split('||')
            assert len(phrases) > 1
            # Each phrase should be non-empty
            assert all(len(phrase.strip()) > 0 for phrase in phrases)
        else:
            # Single phrase should be non-empty
            assert len(keywords.strip()) > 0
    
    # Check if fashion entities are linked
    assert all(users_with_keywords['fashion_entities'].str.len() >= 0)  # Can be empty if no fashion terms found
    
    # Check specific fashion entity matches
    fashion1_entities = users_with_keywords.loc[users_with_keywords['user_id'] == 'fashion1', 'fashion_entities'].iloc[0]
    assert 'Q1' in fashion1_entities  # Minimalist Style
    assert 'Q4' in fashion1_entities  # Sustainable Fashion
    
    fashion2_entities = users_with_keywords.loc[users_with_keywords['user_id'] == 'fashion2', 'fashion_entities'].iloc[0]
    assert 'Q2' in fashion2_entities  # Streetwear
    
    fashion3_entities = users_with_keywords.loc[users_with_keywords['user_id'] == 'fashion3', 'fashion_entities'].iloc[0]
    assert 'Q3' in fashion3_entities  # Vintage Fashion


def test_extract_keyword_from_bio_with_special_characters(fashion_entities_data):
    """Test keyword extraction with special characters in bios."""
    special_chars_data = pd.DataFrame({
        'user_id': ['user1'],
        'bio': ['AI & ML expert (10+ years) - working on #DeepLearning, @NLP projects'],
        'topics_of_interest': ['AI||MachineLearning']
    })
    
    users_with_keywords = extract_keyword_from_bio(special_chars_data, fashion_entities_data)
    
    # Check if special characters are handled
    assert len(users_with_keywords['extracted_keywords'].iloc[0]) > 0
    # Check if keywords are properly formatted
    keywords = users_with_keywords['extracted_keywords'].iloc[0]
    if '||' in keywords:
        phrases = keywords.split('||')
        assert len(phrases) > 1
        assert all(len(phrase.strip()) > 0 for phrase in phrases)
    else:
        assert len(keywords.strip()) > 0


def test_compute_authority_score(sample_followers_data, sample_users_data, fashion_entities_data):
    """Test the authority score computation."""
    # Create graph and process users
    graph = preprocess_followers(sample_followers_data)
    
    # Debug: Print graph information
    print("\nGraph Information:")
    print(f"Number of nodes: {len(graph.nodes())}")
    print(f"Number of edges: {len(graph.edges())}")
    print("Nodes:", list(graph.nodes()))
    print("Edges:", list(graph.edges()))
    
    # Filter users to only include those in the graph
    users_in_graph = sample_users_data[sample_users_data['user_id'].isin(graph.nodes())]
    print("\nUsers in graph:")
    print(users_in_graph[['user_id', 'topics_of_interest']])
    
    users_with_keywords = extract_keyword_from_bio(users_in_graph, fashion_entities_data)
    
    # Compute authority scores
    authority_scores = compute_authority_score(graph, users_with_keywords)
    
    # Debug: Print authority scores
    print("\nAuthority Scores:")
    print(authority_scores)
    print("\nScore Statistics:")
    print("Min score:", authority_scores.min().min())
    print("Max score:", authority_scores.max().max())
    print("Mean score:", authority_scores.mean().mean())
    
    # Check if output is a DataFrame
    assert isinstance(authority_scores, pd.DataFrame)
    
    # Check if all users are present
    assert set(authority_scores.index) == set(graph.nodes())
    
    # Check if all topics are present
    expected_topics = {'AI', 'MachineLearning', 'DataScience', 'BigData', 'Statistics', 'DeepLearning'}
    print("\nTopics Check:")
    print("Expected topics:", expected_topics)
    print("Actual topics:", set(authority_scores.columns))
    assert set(authority_scores.columns) == expected_topics
    
    # Check if scores are between 0 and 1
    assert authority_scores.min().min() >= 0
    # Debug: Print problematic scores
    if authority_scores.max().max() > 1:
        print("\nScores above 1:")
        for col in authority_scores.columns:
            max_score = authority_scores[col].max()
            if max_score > 1:
                print(f"{col}: {max_score}")
                print("Users with high scores:")
                print(authority_scores[authority_scores[col] > 1][[col]])
    assert authority_scores.max().max() <= 1
    
    # Check specific authority relationships
    print("\nAuthority Relationships:")
    for user in ['authority1', 'authority2']:
        print(f"\n{user} scores:")
        print(authority_scores.loc[user])
    
    # authority1 should have high AI and MachineLearning scores
    assert authority_scores.loc['authority1', 'AI'] > 0.4
    assert authority_scores.loc['authority1', 'MachineLearning'] > 0.4
    
    # authority2 should have high DataScience and BigData scores
    assert authority_scores.loc['authority2', 'DataScience'] > 0.4
    assert authority_scores.loc['authority2', 'BigData'] > 0.4


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


def test_compute_authority_score_with_real_world_data(real_world_sns_data, fashion_entities_data):
    """Test authority score computation with real-world SNS community data."""
    # Create graph and process users
    graph = preprocess_followers(real_world_sns_data['followers'])
    
    # Debug: Print graph information
    print("\nGraph Information:")
    print(f"Number of nodes: {len(graph.nodes())}")
    print(f"Number of edges: {len(graph.edges())}")
    print("Nodes:", list(graph.nodes()))
    print("Edges:", list(graph.edges()))
    
    # Filter users to only include those in the graph
    users_in_graph = real_world_sns_data['users'][real_world_sns_data['users']['user_id'].isin(graph.nodes())]
    print("\nUsers in graph:")
    print(users_in_graph[['user_id', 'topics_of_interest']])
    
    users_with_keywords = extract_keyword_from_bio(users_in_graph, fashion_entities_data)
    
    # Compute authority scores
    authority_scores = compute_authority_score(graph, users_with_keywords)
    
    # Debug: Print authority scores
    print("\nAuthority Scores:")
    print(authority_scores)
    print("\nScore Statistics:")
    print("Min score:", authority_scores.min().min())
    print("Max score:", authority_scores.max().max())
    print("Mean score:", authority_scores.mean().mean())
    
    # Check expert authority scores
    print("\nExpert Scores:")
    for expert in ['ai_expert', 'data_science_expert', 'ml_expert']:
        print(f"\n{expert} scores:")
        print(authority_scores.loc[expert])
    
    # AI expert should have high scores in AI-related topics
    assert authority_scores.loc['ai_expert', 'AI'] > 0.5
    assert authority_scores.loc['ai_expert', 'DeepLearning'] > 0.5
    
    # Data Science expert should have high scores in Data Science topics
    assert authority_scores.loc['data_science_expert', 'DataScience'] > 0.5
    assert authority_scores.loc['data_science_expert', 'Statistics'] > 0.5
    
    # ML expert should have high scores in ML topics
    assert authority_scores.loc['ml_expert', 'MachineLearning'] > 0.5
    # Debug: Print MLOps scores
    print("\nMLOps Scores:")
    print(authority_scores['MLOps'].sort_values(ascending=False))
    assert authority_scores.loc['ml_expert', 'MLOps'] > 0.5
    
    # Check follower influence
    print("\nFollower Influence:")
    # Data scientists should have some authority in Data Science
    data_scientist_scores = authority_scores.loc[
        ['data_scientist1', 'data_scientist2', 'data_scientist3'],
        'DataScience'
    ]
    print("Data Scientist scores:", data_scientist_scores)
    assert all(score > 0.3 for score in data_scientist_scores)
    
    # ML engineers should have some authority in Machine Learning
    ml_engineer_scores = authority_scores.loc[
        ['ml_engineer1', 'ml_engineer2', 'ml_engineer3'],
        'MachineLearning'
    ]
    print("ML Engineer scores:", ml_engineer_scores)
    assert all(score > 0.3 for score in ml_engineer_scores)


def test_compute_authority_score_with_topic_overlap(real_world_sns_data, fashion_entities_data):
    """Test authority scores for overlapping topics."""
    graph = preprocess_followers(real_world_sns_data['followers'])
    users_with_keywords = extract_keyword_from_bio(real_world_sns_data['users'], fashion_entities_data)
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


def test_construct_fashion_knowledge_base(fashion_entities_data):
    """Test the construction of the fashion knowledge base."""
    kb = construct_fashion_knowledge_base(fashion_entities_data)
    
    # Test that knowledge base is created
    assert kb is not None
    
    # Test that entities are added
    entity_strings = kb.get_entity_strings()
    assert len(entity_strings) == 5  # Should have 5 entities
    
    # Test that each entity has the correct name
    for entity_id, name in fashion_entities_data.set_index('entity_id')['name'].items():
        assert entity_id in entity_strings
        assert name in entity_strings[entity_id]


def test_link_fashion_entities(fashion_entities_data):
    """Test linking fashion entities in text."""
    kb = construct_fashion_knowledge_base(fashion_entities_data)
    
    # Test with known fashion terms
    text = "She loves minimalist style and sustainable fashion"
    entities = link_fashion_entities(text, kb)
    
    assert len(entities) > 0
    assert any(e["entity_id"] == "Q1" for e in entities)  # Minimalist Style
    assert any(e["entity_id"] == "Q4" for e in entities)  # Sustainable Fashion
    
    # Test with unknown terms
    text = "She loves pizza and movies"
    entities = link_fashion_entities(text, kb)
    assert len(entities) == 0


def test_extract_keyword_from_bio_with_fashion(fashion_test_data, fashion_entities_data):
    """Test keyword extraction with fashion entity linking."""
    users_with_keywords = extract_keyword_from_bio(fashion_test_data, fashion_entities_data)
    
    # Check if new columns are added
    assert 'extracted_keywords' in users_with_keywords.columns
    assert 'fashion_entities' in users_with_keywords.columns
    
    # Check if fashion entities are linked
    assert all(users_with_keywords['fashion_entities'].str.len() > 0)
    
    # Check specific fashion entity links
    fashion1_entities = users_with_keywords.loc[users_with_keywords['user_id'] == 'fashion1', 'fashion_entities'].iloc[0]
    assert 'Q1' in fashion1_entities  # Minimalist Style
    assert 'Q4' in fashion1_entities  # Sustainable Fashion
    
    fashion2_entities = users_with_keywords.loc[users_with_keywords['user_id'] == 'fashion2', 'fashion_entities'].iloc[0]
    assert 'Q2' in fashion2_entities  # Streetwear
    
    fashion3_entities = users_with_keywords.loc[users_with_keywords['user_id'] == 'fashion3', 'fashion_entities'].iloc[0]
    assert 'Q3' in fashion3_entities  # Vintage Fashion


def test_link_fashion_entities_batch_processing(fashion_entities_data):
    """Test linking fashion entities using batch processing with multiple phrases."""
    print("\n=== Starting Batch Processing Test ===")
    print("\n1. Constructing Knowledge Base...")
    kb = construct_fashion_knowledge_base(fashion_entities_data)
    print(f"Knowledge Base created with {len(kb.get_entity_strings())} entities")
    
    # Test with multiple phrases at once
    test_phrases = [
        "minimalist style and sustainable fashion",
        "streetwear and urban style",
        "vintage clothing and retro fashion",
        "haute couture and luxury fashion",
        "eco fashion and ethical clothing"
    ]
    
    # Map phrases to expected entity IDs
    expected_entities = {
        "minimalist style and sustainable fashion": ["Q1", "Q4"],  # Minimalist Style, Sustainable Fashion
        "streetwear and urban style": ["Q2"],  # Streetwear
        "vintage clothing and retro fashion": ["Q3"],  # Vintage Fashion
        "haute couture and luxury fashion": ["Q5"],  # High Fashion
        "eco fashion and ethical clothing": ["Q4"]  # Sustainable Fashion
    }
    
    print("\n2. Processing Phrases in Batch:")
    # Process all phrases
    for phrase in test_phrases:
        print(f"\n--- Processing Phrase: '{phrase}' ---")
        print(f"Expected entities: {expected_entities[phrase]}")
        
        entities = link_fashion_entities(phrase, kb)
        print(f"Found entities: {entities}")
        
        # Get expected entity IDs for this phrase
        expected_ids = expected_entities[phrase]
        
        # Verify number of entities found
        print(f"Verifying number of entities: Expected {len(expected_ids)}, Found {len(entities)}")
        assert len(entities) == len(expected_ids), \
            f"Expected {len(expected_ids)} entities for phrase '{phrase}', but found {len(entities)}"
        
        # Verify each expected entity is found
        found_entity_ids = {entity['entity_id'] for entity in entities}
        print(f"Found entity IDs: {found_entity_ids}")
        for expected_id in expected_ids:
            print(f"Checking for expected entity: {expected_id}")
            assert expected_id in found_entity_ids, \
                f"Expected entity {expected_id} not found for phrase '{phrase}'"
        
        # Verify confidence scores
        print("Verifying confidence scores:")
        for entity in entities:
            print(f"Entity {entity['entity_id']}: confidence = {entity['confidence']}")
            assert 0 <= entity['confidence'] <= 1, \
                f"Invalid confidence score {entity['confidence']} for entity {entity['entity_id']}"
    
    print("\n=== Batch Processing Test Completed Successfully ===")


def test_link_fashion_entities_detailed(fashion_entities_data):
    """Test linking fashion entities with detailed input/output examples."""
    print("\n=== Starting Detailed Test ===")
    print("\n1. Constructing Knowledge Base...")
    kb = construct_fashion_knowledge_base(fashion_entities_data)
    print(f"Knowledge Base created with {len(kb.get_entity_strings())} entities")
    
    # Test case 1: Single fashion term
    text = "minimalist style"
    print(f"\n2. Test Case 1 - Single Term:")
    print(f"Input text: {text}")
    entities = link_fashion_entities(text, kb)
    print(f"Found entities: {entities}")
    assert len(entities) == 1
    assert entities[0]["entity_id"] == "Q1"
    assert entities[0]["text"] == "minimalist style"
    print("✓ Single term test passed")
    
    # Test case 2: Multiple fashion terms
    text = "sustainable fashion and vintage clothing"
    print(f"\n3. Test Case 2 - Multiple Terms:")
    print(f"Input text: {text}")
    entities = link_fashion_entities(text, kb)
    print(f"Found entities: {entities}")
    assert len(entities) == 2
    entity_ids = {e["entity_id"] for e in entities}
    assert "Q4" in entity_ids  # Sustainable Fashion
    assert "Q3" in entity_ids  # Vintage Fashion
    print("✓ Multiple terms test passed")
    
    # Test case 3: Alias matching
    text = "haute couture"
    print(f"\n4. Test Case 3 - Alias Matching:")
    print(f"Input text: {text}")
    entities = link_fashion_entities(text, kb)
    print(f"Found entities: {entities}")
    assert len(entities) == 1
    assert entities[0]["entity_id"] == "Q5"  # High Fashion
    print("✓ Alias matching test passed")
    
    # Test case 4: No fashion terms
    text = "pizza and movies"
    print(f"\n5. Test Case 4 - No Fashion Terms:")
    print(f"Input text: {text}")
    entities = link_fashion_entities(text, kb)
    print(f"Found entities: {entities}")
    assert len(entities) == 0
    print("✓ No fashion terms test passed")
    
    # Test case 5: Mixed content
    text = "Fashion designer specializing in minimalist style and sustainable fashion"
    print(f"\n6. Test Case 5 - Mixed Content:")
    print(f"Input text: {text}")
    entities = link_fashion_entities(text, kb)
    print(f"Found entities: {entities}")
    assert len(entities) >= 2
    entity_ids = {e["entity_id"] for e in entities}
    assert "Q1" in entity_ids  # Minimalist Style
    assert "Q4" in entity_ids  # Sustainable Fashion
    print("✓ Mixed content test passed")
    
    print("\n=== Detailed Test Completed Successfully ===")


def test_construct_fashion_knowledge_base_basic(fashion_entities_data):
    """Test basic knowledge base construction without file persistence."""
    # Construct knowledge base
    kb = construct_fashion_knowledge_base(fashion_entities_data)
    
    # Check if knowledge base is created
    assert kb is not None
    
    # Check if all entities are added
    entity_strings = kb.get_entity_strings()
    assert len(entity_strings) == len(fashion_entities_data)
    
    # Check if each entity has the correct name
    for entity_id, name in fashion_entities_data.set_index('entity_id')['name'].items():
        assert entity_id in entity_strings
        assert name in entity_strings[entity_id]
    
    # Check if aliases are properly added
    for _, row in fashion_entities_data.iterrows():
        aliases = row['aliases'].split('||')
        for alias in aliases:
            candidates = kb.get_candidates(alias.strip())
            assert len(candidates) > 0
            assert any(c.entity_ == row['entity_id'] for c in candidates)


def test_construct_fashion_knowledge_base_with_file_persistence(fashion_entities_data):
    """Test knowledge base construction with file persistence."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_path = os.path.join(temp_dir, "test_kb.spacy")
        
        # First construction and save
        kb1 = construct_fashion_knowledge_base(fashion_entities_data, kb_path)
        
        # Verify file was created
        assert os.path.exists(kb_path)
        
        # Second construction should load from file
        kb2 = construct_fashion_knowledge_base(fashion_entities_data, kb_path)
        
        # Both knowledge bases should have the same entities
        assert set(kb1.get_entity_strings().keys()) == set(kb2.get_entity_strings().keys())
        
        # Both knowledge bases should have the same aliases
        for entity_id in kb1.get_entity_strings().keys():
            kb1_aliases = set(kb1.get_aliases(entity_id))
            kb2_aliases = set(kb2.get_aliases(entity_id))
            assert kb1_aliases == kb2_aliases


def test_construct_fashion_knowledge_base_with_invalid_file(fashion_entities_data):
    """Test knowledge base construction with invalid file path."""
    # Try to load from non-existent file
    kb = construct_fashion_knowledge_base(fashion_entities_data, "nonexistent/path/kb.spacy")
    
    # Should still create a valid knowledge base
    assert kb is not None
    assert len(kb.get_entity_strings()) == len(fashion_entities_data)


def test_construct_fashion_knowledge_base_with_empty_data():
    """Test knowledge base construction with empty data."""
    empty_data = pd.DataFrame(columns=['entity_id', 'name', 'aliases'])
    
    # Should create an empty knowledge base
    kb = construct_fashion_knowledge_base(empty_data)
    assert kb is not None
    assert len(kb.get_entity_strings()) == 0


def test_construct_fashion_knowledge_base_with_missing_columns():
    """Test knowledge base construction with missing required columns."""
    invalid_data = pd.DataFrame({
        'entity_id': ['Q1'],
        'name': ['Test Entity']
        # Missing 'aliases' column
    })
    
    with pytest.raises(KeyError):
        construct_fashion_knowledge_base(invalid_data)


def test_construct_fashion_knowledge_base_with_duplicate_entities(fashion_entities_data):
    """Test knowledge base construction with duplicate entity IDs."""
    # Create duplicate entity data
    duplicate_data = pd.concat([
        fashion_entities_data,
        fashion_entities_data.iloc[[0]]  # Duplicate first row
    ])
    
    # Should handle duplicates gracefully
    kb = construct_fashion_knowledge_base(duplicate_data)
    
    # Should have unique entities
    entity_strings = kb.get_entity_strings()
    assert len(entity_strings) == len(fashion_entities_data)
    
    # Duplicate entity should have combined aliases
    first_entity = fashion_entities_data.iloc[0]
    aliases = kb.get_aliases(first_entity['entity_id'])
    assert len(aliases) > 0


def test_link_fashion_entities_with_phrases(fashion_entities_data):
    """Test linking fashion entities with specific phrases from the test data."""
    kb = construct_fashion_knowledge_base(fashion_entities_data)
    
    # Test with phrases from fashion_test_data
    test_phrases = [
        "minimalist style",
        "sustainable fashion",
        "streetwear",
        "vintage fashion",
        "retro style"
    ]
    
    # Map phrases to expected entity IDs
    expected_entities = {
        "minimalist style": "Q1",
        "sustainable fashion": "Q4",
        "streetwear": "Q2",
        "vintage fashion": "Q3",
        "retro style": "Q3"
    }
    
    for phrase in test_phrases:
        print(f"Testing phrase: {phrase}")
        entities = link_fashion_entities(phrase, kb)
        print(f"Found entities: {entities}")
        
        if phrase in expected_entities:
            assert len(entities) > 0, f"No entities found for phrase: {phrase}"
            assert any(entity['entity_id'] == expected_entities[phrase] for entity in entities), \
                f"Expected entity {expected_entities[phrase]} not found for phrase: {phrase}" 