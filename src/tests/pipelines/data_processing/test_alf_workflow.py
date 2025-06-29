import pytest
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple

from src.topical_authority_discovery.pipelines.data_processing.nodes import (
    create_graph_and_initial_sc,
    propagate_interests,
    compute_authority_scores,
    assign_topical_authorities
)


class TestPropagateInterests:
    """Test cases for the propagate_interests function."""
    
    def test_basic_interest_propagation(self):
        """Test basic interest propagation with a simple graph."""
        # Create a simple follower graph
        follower_data = pd.DataFrame({
            'follower_id': ['user1', 'user2', 'user3'],
            'followed_id': ['authority1', 'authority1', 'authority2']
        })
        
        # Create user interests
        user_interests = pd.DataFrame({
            'user_id': ['user1', 'user2', 'user3'],
            'topic': ['AI', 'AI', 'ML'],
            'score': [1.0, 1.0, 1.0]
        })
        
        # Create graph and initial matrix
        graph, sc_matrix, topic_to_idx, user_to_idx = create_graph_and_initial_sc(
            follower_data, user_interests
        )
        
        # Test parameters
        alg_params = {'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0}
        
        # Run interest propagation
        f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, alg_params)
        
        # Basic assertions
        assert f_matrix.shape == (2, 5)  # 2 topics, 5 users
        assert np.all(f_matrix >= 0)  # All scores should be non-negative
        assert isinstance(f_matrix, np.ndarray)
        
        # Check that authority1 gets AI authority from followers
        authority1_idx = user_to_idx['authority1']
        ai_idx = topic_to_idx['AI']
        assert f_matrix[ai_idx, authority1_idx] > 0
        
        # Check that authority2 gets ML authority from followers
        authority2_idx = user_to_idx['authority2']
        ml_idx = topic_to_idx['ML']
        assert f_matrix[ml_idx, authority2_idx] > 0
    
    def test_empty_graph(self):
        """Test with empty graph."""
        # Create empty graph
        graph = nx.DiGraph()
        sc_matrix = np.zeros((2, 0))  # 2 topics, 0 users
        topic_to_idx = {'AI': 0, 'ML': 1}
        user_to_idx = {}
        alg_params = {'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0}
        
        f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, alg_params)
        
        assert f_matrix.shape == (2, 0)
        assert np.all(f_matrix == 0)
    
    def test_single_user_graph(self):
        """Test with graph containing only one user."""
        # Create single user graph
        graph = nx.DiGraph()
        graph.add_node('user1')
        
        sc_matrix = np.zeros((2, 1))  # 2 topics, 1 user
        sc_matrix[0, 0] = 1.0  # user1 has interest in topic 0
        
        topic_to_idx = {'AI': 0, 'ML': 1}
        user_to_idx = {'user1': 0}
        alg_params = {'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0}
        
        f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, alg_params)
        
        assert f_matrix.shape == (2, 1)
        # User with no followers should have zero authority scores
        assert np.all(f_matrix[:, 0] == 0)
    
    def test_no_user_interests(self):
        """Test when no users have expressed interests."""
        # Create graph with users but no interests
        follower_data = pd.DataFrame({
            'follower_id': ['user1', 'user2'],
            'followed_id': ['authority1', 'authority1']
        })
        
        # Empty user interests
        user_interests = pd.DataFrame(columns=['user_id', 'topic', 'score'])
        
        graph, sc_matrix, topic_to_idx, user_to_idx = create_graph_and_initial_sc(
            follower_data, user_interests
        )
        
        alg_params = {'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0}
        
        f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, alg_params)
        
        # Should return zero matrix when no interests
        assert f_matrix.shape == (0, 3)  # 0 topics, 3 users
        assert np.all(f_matrix == 0)
    
    def test_disconnected_graph(self):
        """Test with disconnected graph components."""
        # Create disconnected graph
        follower_data = pd.DataFrame({
            'follower_id': ['user1', 'user2'],
            'followed_id': ['authority1', 'authority2']
        })
        
        user_interests = pd.DataFrame({
            'user_id': ['user1', 'user2'],
            'topic': ['AI', 'ML'],
            'score': [1.0, 1.0]
        })
        
        graph, sc_matrix, topic_to_idx, user_to_idx = create_graph_and_initial_sc(
            follower_data, user_interests
        )
        
        alg_params = {'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0}
        
        f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, alg_params)
        
        assert f_matrix.shape == (2, 4)  # 2 topics, 4 users
        
        # Each authority should only get authority from their own followers
        authority1_idx = user_to_idx['authority1']
        authority2_idx = user_to_idx['authority2']
        ai_idx = topic_to_idx['AI']
        ml_idx = topic_to_idx['ML']
        
        # authority1 should have AI authority but not ML
        assert f_matrix[ai_idx, authority1_idx] > 0
        assert f_matrix[ml_idx, authority1_idx] == 0
        
        # authority2 should have ML authority but not AI
        assert f_matrix[ml_idx, authority2_idx] > 0
        assert f_matrix[ai_idx, authority2_idx] == 0
    
    def test_self_loops(self):
        """Test with self-loops in the graph."""
        # Create graph with self-loop
        follower_data = pd.DataFrame({
            'follower_id': ['user1', 'user1'],
            'followed_id': ['authority1', 'user1']  # user1 follows themselves
        })
        
        user_interests = pd.DataFrame({
            'user_id': ['user1'],
            'topic': ['AI'],
            'score': [1.0]
        })
        
        graph, sc_matrix, topic_to_idx, user_to_idx = create_graph_and_initial_sc(
            follower_data, user_interests
        )
        
        alg_params = {'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0}
        
        f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, alg_params)
        
        assert f_matrix.shape == (1, 2)  # 1 topic, 2 users
        # Self-loops should be handled gracefully
        assert np.all(f_matrix >= 0)
    
    def test_different_parameter_values(self):
        """Test with different alpha and beta values."""
        # Create simple test data
        follower_data = pd.DataFrame({
            'follower_id': ['user1'],
            'followed_id': ['authority1']
        })
        
        user_interests = pd.DataFrame({
            'user_id': ['user1'],
            'topic': ['AI'],
            'score': [1.0]
        })
        
        graph, sc_matrix, topic_to_idx, user_to_idx = create_graph_and_initial_sc(
            follower_data, user_interests
        )
        
        # Test different parameter combinations
        test_params = [
            {'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0},
            {'alpha': 0.5, 'beta': 0.1, 'gamma': 1.0},
            {'alpha': 1.0, 'beta': 0.5, 'gamma': 1.0},
        ]
        
        results = []
        for params in test_params:
            f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, params)
            results.append(f_matrix.copy())
        
        # All results should be valid
        for result in results:
            assert result.shape == (1, 2)
            assert np.all(result >= 0)
        
        # Different parameters should produce different results
        assert not np.allclose(results[0], results[1])
        assert not np.allclose(results[1], results[2])
    
    def test_missing_parameters(self):
        """Test with missing parameters (should use defaults)."""
        follower_data = pd.DataFrame({
            'follower_id': ['user1'],
            'followed_id': ['authority1']
        })
        
        user_interests = pd.DataFrame({
            'user_id': ['user1'],
            'topic': ['AI'],
            'score': [1.0]
        })
        
        graph, sc_matrix, topic_to_idx, user_to_idx = create_graph_and_initial_sc(
            follower_data, user_interests
        )
        
        # Test with empty parameters dict
        alg_params = {}
        
        f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, alg_params)
        
        assert f_matrix.shape == (1, 2)
        assert np.all(f_matrix >= 0)
    
    def test_large_graph_performance(self):
        """Test performance with larger graph."""
        # Create larger test data
        n_users = 100
        n_topics = 10
        
        # Create random follower relationships
        follower_data = pd.DataFrame({
            'follower_id': [f'user{i}' for i in range(n_users)],
            'followed_id': [f'authority{i % 10}' for i in range(n_users)]
        })
        
        # Create random user interests
        user_interests = []
        for i in range(n_users):
            for j in range(3):  # Each user has 3 interests
                user_interests.append({
                    'user_id': f'user{i}',
                    'topic': f'topic{j}',
                    'score': 1.0
                })
        
        user_interests = pd.DataFrame(user_interests)
        
        graph, sc_matrix, topic_to_idx, user_to_idx = create_graph_and_initial_sc(
            follower_data, user_interests
        )
        
        alg_params = {'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0}
        
        # Should complete without errors
        f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, alg_params)
        
        assert f_matrix.shape == (len(topic_to_idx), len(user_to_idx))
        assert np.all(f_matrix >= 0)
    
    def test_zero_scores_in_sc_matrix(self):
        """Test with zero scores in the initial interest matrix."""
        follower_data = pd.DataFrame({
            'follower_id': ['user1', 'user2'],
            'followed_id': ['authority1', 'authority1']
        })
        
        # Create user interests with some zero scores
        user_interests = pd.DataFrame({
            'user_id': ['user1', 'user2'],
            'topic': ['AI', 'AI'],
            'score': [1.0, 0.0]  # user2 has zero interest
        })
        
        graph, sc_matrix, topic_to_idx, user_to_idx = create_graph_and_initial_sc(
            follower_data, user_interests
        )
        
        alg_params = {'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0}
        
        f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, alg_params)
        
        assert f_matrix.shape == (1, 3)  # 1 topic, 3 users
        
        # authority1 should still get some authority from user1
        authority1_idx = user_to_idx['authority1']
        ai_idx = topic_to_idx['AI']
        assert f_matrix[ai_idx, authority1_idx] > 0


class TestCreateGraphAndInitialSc:
    """Test cases for the create_graph_and_initial_sc function."""
    
    def test_basic_graph_creation(self):
        """Test basic graph and matrix creation."""
        follower_data = pd.DataFrame({
            'follower_id': ['user1', 'user2'],
            'followed_id': ['authority1', 'authority1']
        })
        
        user_interests = pd.DataFrame({
            'user_id': ['user1', 'user2'],
            'topic': ['AI', 'ML'],
            'score': [1.0, 1.0]
        })
        
        graph, sc_matrix, topic_to_idx, user_to_idx = create_graph_and_initial_sc(
            follower_data, user_interests
        )
        
        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes()) == 3  # user1, user2, authority1
        assert len(graph.edges()) == 2
        
        assert sc_matrix.shape == (2, 3)  # 2 topics, 3 users
        assert len(topic_to_idx) == 2
        assert len(user_to_idx) == 3
        
        # Check that interests are properly set
        user1_idx = user_to_idx['user1']
        user2_idx = user_to_idx['user2']
        ai_idx = topic_to_idx['AI']
        ml_idx = topic_to_idx['ML']
        
        assert sc_matrix[ai_idx, user1_idx] == 1.0
        assert sc_matrix[ml_idx, user2_idx] == 1.0


class TestComputeAuthorityScores:
    """Test cases for the compute_authority_scores function."""
    
    def test_basic_authority_scoring(self):
        """Test basic authority score computation."""
        # Create test data
        follower_data = pd.DataFrame({
            'follower_id': ['user1', 'user2'],
            'followed_id': ['authority1', 'authority1']
        })
        
        user_interests = pd.DataFrame({
            'user_id': ['user1', 'user2'],
            'topic': ['AI', 'AI'],
            'score': [1.0, 1.0]
        })
        
        graph, sc_matrix, topic_to_idx, user_to_idx = create_graph_and_initial_sc(
            follower_data, user_interests
        )
        
        alg_params = {'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0}
        f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, alg_params)
        
        wzf_matrix, zf_matrix = compute_authority_scores(
            f_matrix, graph, topic_to_idx, user_to_idx
        )
        
        assert wzf_matrix.shape == f_matrix.shape
        assert zf_matrix.shape == f_matrix.shape
        assert isinstance(wzf_matrix, np.ndarray)
        assert isinstance(zf_matrix, np.ndarray)


class TestAssignTopicalAuthorities:
    """Test cases for the assign_topical_authorities function."""
    
    def test_basic_authority_assignment(self):
        """Test basic authority assignment."""
        # Create test data
        follower_data = pd.DataFrame({
            'follower_id': ['user1', 'user2'],
            'followed_id': ['authority1', 'authority1']
        })
        
        user_interests = pd.DataFrame({
            'user_id': ['user1', 'user2'],
            'topic': ['AI', 'AI'],
            'score': [1.0, 1.0]
        })
        
        graph, sc_matrix, topic_to_idx, user_to_idx = create_graph_and_initial_sc(
            follower_data, user_interests
        )
        
        alg_params = {'alpha': 0.1, 'beta': 0.01, 'gamma': 1.0}
        f_matrix = propagate_interests(graph, sc_matrix, topic_to_idx, user_to_idx, alg_params)
        
        wzf_matrix, zf_matrix = compute_authority_scores(
            f_matrix, graph, topic_to_idx, user_to_idx
        )
        
        final_authorities = assign_topical_authorities(
            wzf_matrix, f_matrix, zf_matrix, topic_to_idx, user_to_idx, alg_params
        )
        
        assert isinstance(final_authorities, pd.DataFrame)
        assert 'user_id' in final_authorities.columns
        assert 'assigned_topic' in final_authorities.columns
        assert 'authority_score' in final_authorities.columns 