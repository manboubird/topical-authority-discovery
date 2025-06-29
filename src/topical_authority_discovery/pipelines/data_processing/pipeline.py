from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_followers,
    extract_keyword_from_bio,
    compute_authority_score,
    construct_fashion_knowledge_base,
    combine_user_datasets,
    load_bigquery_to_duckdb,
    create_graph_and_initial_sc,
    propagate_interests,
    compute_authority_scores,
    assign_topical_authorities,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # ALF Algorithm 1 Workflow Nodes
            node(
                func=create_graph_and_initial_sc,
                inputs=["raw_follower_data", "raw_user_interests"],
                outputs=["graph", "sc_matrix", "topic_to_idx", "user_to_idx"],
                name="create_graph_and_initial_sc_node",
                tags=["alf_workflow", "graph_construction"],
            ),
            node(
                func=propagate_interests,
                inputs=["graph", "sc_matrix", "topic_to_idx", "user_to_idx", "params:alg_params"],
                outputs="f_matrix",
                name="propagate_interests_node",
                tags=["alf_workflow", "interest_propagation"],
            ),
            node(
                func=compute_authority_scores,
                inputs=["f_matrix", "graph", "topic_to_idx", "user_to_idx"],
                outputs=["wzf_matrix", "zf_matrix"],
                name="compute_authority_scores_node",
                tags=["alf_workflow", "authority_scoring"],
            ),
            node(
                func=assign_topical_authorities,
                inputs=["wzf_matrix", "f_matrix", "zf_matrix", "topic_to_idx", "user_to_idx", "params:alg_params"],
                outputs="final_authorities",
                name="assign_topical_authorities_node",
                tags=["alf_workflow", "authority_assignment"],
            ),
            # Legacy nodes (keeping for backward compatibility)
            node(
                func=preprocess_followers,
                inputs="followers",
                outputs="preprocessed_followers",
                name="preprocess_followers_node",
                tags=["preprocessing", "graph_construction"],
            ),
            node(
                func=construct_fashion_knowledge_base,
                inputs=["fashion_entities", "params:fashion_knowledge_base_path"],
                outputs=None,
                name="construct_fashion_knowledge_base_node",
                tags=["preprocessing", "knowledge_base"],
            ),
            node(
                func=combine_user_datasets,
                inputs=["fashion_users_40_60", "users"],
                outputs="combined_users",
                name="combine_user_datasets_node",
                tags=["preprocessing", "data_combination"],
            ),
            node(
                func=extract_keyword_from_bio,
                inputs=["combined_users", "fashion_entities", "params:fashion_knowledge_base_path"],
                outputs="users_with_keywords",
                name="extract_keyword_from_bio_node",
                tags=["preprocessing", "keyword_extraction"],
            ),
            node(
                func=compute_authority_score,
                inputs=["preprocessed_followers", "users_with_keywords"],
                outputs="authority_scores",
                name="compute_authority_score_node",
                tags=["authority_discovery"],
            ),
            # BigQuery loading nodes
            node(
                func=load_bigquery_to_duckdb,
                inputs={
                    "table_id": "params:accuweather_table_id",
                    "limit": "params:accuweather_limit"
                },
                outputs="accuweather_data",
                name="load_accuweather_data_node",
                tags=["data_loading", "bigquery"],
            ),
            node(
                func=load_bigquery_to_duckdb,
                inputs={
                    "table_id": "params:google_trends_rising_table_id",
                    "limit": "params:google_trends_rising_limit"
                },
                outputs="google_trends_rising_data",
                name="load_google_trends_rising_node",
                tags=["data_loading", "bigquery"],
            ),
            node(
                func=load_bigquery_to_duckdb,
                inputs={
                    "table_id": "params:google_trends_top_table_id",
                    "limit": "params:google_trends_top_limit"
                },
                outputs="google_trends_top_data",
                name="load_google_trends_top_node",
                tags=["data_loading", "bigquery"],
            )
        ]
    )
