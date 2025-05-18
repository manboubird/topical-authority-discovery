from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_followers,
    extract_keyword_from_bio,
    compute_authority_score,
    construct_fashion_knowledge_base,
    combine_user_datasets,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
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
            )
        ]
    )
