from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table,
    preprocess_companies,
    preprocess_shuttles,
    preprocess_followers,
    extract_keyword_from_bio,
    compute_authority_score,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_followers,
                inputs="followers",
                outputs="preprocessed_followers",
                name="preprocess_followers_node",
            ),
            node(
                func=extract_keyword_from_bio,
                inputs=["users", "fashion_entities"],
                outputs="users_with_keywords",
                name="extract_keyword_from_bio_node",
            ),
            node(
                func=compute_authority_score,
                inputs=["preprocessed_followers", "users_with_keywords"],
                outputs="authority_scores",
                name="compute_authority_score_node",
            ),
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs="preprocessed_companies",
                name="preprocess_companies_node",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="preprocessed_shuttles",
                name="preprocess_shuttles_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_shuttles", "preprocessed_companies", "reviews"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
        ]
    )
