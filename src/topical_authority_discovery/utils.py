import logging
import time
from pathlib import Path
from google.cloud import bigquery
from google.api_core import exceptions
from google.oauth2 import service_account
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

def get_bigquery_client(billing_project_id: str, service_account_path: str) -> bigquery.Client:
    """
    Initialize and return a BigQuery client with proper authentication.
    
    Args:
        billing_project_id: The GCP project where costs are charged
        service_account_path: Path to the service account credentials file
    
    Returns:
        bigquery.Client: Authenticated BigQuery client
    """
    if not Path(service_account_path).exists():
        raise FileNotFoundError(f"Service account credentials file not found at: {service_account_path}")
    
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    logging.info(f"Authenticated as: {credentials.service_account_email}")
    
    return bigquery.Client(
        project=billing_project_id,
        credentials=credentials
    )

def verify_bigquery_permissions(client: bigquery.Client, host_project_id: str, dataset_id: str, table_id: str) -> None:
    """
    Verify that the service account has necessary permissions to access the BigQuery table.
    
    Args:
        client: Authenticated BigQuery client
        host_project_id: The GCP project where the data is stored
        dataset_id: The dataset ID containing the table
        table_id: The table ID to verify access for
    """
    try:
        test_query = f"SELECT 1 FROM `{host_project_id}.{dataset_id}.{table_id}` LIMIT 1"
        client.query(test_query).result()
        logging.info("Successfully verified BigQuery permissions")
    except exceptions.Forbidden as e:
        logging.error(f"Permission denied: {str(e)}")
        logging.error("Please ensure the service account has the following permissions:")
        logging.error("- bigquery.jobs.create")
        logging.error("- bigquery.tables.getData")
        logging.error("- bigquery.tables.get")
        raise

def create_bigquery_query_job(
    client: bigquery.Client,
    host_project_id: str,
    dataset_id: str,
    table_id: str,
    limit: int = 100
) -> bigquery.QueryJob:
    """
    Create and return a BigQuery query job for the specified table.
    
    Args:
        client: Authenticated BigQuery client
        host_project_id: The GCP project where the data is stored
        dataset_id: The dataset ID containing the table
        table_id: The table ID to query
        limit: Maximum number of rows to return
    
    Returns:
        bigquery.QueryJob: The query job object
    """
    query = f"SELECT * FROM `{host_project_id}.{dataset_id}.{table_id}` LIMIT {limit}"
    logging.info(f"Running query: {query}")
    
    try:
        return client.query(query)
    except exceptions.GoogleAPIError as e:
        logging.error(f"Error executing BigQuery query: {str(e)}")
        raise 