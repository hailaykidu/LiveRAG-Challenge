import os
import boto3
from functools import cache
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection


@cache
def get_client(access_key_id: str = None, secret_access_key: str = None, region: str = None):
    if access_key_id is None:
        access_key_id = os.getenv("AWS_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID"))
    if secret_access_key is None:
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if region is None:
        region = os.getenv("AWS_REGION_NAME", "us-east-1")

    credentials = boto3.Session(aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key).get_credentials()
    auth = AWSV4SignerAuth(credentials, region=region)
    aos_client = OpenSearch(
        hosts=[{"host": os.getenv("OPENSEARCH_HOST"), "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )
    return aos_client


def query_opensearch(query: str, top_k: int = 10) -> dict:
    """Query an OpenSearch index and return the results."""
    client = get_client()
    results = client.search(index=os.getenv("OPENSEARCH_INDEX_NAME"), body={"query": {"match": {"text": query}}, "size": top_k})
    return results
