import pytest
from unittest.mock import patch
from google.cloud import bigquery
from discovery_child_development.utils.bigquery import create_client


def test_create_client_with_existing_credentials():
    """
    Test that the client is created successfully when the credentials file already exists.
    """
    # Mock that the credentials file exists
    with patch("os.path.isfile", return_value=True):
        client = create_client()
        assert isinstance(client, bigquery.Client)


def test_create_client_without_credentials_and_successful_download():
    """
    Test that the client is created successfully when the credentials file doesn't exist
    but is downloaded successfully from S3.
    """
    # Mock that the credentials file does not exist and mock the S3 download
    with patch(
        "discovery_child_development.utils.bigquery.path.isfile", return_value=False
    ):
        with patch(
            "discovery_child_development.utils.bigquery.download_file"
        ) as mock_download:
            mock_download.return_value = None  # Successful download
            client = create_client()
            assert isinstance(client, bigquery.Client)


def test_create_client_without_credentials_and_failed_download():
    """
    Test the scenario where the credentials file does not exist and there's an error downloading it.
    """
    with patch(
        "discovery_child_development.utils.bigquery.path.isfile", return_value=False
    ):
        with patch(
            "discovery_child_development.utils.bigquery.download_file"
        ) as mock_download:
            mock_download.side_effect = Exception(
                "Download error"
            )  # Simulate a download error
            with pytest.raises(Exception):
                create_client()


def test_create_client_without_env_variable():
    """
    Test the scenario where the GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.
    """
    with patch.dict(
        "discovery_child_development.utils.bigquery.environ", {}, clear=True
    ):
        with pytest.raises(EnvironmentError):
            create_client()
