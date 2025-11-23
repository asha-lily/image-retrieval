from pathlib import Path

import pytest
from unittest.mock import Mock, MagicMock, patch

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from image_retrieval.create_image_vector_db import VectorDBImageWriter


@pytest.fixture
def mock_collection() -> Mock:
    mock = Mock()
    mock.name = "test_collection"
    return mock

@pytest.fixture
def mock_client(mock_collection) -> Mock:
    mock = Mock()
    mock.create_collection.return_value = mock_collection
    mock.get_or_create_collection.return_value = mock_collection
    return mock

@pytest.fixture
def mock_db_location() -> Mock:
    mock = Mock()
    mock.db_location = "test_db_location"
    return mock

@pytest.fixture
def sample_csv_data() -> tuple[list[str], list[str], list[str]]:
    ids = ["1", "2", "3"]
    image_paths = ["sample/path/1", "sample/path/2", "sample/path/3"]
    descriptions = ["description1", "description2", "description3"]
    return ids, image_paths, descriptions


class TestVectorDBImageWriter:

    @pytest.fixture
    def mock_vector_db_image_writer_instance(self, mock_client: Mock, mock_collection: Mock, mock_db_location: Mock) -> VectorDBImageWriter:
        with patch('chromadb.PersistentClient', return_value=mock_client):
            vector_db_image_writer = VectorDBImageWriter(db_location=mock_db_location)

        vector_db_image_writer.data_loader = Mock(spec=ImageLoader)
        vector_db_image_writer.embedding_function = Mock(spec=OpenCLIPEmbeddingFunction)

        return vector_db_image_writer
        

    def test_create_new_collection_success(self, mock_client: Mock, mock_vector_db_image_writer_instance: VectorDBImageWriter, mock_collection: Mock):
        # Given
        collection_name = "test_collection"

        # When
        mock_vector_db_image_writer_instance.create_new_collection(collection_name)

        # Then
        mock_client.create_collection.assert_called_once_with(
            name=collection_name,
            embedding_function=mock_vector_db_image_writer_instance.embedding_function,
            data_loader=mock_vector_db_image_writer_instance.data_loader
        )

    
    def test_get_collection(self, mock_client: Mock, mock_vector_db_image_writer_instance: VectorDBImageWriter, mock_collection: Mock):
        # Given
        collection_name = "test_collection"

        # When
        result = mock_vector_db_image_writer_instance.get_collection(collection_name)

        # Then
        mock_client.get_or_create_collection.assert_called_once_with(
            name=collection_name,
            embedding_function=mock_vector_db_image_writer_instance.embedding_function,
            data_loader=mock_vector_db_image_writer_instance.data_loader
        )

        assert result == mock_collection

    
    def test_add_to_collection(self, mock_client: Mock, mock_vector_db_image_writer_instance: VectorDBImageWriter, mock_collection: Mock, sample_csv_data: tuple[list[str], list[str], list[str]]):
        # Given
        data_csv_path = Path("test_path/test.csv")
        collection_name = "test_collection"
        ids, image_paths, descriptions = sample_csv_data
        expected_metadata = [{"text": description} for description in descriptions]

        with patch("image_retrieval.create_image_vector_db.load_csv_data") as mock_load_csv_data:
            mock_load_csv_data.return_value = (ids, image_paths, descriptions)

            # When
            mock_vector_db_image_writer_instance.add_to_collection(data_csv_path, collection_name)

            # Then
            mock_load_csv_data.assert_called_once_with(data_csv_path)
            mock_collection.add.assert_called_once_with(
                ids=ids,
                uris=image_paths,
                metadatas=expected_metadata
            )


