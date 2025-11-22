import pytest
from unittest.mock import Mock, MagicMock, patch

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from image_retrieval.create_image_vector_db import VectorDBImageWriter


@pytest.fixture
def mock_collection():
    mock = Mock()
    mock.name = "test_collection"
    return mock

@pytest.fixture
def mock_client(mock_collection):
    mock = Mock()
    mock.create_collection.return_value = mock_collection
    # can add same for 'get_collection' & 'list_collections'
    return mock

@pytest.fixture
def mock_db_location():
    mock = Mock()
    mock.db_location = "test_db_location"
    return mock


class TestVectorDBImageWriter:

    @pytest.fixture
    def mock_vector_db_image_writer_instance(self, mock_client, mock_collection, mock_db_location):
        with patch('chromadb.PersistentClient', return_value=mock_client):
            vector_db_image_writer = VectorDBImageWriter(db_location=mock_db_location)

        vector_db_image_writer.data_loader = Mock(spec=ImageLoader)
        vector_db_image_writer.embedding_function = Mock(spec=OpenCLIPEmbeddingFunction)

        return vector_db_image_writer
        

    def test_create_new_collection_success(
        self, 
        mock_client,
        mock_vector_db_image_writer_instance, 
        mock_collection):

        # Given
        collection_name = "test_collection"

        # When
        result = mock_vector_db_image_writer_instance.create_new_collection(collection_name)

        # Then
        mock_client.create_collection.assert_called_once_with(
            name=collection_name,
            embedding_function=mock_vector_db_image_writer_instance.embedding_function,
            data_loader=mock_vector_db_image_writer_instance.data_loader
        )