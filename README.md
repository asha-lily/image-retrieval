Initial inspiration taken from this tutorial: https://www.youtube.com/watch?v=E4l91XKQSgw

# Image Retrieval

Create an image vector database and query it using text.

## Installation

This project uses `uv` for package management. 

#### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sudo sh
```

Virtual environment management using `uv`:

```bash
# create a virtual environment
uv venv

# activate the virtual environment
source .venv/bin/activate
```
#### Install dependencies

Dependencies are listed in `pyproject.toml`.

```bash
uv pip install -e .
```

## Image Database Run Modes

Note: the term `collection` refers to an image database.

There are 4 different ways to run `create_image_vector_db.py`, and each can be specified using the `--run_mode` arg at runtime.

1. Listing the names of existing collections
2. Viewing the contents of a specified collection
3. Creating a new, empty collection
4. Adding images to an collection

In modes 2-4, a `collection_name` is required; this is also a runtime arg.

See the following sections for instructions on how to use each mode.

### Listing the names of existing collections

```bash
python create_image_vector_db.py --run_mode list_collections
```

### Viewing the contents of a specified collection

```bash
python create_image_vector_db.py --run_mode view_collection_contents --collection_name <existing_collection_name>
```

### Creating a new, empty collection

```bash
python create_image_vector_db.py --run_mode create_new_collection --collection_name <new_collection_name>
```

### Adding images to an collection

Chromadb collections contain the following: 
- `IDs`: a list of unique integer IDs, one for each image in the collection
- `URIs`: a list of image paths, one pointing to each image
- `Metadatas`: a list of dictionaries, one for each image. Each dictionary contains a `text` key, where the corresponding value is a text label for the image

To run in this mode, you must have specified `data_csv_path` in `config.py`. `data_csv_path` should point to a CSV file where each row has the format `,id,image path,image description`; these columns map to `IDs`, `URIs` & `Metadatas` (explained above).

```bash
python create_image_vector_db.py --run_mode add_images_to_collection --collection_name <new_collection_name>
```


## Query the image database

Once you have a collection of images, you can query the collection for an image similar to your input text.

Running `query_image_db.py` prompts the user to `Describe the image you are looking for` in the command line.

```
python query_image_db.py
```

- The text input by the user is embedded using the same `OpenCLIPEmbeddingFunction` used to embed the images when adding them to the database. 
- The `n` image embeddings which are closest to the text embedding (by what distance metric???) are retrieved (where `n` is the `num_results_to_retrieve` parameter defined in `config.py`)
- The text labels associated with these images are returned 


