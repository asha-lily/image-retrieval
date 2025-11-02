Initial inspiration taken from this tutorial: https://www.youtube.com/watch?v=E4l91XKQSgw

# Image Retrieval

Create an image vector database and query it using text.

### Installation

This project uses `uv` for package management. 

#### Install uv

```
curl -LsSf https://astral.sh/uv/install.sh | sudo sh
```

Virtual environment management using `uv`:

```
# create a virtual environment
uv venv

# activate the virtual environment
source .venv/bin/activate
```
#### Install dependencies

Dependencies are listed in `pyproject.toml`.

```
uv pip install -e .
```

### Add images to a local vector database

TO DO

Explain the following steps:

- add images to `data`
- create df containing image paths & text, and write to csv
- config flag to select mode: (a) list collections (b) view collection contents or (c) add to a collection.

```
python create_image_vector_db.py
```

### Query the image database

Running this script prompts the user to `Describe the image you are looking for` in the command line.

- The text input by the user is embedded using the same `OpenCLIPEmbeddingFunction` used to embed the images when adding them to the database. 
- The `n` image embeddings which are closest to the text embedding (by what distance metric???) are retrieved (where `n` is the `num_results_to_retrieve` parameter defined in `config.py`)
- The text labels associated with these images are returned 


```
python query_image_db.py
```

