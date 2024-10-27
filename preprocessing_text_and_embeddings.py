
import os, boto3, requests
from langchain.text_splitter import TokenTextSplitter
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from bs4 import BeautifulSoup
from io import BytesIO


WEBPAGE_URL = os.getenv("WEBPAGE_URL")

# Extraer el texto desde la página web
def extract_text_from_webpage(url):
    """Extrae el texto desde una página web."""
    response = requests.get(url)
    response.raise_for_status()  # Lanza un error si la descarga falla

    # Usar BeautifulSoup para analizar el HTML
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text(separator="\n")  # Obtener el texto plano de la página

    return text

# Crear chunks del documento usando langchain
def split_text_into_chunks(file_content, chunk_size=100, chunk_overlap=10):
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(file_content)
    return chunks

# Extraer el texto desde la página web
text = extract_text_from_webpage(WEBPAGE_URL)
text_chunks = split_text_into_chunks(text)

# Guardar los chunks en un archivo
with open("text_chunks.txt", "w") as f:
    for chunk in text_chunks:
        f.write(chunk + "\n")

# Calcular Embeddings
embeddings_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Convertir los fragmentos en vectores
chunk_vectors = embeddings_model.encode(text_chunks)

# Insertar los datos en una base de datos vectorial
# Definir el número de dimensiones de los vectores
vector_dim = chunk_vectors.shape[1]

# Crear un índice Annoy
annoy_index = AnnoyIndex(vector_dim, "angular")

# Agregar vectores al índice
for index, vector in enumerate(chunk_vectors):
    annoy_index.add_item(index, vector)

# Construir el índice (definir el número de árboles, mayor número da mejor precisión, pero más lento)
n_trees = 10
annoy_index.build(n_trees)

# Guardar el índice en un archivo
annoy_index.save("chunk_vectors.ann")
