import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from opensearchpy import OpenSearch
import json
import os
import boto3
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv

load_dotenv()

class BookRecommendationProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = None
        self.collaborative_model = SVD()
        
    def connect_opensearch(self):
        host = os.getenv('OPENSEARCH_HOST', 'localhost')
        port = int(os.getenv('OPENSEARCH_PORT', '9200'))
        use_ssl = os.getenv('OPENSEARCH_USE_SSL', 'false').lower() == 'true'
        
        if use_ssl:
            # AWS managed OpenSearch
            region = os.getenv('AWS_REGION', 'us-east-2')
            credentials = boto3.Session().get_credentials()
            awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)
            
            self.client = OpenSearch(
                hosts=[{'host': host, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=None,
            )
        else:
            # Local OpenSearch
            self.client = OpenSearch(
                hosts=[{'host': host, 'port': port}],
                use_ssl=False,
                verify_certs=False,
            )
        
        try:
            info = self.client.info()
            print(f"Connected to OpenSearch: {info['version']['number']}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def create_index(self):
        index_settings = {
            "settings": {
                "index": {
                    "knn": True,
                }
            },
            "mappings": {
                "properties": {
                    "book_id": {"type": "integer"},
                    "title": {"type": "text"},
                    "author": {"type": "text"},
                    "isbn": {"type": "keyword"},
                    "description": {"type": "text"},
                    "genre": {"type": "keyword"},
                    "publication_year": {"type": "integer"},
                    "content_embedding": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "collaborative_features": {
                        "type": "knn_vector",
                        "dimension": 100,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    }
                }
            }
        }
        
        if self.client.indices.exists(index="books"):
            self.client.indices.delete(index="books")
        
        self.client.indices.create(index="books", body=index_settings)
        print("Created books index")
    
    def load_data(self):
        book_catalog = pd.read_csv('data/book_catalog.csv')
        rental_history = pd.read_csv('data/rental_history.csv')
        return book_catalog, rental_history
    
    def generate_content_embeddings(self, book_catalog):
        descriptions = book_catalog['description'].tolist()
        embeddings = self.model.encode(descriptions)
        return embeddings
    
    def generate_collaborative_embeddings(self, rental_history, book_catalog):
        reader = Reader(rating_scale=(1, 5))
        rental_history['rating'] = 4.0
        
        data = Dataset.load_from_df(
            rental_history[['user_id', 'book_id', 'rating']], 
            reader
        )
        
        trainset = data.build_full_trainset()
        self.collaborative_model.fit(trainset)
        
        book_factors = {}
        for book_id in book_catalog['book_id']:
            try:
                inner_id = trainset.to_inner_iid(book_id)
                book_factors[book_id] = self.collaborative_model.qi[inner_id]
            except ValueError:
                book_factors[book_id] = np.zeros(self.collaborative_model.n_factors)
        
        # Ensure no None values
        for book_id, factors in book_factors.items():
            if factors is None:
                book_factors[book_id] = np.zeros(self.collaborative_model.n_factors)
        
        return book_factors
    
    def index_books(self, book_catalog, content_embeddings, collaborative_factors):
        for idx, row in book_catalog.iterrows():
            collab_features = collaborative_factors.get(row['book_id'])
            if collab_features is None or not isinstance(collab_features, np.ndarray):
                print(f"Warning: Using default features for book_id {row['book_id']}")
                collab_features = np.zeros(self.collaborative_model.n_factors)
            
            # Ensure collab_features is not None and convert to list
            if collab_features is not None and hasattr(collab_features, 'tolist'):
                # Check for NaN values and replace them
                if np.any(np.isnan(collab_features)):
                    print(f"Warning: Found NaN values in collab_features for book_id {row['book_id']}")
                    collab_features = np.nan_to_num(collab_features, nan=0.0)
                collab_list = collab_features.tolist()
                
                # Final check to ensure no None values in list
                if any(x is None for x in collab_list):
                    print(f"Warning: Found None in collab_list for book_id {row['book_id']}")
                    collab_list = [0.0 if x is None else x for x in collab_list]
            else:
                print(f"Error: collab_features is {type(collab_features)} for book_id {row['book_id']}")
                collab_list = np.zeros(self.collaborative_model.n_factors).tolist()
            
            doc = {
                "book_id": int(row['book_id']),
                "title": row['title'],
                "author": row['author'],
                "isbn": row['isbn'],
                "description": row['description'],
                "genre": row['genre'],
                "publication_year": int(row['publication_year']),
                "content_embedding": content_embeddings[idx].tolist(),
                "collaborative_features": collab_list
            }
            
            self.client.index(index="books", id=row['book_id'], body=doc)
        
        print(f"Indexed {len(book_catalog)} books")
    
    def process_all(self):
        print("Starting data processing...")
        
        if not self.connect_opensearch():
            print("Failed to connect to OpenSearch")
            return
        
        print("Connected to OpenSearch")
        
        self.create_index()
        
        book_catalog, rental_history = self.load_data()
        print(f"Loaded {len(book_catalog)} books and {len(rental_history)} rental records")
        
        print("Generating content embeddings...")
        content_embeddings = self.generate_content_embeddings(book_catalog)
        
        print("Generating collaborative embeddings...")
        collaborative_factors = self.generate_collaborative_embeddings(rental_history, book_catalog)
        print(f"Sample collaborative factors: {list(collaborative_factors.items())[:3]}")
        
        print("Indexing books...")
        self.index_books(book_catalog, content_embeddings, collaborative_factors)
        
        print("Data processing complete!")

if __name__ == "__main__":
    processor = BookRecommendationProcessor()
    processor.process_all()