import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from elasticsearch import Elasticsearch
import json
import os
from dotenv import load_dotenv

load_dotenv()

class BookRecommendationProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.es = None
        self.collaborative_model = SVD()
        
    def connect_elasticsearch(self):
        es_host = os.getenv('ELASTICSEARCH_HOST', 'localhost:9200')
        es_username = os.getenv('ELASTICSEARCH_USERNAME')
        es_password = os.getenv('ELASTICSEARCH_PASSWORD')
        
        if es_username and es_password:
            self.es = Elasticsearch(
                [es_host],
                basic_auth=(es_username, es_password),
                verify_certs=True
            )
        else:
            self.es = Elasticsearch([es_host])
            
        return self.es.ping()
    
    def create_index(self):
        index_mapping = {
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
                        "type": "dense_vector",
                        "dims": 384
                    },
                    "collaborative_features": {
                        "type": "dense_vector",
                        "dims": 50
                    }
                }
            }
        }
        
        if self.es.indices.exists(index="books"):
            self.es.indices.delete(index="books")
        
        self.es.indices.create(index="books", body=index_mapping)
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
        
        return book_factors
    
    def index_books(self, book_catalog, content_embeddings, collaborative_factors):
        for idx, row in book_catalog.iterrows():
            doc = {
                "book_id": int(row['book_id']),
                "title": row['title'],
                "author": row['author'],
                "isbn": row['isbn'],
                "description": row['description'],
                "genre": row['genre'],
                "publication_year": int(row['publication_year']),
                "content_embedding": content_embeddings[idx].tolist(),
                "collaborative_features": collaborative_factors[row['book_id']].tolist()
            }
            
            self.es.index(index="books", id=row['book_id'], body=doc)
        
        print(f"Indexed {len(book_catalog)} books")
    
    def process_all(self):
        print("Starting data processing...")
        
        if not self.connect_elasticsearch():
            print("Failed to connect to Elasticsearch")
            return
        
        print("Connected to Elasticsearch")
        
        self.create_index()
        
        book_catalog, rental_history = self.load_data()
        print(f"Loaded {len(book_catalog)} books and {len(rental_history)} rental records")
        
        print("Generating content embeddings...")
        content_embeddings = self.generate_content_embeddings(book_catalog)
        
        print("Generating collaborative embeddings...")
        collaborative_factors = self.generate_collaborative_embeddings(rental_history, book_catalog)
        
        print("Indexing books...")
        self.index_books(book_catalog, content_embeddings, collaborative_factors)
        
        print("Data processing complete!")

if __name__ == "__main__":
    processor = BookRecommendationProcessor()
    processor.process_all()