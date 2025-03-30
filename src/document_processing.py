import os
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Document parsers
import PyPDF2
import requests
from bs4 import BeautifulSoup
import re

# For embeddings and vector database
import torch
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json

class DocumentProcessor:
    def __init__(self, cache_dir: str = 'data/documents'):
        """
        Initializes the document processor with RAG (Retrieval-Augmented Generation) capabilities
        
        Args:
            cache_dir: Directory for storing processed documents and embeddings
        """
        # Create cache directories
        self.cache_dir = cache_dir
        self.raw_dir = os.path.join(cache_dir, 'raw')
        self.processed_dir = os.path.join(cache_dir, 'processed')
        self.index_dir = os.path.join(cache_dir, 'index')
        
        for directory in [self.cache_dir, self.raw_dir, self.processed_dir, self.index_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Embedding model (can be replaced with a different one)
        self.embedding_model = None
        self.embedding_dim = 384  # Default dimension for 'all-MiniLM-L6-v2'
        self.index = None
        self.document_lookup = {}
        
        # Index initialization
        self._initialize_or_load_index()
    
    def _initialize_embedding_model(self):
        """
        Initializes the model for generating embeddings
        """
        try:
            # Using a lightweight model for embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model has been initialized")
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            print("Embedding functionality will be unavailable")
    
    def _initialize_or_load_index(self):
        """
        Initializes or loads an existing FAISS index
        """
        index_path = os.path.join(self.index_dir, 'faiss_index.bin')
        lookup_path = os.path.join(self.index_dir, 'document_lookup.pkl')
        
        if os.path.exists(index_path) and os.path.exists(lookup_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(lookup_path, 'rb') as f:
                    self.document_lookup = pickle.load(f)
                print(f"Loaded existing index with {len(self.document_lookup)} documents")
            except Exception as e:
                print(f"Error loading index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """
        Creates a new FAISS index
        """
        try:
            if self.embedding_model is None:
                self._initialize_embedding_model()
            
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.document_lookup = {}
            print("Created new FAISS index")
        except Exception as e:
            print(f"Error creating new index: {e}")
            self.index = None
    
    def _save_index(self):
        """
        Saves the index and lookup to files
        """
        if self.index is None:
            print("No index to save")
            return
        
        try:
            index_path = os.path.join(self.index_dir, 'faiss_index.bin')
            lookup_path = os.path.join(self.index_dir, 'document_lookup.pkl')
            
            faiss.write_index(self.index, index_path)
            with open(lookup_path, 'wb') as f:
                pickle.dump(self.document_lookup, f)
            
            print(f"Saved index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def process_pdf(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Processes a PDF file and extracts text
        
        Args:
            file_path: Path to the PDF file
            metadata: Additional document metadata
            
        Returns:
            str: Document ID
        """
        try:
            # Generate document ID
            file_hash = self._get_file_hash(file_path)
            doc_id = f"pdf_{file_hash}"
            
            # Check if the document has already been processed
            processed_path = os.path.join(self.processed_dir, f"{doc_id}.json")
            if os.path.exists(processed_path):
                print(f"Document {file_path} has already been processed. Skipping.")
                return doc_id
            
            # Extract text
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            if not text:
                print(f"Failed to extract text from {file_path}")
                return None
            
            # Save processed document
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'source': file_path,
                'type': 'pdf',
                'processed_date': datetime.now().isoformat(),
                'page_count': len(reader.pages)
            })
            
            document_data = {
                'id': doc_id,
                'text': text,
                'metadata': metadata
            }
            
            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)
            
            # Add to index
            self._index_document(doc_id, text, metadata)
            
            return doc_id
            
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return None
    
    def process_html(self, url: str, metadata: Dict[str, Any] = None) -> str:
        """
        Retrieves and processes an HTML document from a URL
        
        Args:
            url: URL of the HTML document
            metadata: Additional document metadata
            
        Returns:
            str: Document ID
        """
        try:
            # Generate document ID
            url_hash = hashlib.md5(url.encode()).hexdigest()
            doc_id = f"html_{url_hash}"
            
            # Check if the document has already been processed
            processed_path = os.path.join(self.processed_dir, f"{doc_id}.json")
            if os.path.exists(processed_path):
                print(f"Document {url} has already been processed. Skipping.")
                return doc_id
            
            # Retrieve and process HTML
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unnecessary elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            
            # Extract text from main content
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean text
            text = re.sub(r'\n+', '\n', text)  # Remove multiple empty lines
            text = re.sub(r'\s+', ' ', text)   # Remove multiple spaces
            
            if not text:
                print(f"Failed to extract text from {url}")
                return None
            
            # Save processed document
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'source': url,
                'type': 'html',
                'processed_date': datetime.now().isoformat(),
                'title': soup.title.string if soup.title else "No title"
            })
            
            document_data = {
                'id': doc_id,
                'text': text,
                'metadata': metadata
            }
            
            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)
            
            # Add to index
            self._index_document(doc_id, text, metadata)
            
            return doc_id
            
        except Exception as e:
            print(f"Error processing HTML {url}: {e}")
            return None
    
    def process_text(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Processes text and adds it to the index
        
        Args:
            text: Text to process
            metadata: Document metadata (must contain 'source' or 'title')
            
        Returns:
            str: Document ID
        """
        try:
            # Check required metadata
            if 'source' not in metadata and 'title' not in metadata:
                raise ValueError("Metadata must contain 'source' or 'title'")
            
            # Generate document ID
            text_hash = hashlib.md5(text.encode()).hexdigest()
            doc_id = f"text_{text_hash}"
            
            # Check if document has already been processed
            processed_path = os.path.join(self.processed_dir, f"{doc_id}.json")
            if os.path.exists(processed_path):
                print(f"Document has already been processed. Skipping.")
                return doc_id
            
            # Save processed document
            metadata.update({
                'type': 'text',
                'processed_date': datetime.now().isoformat(),
                'char_count': len(text)
            })
            
            document_data = {
                'id': doc_id,
                'text': text,
                'metadata': metadata
            }
            
            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)
            
            # Add to index
            self._index_document(doc_id, text, metadata)
            
            return doc_id
            
        except Exception as e:
            print(f"Error processing text: {e}")
            return None
    
    def _segment_text(self, text: str, max_length: int = 1000, overlap: int = 200) -> List[Tuple[str, int]]:
        """
        Segments text into smaller chunks with overlap
        
        Args:
            text: Text to segment
            max_length: Maximum segment length
            overlap: Size of overlap between segments
            
        Returns:
            List[Tuple[str, int]]: List of pairs (segment, start_position)
        """
        segments = []
        text_length = len(text)
        
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Combine sentences into segments
        current_segment = ""
        current_position = 0
        start_position = 0
        
        for sentence in sentences:
            # If adding the sentence exceeds max_length, save segment and start a new one
            if len(current_segment) + len(sentence) > max_length and current_segment:
                segments.append((current_segment, start_position))
                
                # Go back by overlap characters
                overlap_text = current_segment[-overlap:] if len(current_segment) > overlap else current_segment
                current_position = start_position + len(current_segment) - len(overlap_text)
                start_position = current_position
                current_segment = overlap_text
            
            # Add sentence to segment
            current_segment += sentence + " "
            current_position += len(sentence) + 1
        
        # Add the last segment
        if current_segment:
            segments.append((current_segment, start_position))
        
        return segments
    
    def _index_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Indexes a document - generates embedding and adds to FAISS index
        
        Args:
            doc_id: Document ID
            text: Document text
            metadata: Document metadata
            
        Returns:
            bool: Whether indexing was successful
        """
        try:
            if self.embedding_model is None:
                self._initialize_embedding_model()
            
            if self.embedding_model is None:
                return False
            
            # Text segmentation
            segments = self._segment_text(text)
            
            # Generate embeddings for each segment
            for i, (segment_text, position) in enumerate(segments):
                # Generate embeddings
                embedding = self.embedding_model.encode([segment_text])[0]
                
                # Normalize embeddings (for cosine similarity)
                faiss.normalize_L2(embedding.reshape(1, -1))
                
                # Add to index
                self.index.add(embedding.reshape(1, -1))
                
                # Add to lookup
                segment_id = f"{doc_id}_seg{i}"
                self.document_lookup[self.index.ntotal - 1] = {
                    'id': segment_id,
                    'doc_id': doc_id,
                    'position': position,
                    'text': segment_text,
                    'metadata': metadata
                }
            
            # Save updated index
            self._save_index()
            return True
            
        except Exception as e:
            print(f"Error indexing document {doc_id}: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Searches for documents similar to the query
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_criteria: Result filtering criteria
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        try:
            if self.embedding_model is None:
                self._initialize_embedding_model()
            
            if self.embedding_model is None or self.index is None:
                return []
            
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Normalize embeddings (for cosine similarity)
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            # Search
            k = min(top_k * 3, self.index.ntotal)  # Get more to have extras after filtering
            if k == 0:
                return []
                
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            # Process results
            results = []
            seen_doc_ids = set()  # Avoid duplicates
            
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # -1 means no result
                    continue
                    
                # Get document data
                segment_data = self.document_lookup.get(int(idx))
                if segment_data is None:
                    continue
                
                # Filter results
                if filter_criteria:
                    skip = False
                    for key, value in filter_criteria.items():
                        if key in segment_data['metadata']:
                            if isinstance(value, list):
                                if segment_data['metadata'][key] not in value:
                                    skip = True
                                    break
                            elif segment_data['metadata'][key] != value:
                                skip = True
                                break
                    if skip:
                        continue
                
                # Add result (avoid document duplicates)
                doc_id = segment_data['doc_id']
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    
                    # Calculate score (convert distance to similarity)
                    score = 1.0 - float(distances[0][i])
                    
                    result = {
                        'doc_id': doc_id,
                        'score': score,
                        'segment_text': segment_data['text'],
                        'position': segment_data['position'],
                        'metadata': segment_data['metadata']
                    }
                    results.append(result)
                    
                    # If we have enough results, finish
                    if len(results) >= top_k:
                        break
            
            return results
                
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieves a full document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dict[str, Any]: Document data or None if not found
        """
        try:
            processed_path = os.path.join(self.processed_dir, f"{doc_id}.json")
            if not os.path.exists(processed_path):
                return None
                
            with open(processed_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def list_all_documents(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all documents in the database
        
        Returns:
            List[Dict[str, Any]]: List of document metadata
        """
        documents = []
        try:
            for filename in os.listdir(self.processed_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.processed_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                        # Return only metadata, without full text
                        documents.append({
                            'id': doc_data['id'],
                            'metadata': doc_data['metadata']
                        })
            return documents
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        Generates a file hash based on content
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: File hash (MD5)
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()