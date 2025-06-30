import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Dict, Optional
import pickle

class DatabaseManager:
    def __init__(self, host='localhost', database='facial_recognition_db', 
                 user='postgres', password='your_password'):
        self.connection_params = {
            'host': host,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            print("Database connection established")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise
            
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            
    def add_person(self, name: str, employee_id: str, department: str = None, 
                   authorized_zones: List[str] = None) -> int:
        """Add a new person to the database"""
        try:
            query = """
                INSERT INTO persons (name, employee_id, department, authorized_zones)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """
            self.cursor.execute(query, (name, employee_id, department, authorized_zones))
            self.conn.commit()
            return self.cursor.fetchone()['id']
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding person: {e}")
            raise
            
    def add_face_embedding(self, person_id: int, embedding: np.ndarray):
        """Store face embedding for a person"""
        try:
            # Serialize the numpy array
            embedding_bytes = pickle.dumps(embedding)
            
            query = """
                INSERT INTO face_embeddings (person_id, embedding)
                VALUES (%s, %s)
            """
            self.cursor.execute(query, (person_id, psycopg2.Binary(embedding_bytes)))
            self.conn.commit()
            print(f"Embedding added for person_id: {person_id}")
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding face embedding: {e}")
            raise
            
    def get_all_embeddings(self) -> List[Dict]:
        """Retrieve all face embeddings with person information"""
        try:
            query = """
                SELECT fe.id, fe.person_id, fe.embedding, p.name, p.employee_id
                FROM face_embeddings fe
                JOIN persons p ON fe.person_id = p.id
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            # Deserialize embeddings
            for result in results:
                result['embedding'] = pickle.loads(result['embedding'])
                
            return results
        except Exception as e:
            print(f"Error retrieving embeddings: {e}")
            raise
            
    def log_access(self, person_id: int, camera_id: str, confidence: float):
        """Log an access event"""
        try:
            query = """
                INSERT INTO access_logs (person_id, camera_id, confidence)
                VALUES (%s, %s, %s)
            """
            self.cursor.execute(query, (person_id, camera_id, confidence))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error logging access: {e}")
            raise
            
    def get_person_by_id(self, person_id: int) -> Optional[Dict]:
        """Get person information by ID"""
        try:
            query = "SELECT * FROM persons WHERE id = %s"
            self.cursor.execute(query, (person_id,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error retrieving person: {e}")
            raise