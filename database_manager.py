import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Dict, Optional
import pickle
from datetime import datetime

class DatabaseManager:
    def __init__(self, host='localhost', database='facial_recognition', 
                 user='postgres', password='incorect'):
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
            # Create tables if they don't exist
            self._create_tables()
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise
            
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            # Create persons table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    employee_id VARCHAR(50) UNIQUE NOT NULL,
                    department VARCHAR(100),
                    authorized_zones TEXT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create face embeddings table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                    embedding BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create access logs table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_logs (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                    camera_id VARCHAR(50),
                    confidence FLOAT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create license plates table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS license_plates (
                    id SERIAL PRIMARY KEY,
                    plate_number VARCHAR(20) UNIQUE NOT NULL,
                    vehicle_type VARCHAR(50),
                    owner_name VARCHAR(255),
                    owner_id VARCHAR(50),
                    is_authorized BOOLEAN DEFAULT TRUE,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expiry_date TIMESTAMP,
                    notes TEXT
                )
            """)
            
            # Create vehicle access logs table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_access_logs (
                    id SERIAL PRIMARY KEY,
                    plate_number VARCHAR(20),
                    camera_id VARCHAR(50),
                    confidence FLOAT,
                    vehicle_type VARCHAR(50),
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT,
                    is_authorized BOOLEAN,
                    FOREIGN KEY (plate_number) REFERENCES license_plates(plate_number) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_plate_number ON license_plates(plate_number)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_access_time ON vehicle_access_logs(detected_at)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_camera ON vehicle_access_logs(camera_id)")
            
            self.conn.commit()
            print("Database tables created/verified")
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating tables: {e}")
            raise
            
    # Original person-related methods
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
            # Convert numpy float to Python float
            confidence = float(confidence)
            
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
            
    # License plate related methods
    def add_license_plate(self, plate_number: str, vehicle_type: str = None,
                         owner_name: str = None, owner_id: str = None,
                         is_authorized: bool = True, expiry_date: datetime = None,
                         notes: str = None) -> int:
        """Add a new license plate to the database"""
        try:
            query = """
                INSERT INTO license_plates 
                (plate_number, vehicle_type, owner_name, owner_id, is_authorized, expiry_date, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            self.cursor.execute(query, (plate_number, vehicle_type, owner_name, 
                                      owner_id, is_authorized, expiry_date, notes))
            self.conn.commit()
            return self.cursor.fetchone()['id']
        except psycopg2.IntegrityError:
            self.conn.rollback()
            print(f"License plate {plate_number} already exists")
            raise
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding license plate: {e}")
            raise
            
    def get_license_plate(self, plate_number: str) -> Optional[Dict]:
        """Get license plate information"""
        try:
            query = "SELECT * FROM license_plates WHERE plate_number = %s"
            self.cursor.execute(query, (plate_number,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error retrieving license plate: {e}")
            raise

    def lookup_owner_by_plate(self, plate_number: str) -> Optional[str]:
        """Get owner name by plate number"""
        try:
            plate_info = self.get_license_plate(plate_number)
            return plate_info['owner_name'] if plate_info else None
        except Exception as e:
            print(f"Error looking up owner for plate {plate_number}: {e}")
            return None
            
    def update_license_plate_authorization(self, plate_number: str, is_authorized: bool):
        """Update license plate authorization status"""
        try:
            query = """
                UPDATE license_plates 
                SET is_authorized = %s 
                WHERE plate_number = %s
            """
            self.cursor.execute(query, (is_authorized, plate_number))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating license plate authorization: {e}")
            raise
            
    def log_vehicle_access(self, plate_number: str, camera_id: str, 
                          confidence: float, vehicle_type: str = None,
                          image_path: str = None):
        """Log a vehicle access event"""
        try:
            # Check if plate exists and get authorization status
            plate_info = self.get_license_plate(plate_number)
            
            if plate_info:
                is_authorized = plate_info['is_authorized']
            else:
                # Unknown plate
                is_authorized = False
                # Optionally add unknown plate to database
                self.add_license_plate(plate_number, vehicle_type, is_authorized=False,
                                     notes="Auto-detected unknown plate")
                
            query = """
                INSERT INTO vehicle_access_logs 
                (plate_number, camera_id, confidence, vehicle_type, image_path, is_authorized)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.cursor.execute(query, (plate_number, camera_id, float(confidence), 
                                      vehicle_type, image_path, is_authorized))
            self.conn.commit()
            
            return is_authorized
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error logging vehicle access: {e}")
            raise
            
    def get_vehicle_access_logs(self, limit: int = 100, 
                               camera_id: str = None,
                               start_time: datetime = None,
                               end_time: datetime = None,
                               authorized_only: bool = None) -> List[Dict]:
        """Get vehicle access logs with filters"""
        try:
            query = """
                SELECT val.*, lp.owner_name, lp.owner_id 
                FROM vehicle_access_logs val
                LEFT JOIN license_plates lp ON val.plate_number = lp.plate_number
                WHERE 1=1
            """
            params = []
            
            if camera_id:
                query += " AND val.camera_id = %s"
                params.append(camera_id)
                
            if start_time:
                query += " AND val.detected_at >= %s"
                params.append(start_time)
                
            if end_time:
                query += " AND val.detected_at <= %s"
                params.append(end_time)
                
            if authorized_only is not None:
                query += " AND val.is_authorized = %s"
                params.append(authorized_only)
                
            query += " ORDER BY val.detected_at DESC LIMIT %s"
            params.append(limit)
            
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
            
        except Exception as e:
            print(f"Error retrieving vehicle access logs: {e}")
            raise
            
    def get_all_authorized_plates(self) -> List[str]:
        """Get all authorized license plates"""
        try:
            query = """
                SELECT plate_number 
                FROM license_plates 
                WHERE is_authorized = TRUE 
                AND (expiry_date IS NULL OR expiry_date > CURRENT_TIMESTAMP)
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return [r['plate_number'] for r in results]
        except Exception as e:
            print(f"Error retrieving authorized plates: {e}")
            raise
            
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {}
            
            # Person statistics
            self.cursor.execute("SELECT COUNT(*) as count FROM persons")
            stats['total_persons'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("SELECT COUNT(*) as count FROM face_embeddings")
            stats['total_face_embeddings'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("SELECT COUNT(*) as count FROM access_logs")
            stats['total_face_accesses'] = self.cursor.fetchone()['count']
            
            # Vehicle statistics
            self.cursor.execute("SELECT COUNT(*) as count FROM license_plates")
            stats['total_plates'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("SELECT COUNT(*) as count FROM license_plates WHERE is_authorized = TRUE")
            stats['authorized_plates'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("SELECT COUNT(*) as count FROM vehicle_access_logs")
            stats['total_vehicle_accesses'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("""
                SELECT COUNT(*) as count 
                FROM vehicle_access_logs 
                WHERE is_authorized = FALSE
            """)
            stats['unauthorized_vehicle_accesses'] = self.cursor.fetchone()['count']
            
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            raise