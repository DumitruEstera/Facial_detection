"""
DEBUG version of Database manager to show similarity scores
This will help us understand why recognition isn't working
"""

import os
import asyncio
import asyncpg
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import json
from dataclasses import dataclass
from pgvector.asyncpg import register_vector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PersonnelData:
    """Data class for personnel information"""
    id: Optional[int] = None
    personnel_id: str = ""
    first_name: str = ""
    last_name: str = ""
    rank: str = "STUDENT"
    unit: str = ""
    email: str = ""
    phone: str = ""
    is_active: bool = True
    group_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_seen: Optional[datetime] = None

@dataclass
class FaceEncodingData:
    """Data class for face encoding information"""
    id: Optional[int] = None
    personnel_id: int = 0
    encoding_vector: np.ndarray = None
    confidence_score: float = 0.0
    training_image_path: str = ""
    is_primary: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class SecurityAlert:
    """Data class for security alerts"""
    id: Optional[int] = None
    alert_type: str = ""
    severity: str = "MEDIUM"
    status: str = "ACTIVE"
    zone_id: Optional[int] = None
    personnel_id: Optional[int] = None
    description: str = ""
    additional_data: Dict[str, Any] = None
    image_path: str = ""
    detected_at: Optional[datetime] = None

class DatabaseManager:
    """Database manager for military security facial recognition system"""
    
    def __init__(self):
        self.pool = None
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'military_security'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Register pgvector types
            async with self.pool.acquire() as conn:
                await register_vector(conn)
            
            logger.info("✅ Database connection pool initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            return False
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def get_or_create_personnel(self, first_name: str, last_name: str, 
                                    personnel_id: str = None, rank: str = "STUDENT") -> int:
        """Get existing personnel or create new one"""
        try:
            async with self.pool.acquire() as conn:
                # Try to find existing personnel by name
                existing = await conn.fetchrow(
                    "SELECT id FROM personnel WHERE first_name = $1 AND last_name = $2",
                    first_name, last_name
                )
                
                if existing:
                    return existing['id']
                
                # Create new personnel
                if not personnel_id:
                    # Generate personnel ID based on name and timestamp
                    personnel_id = f"{first_name[:2].upper()}{last_name[:2].upper()}{int(datetime.now().timestamp())}"
                
                # Get default group ID (Students)
                group = await conn.fetchrow(
                    "SELECT id FROM personnel_groups WHERE name = 'Students'"
                )
                group_id = group['id'] if group else 1
                
                new_personnel_id = await conn.fetchval(
                    """
                    INSERT INTO personnel (personnel_id, first_name, last_name, rank, group_id)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    personnel_id, first_name, last_name, rank, group_id
                )
                
                logger.info(f"✅ Created new personnel: {first_name} {last_name} (ID: {new_personnel_id})")
                return new_personnel_id
                
        except Exception as e:
            logger.error(f"❌ Error getting/creating personnel: {e}")
            raise
    
    def _prepare_vector_for_db(self, vector: np.ndarray) -> List[float]:
        """Convert numpy array to list of floats for pgvector"""
        if isinstance(vector, np.ndarray):
            # Convert to list of Python floats
            return [float(x) for x in vector.flatten()]
        elif isinstance(vector, list):
            # If it's already a list, ensure all elements are floats
            return [float(x) for x in vector]
        else:
            raise ValueError(f"Unsupported vector type: {type(vector)}")
    
    async def save_face_encoding(self, personnel_id: int, encoding: np.ndarray, 
                               confidence_score: float = 0.0, 
                               image_path: str = "", is_primary: bool = True) -> bool:
        """Save face encoding to database"""
        try:
            async with self.pool.acquire() as conn:
                # Register vector type for this connection
                await register_vector(conn)
                
                # If this is set as primary, unset other primary encodings for this person
                if is_primary:
                    await conn.execute(
                        "UPDATE face_encodings SET is_primary = false WHERE personnel_id = $1",
                        personnel_id
                    )
                
                # Convert numpy array to list of floats
                vector_list = self._prepare_vector_for_db(encoding)
                
                # DEBUG: Print encoding info
                print(f"🔍 DEBUG: Saving encoding for personnel {personnel_id}")
                print(f"    Vector shape: {encoding.shape}")
                print(f"    Vector length: {len(vector_list)}")
                print(f"    Vector sample: {vector_list[:5]}...")
                
                # Insert new encoding - let pgvector handle the conversion automatically
                await conn.execute(
                    """
                    INSERT INTO face_encodings (personnel_id, encoding_vector, confidence_score, 
                                              training_image_path, is_primary)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    personnel_id, vector_list, confidence_score, image_path, is_primary
                )
                
                logger.info(f"✅ Saved face encoding for personnel ID: {personnel_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving face encoding: {e}")
            return False
    
    async def get_all_face_encodings(self) -> List[Tuple[int, str, np.ndarray]]:
        """Get all face encodings from database"""
        try:
            async with self.pool.acquire() as conn:
                # Register vector type for this connection
                await register_vector(conn)
                
                rows = await conn.fetch(
                    """
                    SELECT 
                        fe.personnel_id,
                        CONCAT(p.first_name, ' ', p.last_name) as full_name,
                        fe.encoding_vector
                    FROM face_encodings fe
                    JOIN personnel p ON fe.personnel_id = p.id
                    WHERE fe.is_primary = true AND p.is_active = true
                    """
                )
                
                encodings = []
                for row in rows:
                    # The encoding_vector comes back as a list from the database
                    encoding_list = row['encoding_vector']
                    if isinstance(encoding_list, list):
                        encoding_array = np.array(encoding_list, dtype=np.float32)
                    else:
                        # If it's already an array, use it as is
                        encoding_array = np.array(encoding_list, dtype=np.float32)
                    
                    # DEBUG: Print loaded encoding info
                    print(f"🔍 DEBUG: Loaded encoding for {row['full_name']}")
                    print(f"    Vector shape: {encoding_array.shape}")
                    print(f"    Vector sample: {encoding_array[:5]}")
                    
                    encodings.append((row['personnel_id'], row['full_name'], encoding_array))
                
                logger.info(f"✅ Retrieved {len(encodings)} face encodings from database")
                return encodings
                
        except Exception as e:
            logger.error(f"❌ Error retrieving face encodings: {e}")
            return []
    
    async def find_similar_faces(self, query_encoding: np.ndarray, 
                               threshold: float = 0.5, limit: int = 5) -> List[Dict]:  # LOWERED THRESHOLD
        """Find similar faces using vector similarity search"""
        try:
            async with self.pool.acquire() as conn:
                # Register vector type for this connection
                await register_vector(conn)
                
                # Convert query encoding to list of floats
                vector_list = self._prepare_vector_for_db(query_encoding)
                
                # DEBUG: Print query encoding info
                print(f"🔍 DEBUG: Searching for similar faces")
                print(f"    Query vector shape: {query_encoding.shape}")
                print(f"    Query vector sample: {vector_list[:5]}...")
                print(f"    Using threshold: {threshold}")
                
                # Use the vector list directly - pgvector will handle the conversion
                rows = await conn.fetch(
                    """
                    SELECT 
                        fe.personnel_id,
                        CONCAT(p.first_name, ' ', p.last_name) as full_name,
                        p.rank,
                        p.unit,
                        fe.encoding_vector <=> $1 as distance,
                        1 - (fe.encoding_vector <=> $1) as similarity
                    FROM face_encodings fe
                    JOIN personnel p ON fe.personnel_id = p.id
                    WHERE fe.is_primary = true AND p.is_active = true
                    ORDER BY fe.encoding_vector <=> $1
                    LIMIT $2
                    """,
                    vector_list, limit
                )
                
                results = []
                print(f"🔍 DEBUG: Found {len(rows)} potential matches:")
                
                for row in rows:
                    similarity = row['similarity']
                    distance = row['distance']
                    name = row['full_name']
                    
                    print(f"    {name}: similarity={similarity:.4f}, distance={distance:.4f}")
                    
                    if similarity >= threshold:
                        results.append({
                            'personnel_id': row['personnel_id'],
                            'name': row['full_name'],
                            'rank': row['rank'],
                            'unit': row['unit'],
                            'similarity': similarity,
                            'distance': row['distance']
                        })
                        print(f"      ✅ MATCH! Above threshold {threshold}")
                    else:
                        print(f"      ❌ Below threshold {threshold}")
                
                logger.info(f"✅ Found {len(results)} similar faces above threshold {threshold}")
                return results
                
        except Exception as e:
            logger.error(f"❌ Error finding similar faces: {e}")
            return []
    
    async def log_detection(self, zone_id: int, personnel_id: Optional[int], 
                          detection_type: str, confidence_score: float,
                          detection_data: Dict = None, image_path: str = "") -> bool:
        """Log a detection event"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO detection_logs (zone_id, personnel_id, detection_type, 
                                              confidence_score, detection_data, image_path)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    zone_id, personnel_id, detection_type, confidence_score,
                    json.dumps(detection_data) if detection_data else None, image_path
                )
                
                # Update last_seen for personnel
                if personnel_id:
                    await conn.execute(
                        "UPDATE personnel SET last_seen = CURRENT_TIMESTAMP WHERE id = $1",
                        personnel_id
                    )
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Error logging detection: {e}")
            return False
    
    async def create_security_alert(self, alert_type: str, severity: str, 
                                  description: str, zone_id: int = None,
                                  personnel_id: int = None, 
                                  additional_data: Dict = None,
                                  image_path: str = "") -> bool:
        """Create a security alert"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO security_alerts (alert_type, severity, description, zone_id,
                                               personnel_id, additional_data, image_path)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    alert_type, severity, description, zone_id, personnel_id,
                    json.dumps(additional_data) if additional_data else None, image_path
                )
                
                logger.info(f"✅ Created security alert: {alert_type} - {description}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error creating security alert: {e}")
            return False
    
    async def get_personnel_info(self, personnel_id: int) -> Optional[Dict]:
        """Get personnel information by ID"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT p.*, pg.name as group_name, pg.access_level
                    FROM personnel p
                    LEFT JOIN personnel_groups pg ON p.group_id = pg.id
                    WHERE p.id = $1
                    """,
                    personnel_id
                )
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting personnel info: {e}")
            return None
    
    async def check_zone_access(self, personnel_id: int, zone_id: int) -> bool:
        """Check if personnel has access to a specific zone"""
        try:
            async with self.pool.acquire() as conn:
                access = await conn.fetchval(
                    """
                    SELECT gza.access_granted
                    FROM personnel p
                    JOIN group_zone_access gza ON p.group_id = gza.group_id
                    WHERE p.id = $1 AND gza.zone_id = $2
                    """,
                    personnel_id, zone_id
                )
                
                return bool(access) if access is not None else False
                
        except Exception as e:
            logger.error(f"❌ Error checking zone access: {e}")
            return False
    
    async def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent security alerts"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT sa.*, cz.name as zone_name,
                           CONCAT(p.first_name, ' ', p.last_name) as personnel_name
                    FROM security_alerts sa
                    LEFT JOIN camera_zones cz ON sa.zone_id = cz.id
                    LEFT JOIN personnel p ON sa.personnel_id = p.id
                    ORDER BY sa.detected_at DESC
                    LIMIT $1
                    """,
                    limit
                )
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error getting recent alerts: {e}")
            return []
    
    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False