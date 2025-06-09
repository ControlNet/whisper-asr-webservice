import hashlib
import json
import sqlite3
import threading
from typing import Dict, Optional
from pathlib import Path



class Database:
    """SQLite database manager for job tracking and subtitle caching."""
    
    def __init__(self, db_path: str = "whisper_cache.db"):
        """Initialize database connection and create tables if they don't exist."""
        # Create database directory if it doesn't exist
        db_dir = Path(db_path).parent
        if db_dir != Path("."):
            db_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    @property
    def connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP NULL,
                    completed_at TIMESTAMP NULL,
                    audio_hash TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    result TEXT NULL,
                    error TEXT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subtitle_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_hash TEXT NOT NULL,
                    parameters_hash TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(audio_hash, parameters_hash)
                )
            """)
            
            # Create indices for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_hash ON subtitle_cache(audio_hash, parameters_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_accessed ON subtitle_cache(last_accessed)")
            
            conn.commit()
    
    def generate_audio_hash(self, audio_data: bytes) -> str:
        """Generate a hash for audio data to use as cache key."""
        return hashlib.sha256(audio_data).hexdigest()
    
    def generate_parameters_hash(self, parameters: Dict) -> str:
        """Generate a hash for transcription parameters."""
        # Sort parameters to ensure consistent hashing
        sorted_params = json.dumps(parameters, sort_keys=True)
        return hashlib.sha256(sorted_params.encode()).hexdigest()
    
    def create_job(self, job_id: str, audio_hash: str, parameters: Dict) -> None:
        """Create a new job entry."""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO jobs (id, status, audio_hash, parameters)
            VALUES (?, ?, ?, ?)
        """, (job_id, "pending", audio_hash, json.dumps(parameters)))
        self.connection.commit()
    
    def update_job_status(self, job_id: str, status: str, result: Optional[str] = None, error: Optional[str] = None) -> None:
        """Update job status and result."""
        cursor = self.connection.cursor()
        
        updates = ["status = ?"]
        values = [status]
        
        if status == "processing":
            updates.append("started_at = CURRENT_TIMESTAMP")
        elif status in ["completed", "failed"]:
            updates.append("completed_at = CURRENT_TIMESTAMP")
        
        if result is not None:
            updates.append("result = ?")
            values.append(result)
        
        if error is not None:
            updates.append("error = ?")
            values.append(error)
        
        values.append(job_id)
        
        cursor.execute(f"""
            UPDATE jobs 
            SET {', '.join(updates)}
            WHERE id = ?
        """, values)
        self.connection.commit()
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job information by ID."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                "id": row["id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "result": row["result"],
                "error": row["error"]
            }
        return None
    
    def get_cached_subtitle(self, audio_hash: str, parameters_hash: str) -> Optional[str]:
        """Get cached subtitle result."""
        cursor = self.connection.cursor()
        cursor.execute("""
            UPDATE subtitle_cache 
            SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
            WHERE audio_hash = ? AND parameters_hash = ?
        """, (audio_hash, parameters_hash))
        
        cursor.execute("""
            SELECT result FROM subtitle_cache 
            WHERE audio_hash = ? AND parameters_hash = ?
        """, (audio_hash, parameters_hash))
        
        row = cursor.fetchone()
        self.connection.commit()
        
        return row["result"] if row else None
    
    def cache_subtitle(self, audio_hash: str, parameters_hash: str, result: str) -> None:
        """Cache subtitle result."""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO subtitle_cache 
            (audio_hash, parameters_hash, result)
            VALUES (?, ?, ?)
        """, (audio_hash, parameters_hash, result))
        self.connection.commit()
    
    def cleanup_old_jobs(self, days_old: int = 7) -> int:
        """Clean up old completed/failed jobs."""
        cursor = self.connection.cursor()
        cursor.execute("""
            DELETE FROM jobs 
            WHERE status IN ('completed', 'failed') 
            AND datetime(created_at) < datetime('now', '-{} days')
        """.format(days_old))
        deleted = cursor.rowcount
        self.connection.commit()
        return deleted
    
    def cleanup_old_cache(self, days_old: int = 30) -> int:
        """Clean up old cache entries."""
        cursor = self.connection.cursor()
        cursor.execute("""
            DELETE FROM subtitle_cache 
            WHERE datetime(last_accessed) < datetime('now', '-{} days')
        """.format(days_old))
        deleted = cursor.rowcount
        self.connection.commit()
        return deleted
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) as count, SUM(access_count) as total_hits FROM subtitle_cache")
        row = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*) as pending FROM jobs WHERE status = 'pending'")
        pending_jobs = cursor.fetchone()["pending"]
        
        cursor.execute("SELECT COUNT(*) as processing FROM jobs WHERE status = 'processing'")
        processing_jobs = cursor.fetchone()["processing"]
        
        return {
            "cached_entries": row["count"],
            "total_cache_hits": row["total_hits"] or 0,
            "pending_jobs": pending_jobs,
            "processing_jobs": processing_jobs
        }
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()


# Global database instance
db = Database() 