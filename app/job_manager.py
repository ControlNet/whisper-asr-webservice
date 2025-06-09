import asyncio
import base64
import io
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, BinaryIO
import logging
from contextlib import contextmanager

from app.database import db
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio, download_audio_from_url


class JobManager:
    """Manages background job processing for audio transcription."""
    
    def __init__(self, max_workers: int = 2):
        """Initialize job manager with thread pool."""
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.asr_model = ASRModelFactory.create_asr_model()
        self.asr_model.load_model()
        self._running = True
        
        # Start background job processor
        self._processor_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self._processor_thread.start()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def submit_job(
        self,
        audio_file: Optional[BinaryIO] = None,
        audio_url: Optional[str] = None,
        audio_id: Optional[str] = None,
        filename: Optional[str] = None,
        encode: bool = True,
        task: str = "transcribe",
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        vad_filter: bool = False,
        word_timestamps: bool = False,
        diarize_options: Dict = None,
        output: str = "txt",
        job_type: str = "transcribe"  # "transcribe" or "detect_language"
    ) -> str:
        """Submit a job for processing and return job ID."""
        job_id = str(uuid.uuid4())
        
        # Generate cache key based on priority: audio_id > URL > file hash
        if audio_id:
            # Priority 1: Use provided audio_id
            cache_key = audio_id
            print(f"🔑 DEBUG Using audio_id as cache key: {audio_id}")
        elif audio_url:
            # Priority 2: Use URL + encode for cache key
            cache_key = f"{audio_url}_{encode}"
            print(f"🔑 DEBUG Using URL as cache key: {cache_key}")
        else:
            # Priority 3: Will use file hash (calculated below)
            cache_key = None
            print(f"🔑 DEBUG Will use file hash as cache key")
        
        # Handle audio input - for URLs, we'll download in background
        if audio_url:
            audio_data = None
            if not filename:
                ext = audio_url.split('.')[-1].split("?")[0]
                filename = f"{job_id}.{ext}"
        else:
            # For uploaded files, read immediately
            audio_data = audio_file.read()
            audio_file.seek(0)  # Reset file pointer
            if not filename:
                filename = getattr(audio_file, 'filename', 'uploaded_audio')
            # For uploaded files without audio_id, use file hash as cache key
            if not cache_key:
                cache_key = db.generate_audio_hash(audio_data)
                print(f"🔑 DEBUG Generated file hash as cache key: {cache_key}")
        
        # Prepare parameters for processing
        parameters = {
            "job_type": job_type,
            "task": task,
            "language": language,
            "initial_prompt": initial_prompt,
            "vad_filter": vad_filter,
            "word_timestamps": word_timestamps,
            "diarize_options": diarize_options or {},
            "output": output,
            "encode": encode,
            "filename": filename,
            "audio_url": audio_url,  # Store URL for background downloading
            "audio_id": audio_id  # Store audio_id for reference
        }
        
        # Create cache parameters (exclude non-essential params that shouldn't affect caching)
        cache_parameters = {
            "job_type": job_type,
            "task": task,
            "language": language,
            "initial_prompt": initial_prompt,
            "vad_filter": vad_filter,
            "word_timestamps": word_timestamps,
            "diarize_options": diarize_options or {},
            "output": output,
            "encode": encode
            # Exclude: filename, audio_url, audio_id - these don't affect processing results
        }
        
        parameters_hash = db.generate_parameters_hash(cache_parameters)
        
        # Debug logging
        print(f"🔍 DEBUG Job {job_id} - Cache key: {cache_key}")
        print(f"🔍 DEBUG Job {job_id} - Parameters hash: {parameters_hash}")
        print(f"🔍 DEBUG Job {job_id} - Cache parameters: {cache_parameters}")
        
        # Show current cache entries for debugging
        cursor = db.connection.cursor()
        cursor.execute("SELECT cache_key, parameters_hash FROM subtitle_cache WHERE cache_key = ?", (cache_key,))
        existing_entries = cursor.fetchall()
        if existing_entries:
            print(f"📋 DEBUG Existing cache entries for key '{cache_key}': {[(row[0], row[1]) for row in existing_entries]}")
        else:
            print(f"📋 DEBUG No existing cache entries for key '{cache_key}'")
        
        # Check cache first with the determined cache key
        cached_result = db.get_cached_subtitle(cache_key, parameters_hash)
        if cached_result:
            print(f"✅ DEBUG Cache HIT for job {job_id} with cache key: {cache_key}")
            # Create job entry but mark as completed immediately
            db.create_job(job_id, cache_key, parameters)
            db.update_job_status(job_id, "completed", cached_result)
            return job_id
        else:
            print(f"❌ DEBUG Cache MISS for job {job_id} with cache key: {cache_key}")
            print(f"🔄 DEBUG Will process and cache result...")
        
        # Store audio data temporarily for processing (encode as base64 for JSON storage)
        if audio_data is not None:
            parameters["audio_data"] = base64.b64encode(audio_data).decode('utf-8')
        else:
            parameters["audio_data"] = None  # Will download in background
        parameters["cache_key"] = cache_key
        parameters["parameters_hash"] = parameters_hash
        
        # Create job entry
        db.create_job(job_id, cache_key, parameters)
        
        self.logger.info(f"Job {job_id} submitted for processing")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status and result."""
        return db.get_job(job_id)
    
    def _process_jobs(self):
        """Background thread to process pending jobs."""
        while self._running:
            try:
                # Get pending jobs
                cursor = db.connection.cursor()
                cursor.execute("""
                    SELECT id, cache_key, parameters 
                    FROM jobs 
                    WHERE status = 'pending' 
                    ORDER BY created_at ASC 
                    LIMIT 1
                """)
                row = cursor.fetchone()
                
                if row:
                    job_id = row["id"]
                    parameters = json.loads(row["parameters"])
                    
                    self.logger.info(f"Processing job {job_id}")
                    
                    # Submit to thread pool
                    future = self.executor.submit(self._process_single_job, job_id, parameters)
                    
                    # Don't wait for completion, let it run in background
                    threading.Thread(
                        target=self._handle_job_completion,
                        args=(job_id, future),
                        daemon=True
                    ).start()
                
                # Sleep before checking for more jobs
                threading.Event().wait(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in job processor: {e}")
                threading.Event().wait(5.0)
    
    def _handle_job_completion(self, job_id: str, future):
        """Handle job completion in a separate thread."""
        try:
            result = future.result()  # This will block until job completes
            # Result is already handled in _process_single_job
        except Exception as e:
            self.logger.error(f"Job {job_id} failed with error: {e}")
            db.update_job_status(job_id, "failed", error=str(e))
    
    def _process_single_job(self, job_id: str, parameters: Dict) -> str:
        """Process a single job."""
        try:
            # Update status to processing
            db.update_job_status(job_id, "processing")
            
            # Extract parameters
            audio_data_b64 = parameters.get("audio_data")
            audio_url = parameters.get("audio_url")
            audio_id = parameters.get("audio_id")
            cache_key = parameters.get("cache_key")
            parameters_hash = parameters.get("parameters_hash")
            encode = parameters.get("encode", True)
            
            # Handle audio input - either from uploaded data or URL
            if audio_data_b64:
                # Decode audio data from base64 (uploaded file)
                audio_data = base64.b64decode(audio_data_b64)
                audio_file_obj = io.BytesIO(audio_data)
            elif audio_url:
                # Download from URL in background
                self.logger.info(f"Downloading audio from URL for job {job_id}: {audio_url}")
                downloaded_file = download_audio_from_url(audio_url)
                audio_data = downloaded_file.read()
                downloaded_file.close()
                # Clean up temporary file
                import os
                os.unlink(downloaded_file.name)
                
                # If no audio_id was provided, update cache key with actual file hash for better caching
                if not audio_id:
                    actual_file_hash = db.generate_audio_hash(audio_data)
                    # Check cache again with actual file hash
                    cached_result = db.get_cached_subtitle(actual_file_hash, parameters_hash)
                    if cached_result:
                        self.logger.info(f"Cache hit after download for job {job_id} with file hash")
                        db.update_job_status(job_id, "completed", cached_result)
                        return cached_result
                    # Update cache key to file hash for this session
                    cache_key = actual_file_hash
                
                audio_file_obj = io.BytesIO(audio_data)
            else:
                raise ValueError("No audio data or URL found in job parameters")
            
            # Load and process audio
            processed_audio = load_audio(audio_file_obj, encode)
            
            job_type = parameters.get("job_type", "transcribe")
            
            if job_type == "detect_language":
                # Language detection
                detected_lang_code, confidence = self.asr_model.language_detection(processed_audio)
                from whisper import tokenizer
                result_data = {
                    "detected_language": tokenizer.LANGUAGES[detected_lang_code],
                    "language_code": detected_lang_code,
                    "confidence": confidence,
                }
                result_text = json.dumps(result_data)
            else:
                # Transcribe audio
                result = self.asr_model.transcribe(
                    processed_audio,
                    parameters.get("task", "transcribe"),
                    parameters.get("language"),
                    parameters.get("initial_prompt"),
                    parameters.get("vad_filter", False),
                    parameters.get("word_timestamps", False),
                    parameters.get("diarize_options", {}),
                    parameters.get("output", "txt")
                )
                
                # Convert result to string
                result_text = ""
                for chunk in result:
                    result_text += chunk
            
            # Cache the result
            print(f"💾 DEBUG Caching result for job {job_id} with cache key: {cache_key}, params hash: {parameters_hash}")
            db.cache_subtitle(cache_key, parameters_hash, result_text)
            
            # Update job status
            db.update_job_status(job_id, "completed", result_text)
            
            self.logger.info(f"Job {job_id} completed successfully")
            return result_text
            
        except Exception as e:
            self.logger.error(f"Error processing job {job_id}: {e}")
            db.update_job_status(job_id, "failed", error=str(e))
            raise
    
    def get_stats(self) -> Dict:
        """Get job manager statistics."""
        cache_stats = db.get_cache_stats()
        
        return {
            **cache_stats,
            "max_workers": self.max_workers,
            "active_workers": len([f for f in self.executor._threads if f.is_alive()]) if hasattr(self.executor, '_threads') else 0
        }
    
    def cleanup_old_data(self, job_days: int = 7, cache_days: int = 30) -> Dict:
        """Clean up old jobs and cache entries."""
        deleted_jobs = db.cleanup_old_jobs(job_days)
        deleted_cache = db.cleanup_old_cache(cache_days)
        
        return {
            "deleted_jobs": deleted_jobs,
            "deleted_cache_entries": deleted_cache
        }
    
    def shutdown(self):
        """Shutdown the job manager."""
        self._running = False
        self.executor.shutdown(wait=True)


# Global job manager instance
job_manager = JobManager() 