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
        
        # Handle audio input - for URLs, we'll download in background
        if audio_url:
            # For URLs, create a temporary hash based on URL and parameters
            # We'll download and process in the background job
            import hashlib
            url_content = f"{audio_url}_{encode}".encode()
            audio_hash = hashlib.sha256(url_content).hexdigest()
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
            # Generate hash for uploaded file
            audio_hash = db.generate_audio_hash(audio_data)
        
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
            "audio_url": audio_url  # Store URL for background downloading
        }
        
        parameters_hash = db.generate_parameters_hash(parameters)
        
        # Check cache first (only for uploaded files where we have audio data)
        if audio_data is not None:
            cached_result = db.get_cached_subtitle(audio_hash, parameters_hash)
            if cached_result:
                self.logger.info(f"Cache hit for job {job_id}")
                # Create job entry but mark as completed immediately
                db.create_job(job_id, audio_hash, parameters)
                db.update_job_status(job_id, "completed", cached_result)
                return job_id
        
        # Store audio data temporarily for processing (encode as base64 for JSON storage)
        if audio_data is not None:
            parameters["audio_data"] = base64.b64encode(audio_data).decode('utf-8')
        else:
            parameters["audio_data"] = None  # Will download in background
        parameters["audio_hash"] = audio_hash
        parameters["parameters_hash"] = parameters_hash
        
        # Create job entry
        db.create_job(job_id, audio_hash, parameters)
        
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
                    SELECT id, audio_hash, parameters 
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
            audio_hash = parameters.get("audio_hash")
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
                
                # Update audio hash with actual file content for proper caching
                actual_audio_hash = db.generate_audio_hash(audio_data)
                
                # Check cache again with actual audio hash
                cached_result = db.get_cached_subtitle(actual_audio_hash, parameters_hash)
                if cached_result:
                    self.logger.info(f"Cache hit after download for job {job_id}")
                    db.update_job_status(job_id, "completed", cached_result)
                    return cached_result
                
                # Update the audio hash for this job
                audio_hash = actual_audio_hash
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
            db.cache_subtitle(audio_hash, parameters_hash, result_text)
            
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