import os
from os import path
from typing import Annotated, Optional, Union
from urllib.parse import quote
from contextlib import asynccontextmanager

import click
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, applications, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer

from app.config import CONFIG
from app.job_manager import job_manager
from app.database import db

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())


# Use lifespan events instead of deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    job_manager.shutdown()
    db.close()

async def wait_for_job_completion(job_id: str, max_wait_time: int = 300) -> str:
    """Asynchronously wait for job completion and return result."""
    import asyncio
    
    start_time = asyncio.get_event_loop().time()
    
    while (asyncio.get_event_loop().time() - start_time) < max_wait_time:
        job_status = job_manager.get_job_status(job_id)
        
        if job_status["status"] == "completed":
            return job_status["result"]
        elif job_status["status"] == "failed":
            raise RuntimeError(f"Job failed: {job_status.get('error', 'Unknown error')}")
        
        # Non-blocking async sleep
        await asyncio.sleep(0.5)
    
    raise TimeoutError("Job processing timed out")


app = FastAPI(
    title="Whisper ASR Webservice",
    description="Whisper ASR Webservice",
    version="0.0.0",
    contact={"url": "https://github.com/ControlNet/whisper-asr-webservice"},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": "https://github.com/ControlNet/whisper-asr-webservice/blob/main/LICENSE"},
    lifespan=lifespan,
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/asr", tags=["Endpoints"])
async def asr(
    audio_file: Optional[UploadFile] = File(None, description="Audio file to transcribe"),
    audio_url: Optional[str] = Query(None, description="URL to download audio file from"),
    audio_id: Optional[str] = Query(None, description="Optional audio ID for caching (overrides URL/file hash)"),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False),
    ),
    min_speakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    # Validate that either audio_file or audio_url is provided, but not both
    if not audio_file and not audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Either audio_file or audio_url must be provided"
        )
    
    if audio_file and audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Cannot provide both audio_file and audio_url. Choose one."
        )
    
    try:
        # Submit job to job manager
        job_id = job_manager.submit_job(
            audio_file=audio_file.file if audio_file else None,
            audio_url=audio_url,
            audio_id=audio_id,
            filename=audio_file.filename if audio_file else None,
            encode=encode,
            task=task,
            language=language if language != "auto" else None,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            diarize_options={"diarize": diarize, "min_speakers": min_speakers, "max_speakers": max_speakers},
            output=output,
            job_type="transcribe"
        )
        
        # Wait for completion asynchronously
        result = await wait_for_job_completion(job_id)
        
        # Extract filename for response headers
        if audio_url:
            filename = audio_url.split('/')[-1] or "downloaded_audio"
        else:
            filename = audio_file.filename
        
        # Return result as streaming response
        def generate():
            yield result
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Asr-Engine": CONFIG.ASR_ENGINE,
                "Content-Disposition": f'attachment; filename="{quote(filename)}.{output}"',
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: Optional[UploadFile] = File(None, description="Audio file to analyze"),
    audio_url: Optional[str] = Query(None, description="URL to download audio file from"),
    audio_id: Optional[str] = Query(None, description="Optional audio ID for caching (overrides URL/file hash)"),
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    # Validate that either audio_file or audio_url is provided, but not both
    if not audio_file and not audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Either audio_file or audio_url must be provided"
        )
    
    if audio_file and audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Cannot provide both audio_file and audio_url. Choose one."
        )
    
    try:
        # Submit language detection job to job manager
        job_id = job_manager.submit_job(
            audio_file=audio_file.file if audio_file else None,
            audio_url=audio_url,
            audio_id=audio_id,
            encode=encode,
            job_type="detect_language"
        )
        
        # Wait for completion asynchronously
        result = await wait_for_job_completion(job_id)
        
        # Parse and return the JSON result
        import json
        return json.loads(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")


@app.post("/submit", tags=["Async Endpoints"])
async def submit_job(
    audio_file: Optional[UploadFile] = File(None, description="Audio file to transcribe"),
    audio_url: Optional[str] = Query(None, description="URL to download audio file from"),
    audio_id: Optional[str] = Query(None, description="Optional audio ID for caching (overrides URL/file hash)"),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False),
    ),
    min_speakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    """Submit an ASR job for background processing. Returns a job ID that can be used to check status and retrieve results."""
    # Validate that either audio_file or audio_url is provided, but not both
    if not audio_file and not audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Either audio_file or audio_url must be provided"
        )
    
    if audio_file and audio_url:
        raise HTTPException(
            status_code=400, 
            detail="Cannot provide both audio_file and audio_url. Choose one."
        )
    
    try:
        # Submit job to job manager
        job_id = job_manager.submit_job(
            audio_file=audio_file.file if audio_file else None,
            audio_url=audio_url,
            audio_id=audio_id,
            filename=audio_file.filename if audio_file else None,
            encode=encode,
            task=task,
            language=language if language != "auto" else None,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            diarize_options={"diarize": diarize, "min_speakers": min_speakers, "max_speakers": max_speakers},
            output=output
        )
        
        # Check if job was completed immediately (cache hit)
        job_status = job_manager.get_job_status(job_id)
        if job_status and job_status["status"] == "completed":
            # Cache hit - return result directly
            return JSONResponse(
                content={
                    "job_id": job_id,
                    "status": "completed",
                    "result": job_status["result"],
                    "created_at": job_status["created_at"],
                    "started_at": job_status["started_at"], 
                    "completed_at": job_status["completed_at"],
                    "message": "Cache hit - result returned immediately",
                    "cached": True
                },
                status_code=200
            )
        else:
            # Job queued for processing
            return JSONResponse(
                content={
                    "job_id": job_id,
                    "status": "submitted",
                    "created_at": job_status["created_at"] if job_status else None,
                    "started_at": job_status["started_at"] if job_status else None,
                    "completed_at": job_status["completed_at"] if job_status else None,
                    "message": "Job submitted successfully. Use the /get endpoint to check status and retrieve results.",
                    "cached": False
                },
                status_code=202
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@app.get("/get/{job_id}", tags=["Async Endpoints"])
async def get_job_result(job_id: str):
    """Get the status and result of a submitted job."""
    job = job_manager.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response_data = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"]
    }
    
    if job["status"] == "completed":
        response_data["result"] = job["result"]
        return JSONResponse(content=response_data, status_code=200)
    elif job["status"] == "failed":
        response_data["error"] = job["error"]
        return JSONResponse(content=response_data, status_code=200)
    elif job["status"] == "processing":
        response_data["message"] = "Job is currently being processed"
        return JSONResponse(content=response_data, status_code=202)
    else:  # pending
        response_data["message"] = "Job is queued for processing"
        return JSONResponse(content=response_data, status_code=202)


@app.get("/stats", tags=["Admin"])
async def get_system_stats():
    """Get system statistics including cache hits, job queue status, etc."""
    stats = job_manager.get_stats()
    return JSONResponse(content=stats)


@app.post("/admin/cleanup", tags=["Admin"])
async def cleanup_old_data(
    job_days: int = Query(default=7, description="Delete jobs older than this many days"),
    cache_days: int = Query(default=30, description="Delete cache entries older than this many days")
):
    """Clean up old jobs and cache entries."""
    result = job_manager.cleanup_old_data(job_days, cache_days)
    return JSONResponse(content={
        "message": "Cleanup completed",
        **result
    })


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    help="Host for the webservice (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    default=9000,
    help="Port for the webservice (default: 9000)",
)
@click.version_option(version="0.0.0")
def start(host: str, port: Optional[int] = None):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
