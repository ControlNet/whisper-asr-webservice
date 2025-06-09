import os
from os import path
from typing import Annotated, Optional, Union
from urllib.parse import quote

import click
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, applications, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio, download_audio_from_url

asr_model = ASRModelFactory.create_asr_model()
asr_model.load_model()

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

app = FastAPI(
    title="Whisper ASR Webservice",
    description="Whisper ASR Webservice",
    version="0.0.0",
    contact={"url": "https://github.com/ControlNet/whisper-asr-webservice"},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": "https://github.com/ControlNet/whisper-asr-webservice/blob/main/LICENSE"},
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
    
    # Handle audio input based on source
    if audio_url:
        try:
            # Download audio from URL
            downloaded_file = download_audio_from_url(audio_url)
            audio_data = load_audio(downloaded_file, encode)
            # Extract filename from URL for response headers
            filename = audio_url.split('/')[-1] or "downloaded_audio"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process audio URL: {str(e)}")
        finally:
            # Clean up temporary file if it exists
            if 'downloaded_file' in locals():
                try:
                    downloaded_file.close()
                    import os
                    os.unlink(downloaded_file.name)
                except:
                    pass
    else:
        # Handle uploaded file
        audio_data = load_audio(audio_file.file, encode)
        filename = audio_file.filename

    result = asr_model.transcribe(
        audio_data,
        task,
        language if language != "auto" else None,
        initial_prompt,
        vad_filter,
        word_timestamps,
        {"diarize": diarize, "min_speakers": min_speakers, "max_speakers": max_speakers},
        output,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(filename)}.{output}"',
        },
    )


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: Optional[UploadFile] = File(None, description="Audio file to analyze"),
    audio_url: Optional[str] = Query(None, description="URL to download audio file from"),
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
    
    # Handle audio input based on source
    if audio_url:
        try:
            # Download audio from URL
            downloaded_file = download_audio_from_url(audio_url)
            audio_data = load_audio(downloaded_file, encode)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process audio URL: {str(e)}")
        finally:
            # Clean up temporary file if it exists
            if 'downloaded_file' in locals():
                try:
                    downloaded_file.close()
                    import os
                    os.unlink(downloaded_file.name)
                except:
                    pass
    else:
        # Handle uploaded file
        audio_data = load_audio(audio_file.file, encode)

    detected_lang_code, confidence = asr_model.language_detection(audio_data)
    return {
        "detected_language": tokenizer.LANGUAGES[detected_lang_code],
        "language_code": detected_lang_code,
        "confidence": confidence,
    }


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
