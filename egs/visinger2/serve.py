import logging
logging.basicConfig(level=logging.WARNING)
import argparse
import io
import wave

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

import utils.conversion as conversion
from text import npu
from models import SynthesizerTrn
from inference import parse_label, load_model, inference_once


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-Requested-With", "Content-Type"],
)


@app.get("/")
async def test_connection_endpoint():
    return {"status": "okay"}


@app.post("/inference")
async def inference_endpoint(data: dict):
    global model, hps

    bpm, notes = data['bpm'], data['notes']
    notes = conversion.fill_gaps(notes)
    notes = list(conversion.convert_data_tick_to_second(notes, bpm, 480))
    notes = conversion.split_word_to_phoneme(notes)
    note_string = conversion.phoneme_items_to_string(notes)
    fileid, txt, *label = note_string.split('|')
    label = parse_label(hps, *label)
    audio_np = inference_once(model, label)
    audio_blob = io.BytesIO()
    with wave.open(audio_blob, 'wb') as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(44100)
        wave_file.writeframes(audio_np.tobytes())
    return Response(audio_blob.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dir', '--model_dir', type=str, required=True)
    args = parser.parse_args()
    model, hps = load_model(args.model_dir)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
