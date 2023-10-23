import sys
import argparse
import os
import io
import wave
import json
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch
import tqdm
import scipy
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

import utils.utils as utils
from text import npu
from models import SynthesizerTrn

job_running = False

def parse_label(hps, pho, pitchid, dur, slur, gtdur):
    phos = []
    pitchs = []
    durs = []
    slurs = []
    gtdurs = []

    for index in range(len(pho.split())):
        phos.append(npu.symbol_converter.ttsing_phone_to_int[pho.strip().split()[index]])
        pitchs.append(npu.symbol_converter.ttsing_opencpop_pitch_to_int[pitchid.strip().split()[index]])
        durs.append(float(dur.strip().split()[index]))
        slurs.append(int(slur.strip().split()[index]))
        gtdurs.append(float(gtdur.strip().split()[index]))

    phos = np.asarray(phos, dtype=np.int32)
    pitchs = np.asarray(pitchs, dtype=np.int32)
    durs = np.asarray(durs, dtype=np.float32)
    slurs = np.asarray(slurs, dtype=np.int32)
    gtdurs = np.asarray(gtdurs, dtype=np.float32)
    gtdurs = np.ceil(gtdurs / (hps.data.hop_size / hps.data.sample_rate))

    phos = torch.LongTensor(phos)
    pitchs = torch.LongTensor(pitchs)
    durs = torch.FloatTensor(durs)
    slurs = torch.LongTensor(slurs)
    gtdurs = torch.LongTensor(gtdurs)
    return phos, pitchs, durs, slurs, gtdurs


def load_model(model_dir):
    # load config and model
    model_path = utils.latest_checkpoint_path(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    
    hps = utils.get_hparams_from_file(config_path)

    print("Load model from : ", model_path)
    print("config: ", config_path)

    net_g = SynthesizerTrn(hps)
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)
    return net_g, hps


def inference_label2byte(net_g, input_data: str, hps, cuda_id=None):
    global job_running
    job_running = True
    try:
        id2label = {}
        for line in input_data.split("\n"):
            fileid, txt, phones, pitchid, dur, gtdur, slur = line.split('|')
            id2label[fileid] = [phones, pitchid, dur, slur, gtdur]
            
        wav_byte = {}
        for file_name in tqdm(id2label.keys()):
            pho, pitchid, dur, slur, gtdur = id2label[file_name]
            pho, pitchid, dur, slur, gtdur = parse_label(hps, pho, pitchid, dur, slur, gtdur)

            with torch.no_grad():

                # data
                pho_lengths = torch.LongTensor([pho.size(0)])
                pho = pho.unsqueeze(0)
                pitchid = pitchid.unsqueeze(0)
                dur = dur.unsqueeze(0)
                slur = slur.unsqueeze(0)

                if(cuda_id != None):
                    net_g = net_g.cuda(0)
                    pho = pho.cuda(0)
                    pho_lengths = pho_lengths.cuda(0)
                    pitchid = pitchid.cuda(0)
                    dur = dur.cuda(0)
                    slur = slur.cuda(0)

                # infer
                o, _, _ = net_g.infer(pho, pho_lengths, pitchid, dur, slur)
                audio = o[0,0].data.cpu().float().numpy()
                audio = audio * 32768 #hps.data.max_wav_value
                audio = audio.astype(np.int16)
                
                # debug
                scipy.io.wavfile.write(os.path.join('./', file_name.split('.')[0] + '.wav' ), hps.data.sample_rate, audio)

                # save
                bytes_buffer = io.BytesIO()
                with wave.open(bytes_buffer, 'wb') as wave_file:
                    wave_file.setnchannels(1)  # mono
                    wave_file.setsampwidth(2)  # 2 bytes or 16 bits
                    wave_file.setframerate(44100)  # Sample rate, e.g., 44100Hz
                    wave_file.setnframes(audio.size)
                    wave_file.writeframes(audio.tobytes())
                
                wav_byte[file_name] = bytes_buffer.getvalue()
        return wav_byte
    finally:
        job_running = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dir', '--model_dir', type=str, required=True)
    args = parser.parse_args()
    app = FastAPI()

    model_dir = args.model_dir

    model, hps = load_model(model_dir)
    print("load model end!")
    app = FastAPI()
    uvicorn.run(app, host="0.0.0.0", port=8000)


def fill_gaps(events, starting_point=0):
    ret = []
    if events[0].time > starting_point:
        ret.append({
            'time': starting_point,
            'duration': events[0].time - starting_point,
            'lyric': 'AP',
            'noteNumber': 'rest',
        })
        starting_point = events[0].time
    index = 0
    while index < len(events):
        if events[index].time > starting_point:
            ret.append({
                'time': starting_point,
                'duration': events[0].time - starting_point,
                'lyric': 'AP',
                'noteNumber': 'rest',
            })
        else:
            ret.append({
                'time': events[index].time,
                'duration': events[index].duration,
                'lyric': events[index].lyric,
                'noteNumber': events[index].noteNumber,
            })
        index += 1
    return ret


@app.get("/")
async def test_connection():
    return {"status": "okay"}

@app.websocket("/generate")
async def websocket_endpoint(websocket: WebSocket):
    global job_running
    await websocket.accept()
    while True:
        if job_running:
            break
        else:
            data: str = await websocket.receive_text()
            json.loads(data)
            
            with torch.no_grad():
                synth_result: bytes = inference_label2byte()
                inference_label2byte(model, data, hps, cuda_id=0)
            await websocket.send_text(synth_result)
