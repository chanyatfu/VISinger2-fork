import matplotlib.pyplot as plt
import IPython.display as ipd

import sys
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import modules.commons as commons
import utils.utils as utils
from models import SynthesizerTrn
from text import npu
from scipy.io.wavfile import write
from tqdm import tqdm
import numpy as np
import time
import argparse

def parse_file(label_list_path):
    """
    Returns: { filename: [phones, pitchid, dur, slur, gtdur] }
    """
    id2label = {}
    with open(label_list_path, "r") as in_file:
        for line in in_file.readlines():
            fileid, txt, *label = line.split('|')
            id2label[fileid] = label
    label = {file_name: parse_label(hps, *label) for file_name, label in id2label.items()}
    return label



def parse_label(hps, pho, pitchid, dur, gtdur, slur):
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


def inference_once(net_g, label, cuda_id=None):
    pho, pitchid, dur, slur, _ = label
    device = torch.device(f"cuda:{cuda_id}" if cuda_id != None else "cpu")
    with torch.no_grad():
        # data
        pho_lengths = torch.LongTensor([pho.size(0)])
        pho = pho.unsqueeze(0)
        pitchid = pitchid.unsqueeze(0)
        dur = dur.unsqueeze(0)
        slur = slur.unsqueeze(0)

        if(cuda_id != None):
            net_g = net_g.to(device)
            pho = pho.to(device)
            pho_lengths = pho_lengths.to(device)
            pitchid = pitchid.to(device)
            dur = dur.to(device)
            slur = slur.to(device)

        # infer
        o, _, _ = net_g.infer(pho, pho_lengths, pitchid, dur, slur)
        audio = o[0,0].data.to(device).float().numpy()
        audio = audio * 32768 #hps.data.max_wav_value
        audio = audio.astype(np.int16)
        return audio


def inference_all(net_g, label_list_path, output_dir, hps, cuda_id=None, output_wav=True):
    label = parse_file(label_list_path)
    ret = {}
    for file_name, label in tqdm(label.items()):
        audio = inference_once(net_g, label, cuda_id)
        ret[file_name] = audio
        if output_wav:
            write(os.path.join(output_dir, file_name.split('.')[0] + '.wav' ), hps.data.sample_rate, audio)
    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_dir', '--model_dir', type=str, required=True)
    parser.add_argument('-input_dir', '--input_dir', type=str, required=True)
    parser.add_argument('-output_dir', '--output_dir', type=str, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir
    input_dir = args.input_dir
    output_dir = args.output_dir

    model, hps = load_model(model_dir)
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    print("load model end!")

    audio = inference_all(model, input_dir, output_dir, hps)
