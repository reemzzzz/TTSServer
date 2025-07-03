import os
import json

import torch
import numpy as np

import FastSpeech2.hifigan as hifigan
from FastSpeech2.model import FastSpeech2, ScheduledOptim


import os
import torch
import requests
from io import BytesIO
from FastSpeech2.model import FastSpeech2, ScheduledOptim

def download_from_gdrive(file_id):
    print("📥 Downloading model from Google Drive...")
    base_url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(base_url, params={"id": file_id}, stream=True)
    
    # Check for confirmation token (for large files)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(base_url, params={"id": file_id, "confirm": value}, stream=True)
            break

    file_buffer = BytesIO()
    for chunk in response.iter_content(32768):
        file_buffer.write(chunk)

    file_buffer.seek(0)
    print("✅ Model download complete.")
    return file_buffer

def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)

    if args.restore_step:
        print(f"🔧 Restoring checkpoint from step {args.restore_step}...")

        # ✅ Provide your actual Google Drive file ID here
        gdrive_file_id = "1ukvuRIRJQUATD642az1_KL-yEBkRHKLE"

        # Download .pth.tar checkpoint file into memory
        buffer = download_from_gdrive(gdrive_file_id)
        ckpt = torch.load(buffer, map_location=torch.device('cpu'))

        model.load_state_dict(ckpt["model"])
        print("✅ Model weights loaded.")

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



# def get_model(args, configs, device, train=False):
#     (preprocess_config, model_config, train_config) = configs

#     model = FastSpeech2(preprocess_config, model_config).to(device)
#     if args.restore_step:
#         ckpt_path = os.path.join(
#             train_config["path"]["ckpt_path"],
#             "{}.pth.tar".format(args.restore_step),
#         )
#         ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
#         model.load_state_dict(ckpt["model"])

#     if train:
#         scheduled_optim = ScheduledOptim(
#             model, train_config, model_config, args.restore_step
#         )
#         if args.restore_step:
#             scheduled_optim.load_state_dict(ckpt["optimizer"])
#         model.train()
#         return model, scheduled_optim

#     model.eval()
#     model.requires_grad_ = False
#     return model


# def get_param_num(model):
#     num_param = sum(param.numel() for param in model.parameters())
#     return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt_path = "hifigan/generator_LJSpeech.pth.tar"
            ckpt = torch.load(ckpt_path,map_location=torch.device('cpu'))
        elif speaker == "universal":
            ckpt_path = "hifigan/generator_universal.pth.tar"
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
