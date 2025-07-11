

import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from FastSpeech2.utils.model import get_model, get_vocoder

from FastSpeech2.utils.tools import to_device, synth_samples
from FastSpeech2.dataset import TextDataset
from FastSpeech2.text import text_to_sequence
from FastSpeech2.utils.tools import synth_samples_in_memory
from types import SimpleNamespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


# def synthesize(model, step, configs, vocoder, batchs, control_values):
#     preprocess_config, model_config, train_config = configs
#     pitch_control, energy_control, duration_control = control_values

#     for batch in batchs:
#         batch = to_device(batch, device)
#         with torch.no_grad():
#             # Forward
#             output = model(
#                 *(batch[2:]),
#                 p_control=pitch_control,
#                 e_control=energy_control,
#                 d_control=duration_control
#             )
#             synth_samples(
#                 batch,
#                 output,
#                 vocoder,
#                 model_config,
#                 preprocess_config,
#                 train_config["path"]["result_path"],
#             )


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            # Return the first (or only) sample’s audio buffer
            return synth_samples_in_memory(
                batch, output, vocoder, model_config, preprocess_config
            )



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)




def synthesize_from_text(
    text, step, preprocess_path, model_path, train_path,
    pitch=1.1, energy=1.05, duration=0.95, speaker_id=1
):
    # 1. Load configs
    preprocess_config = yaml.load(open(preprocess_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_path, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # 2. Construct dummy args as required by get_model
    args = SimpleNamespace(
        restore_step=int(step),
        pitch_control=pitch,
        energy_control=energy,
        duration_control=duration
    )

    # 3. Load model and vocoder
    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    # 4. Preprocess text into tensors
    ids = raw_texts = [text[:100]]
    speakers = np.array([speaker_id])

    if preprocess_config["preprocessing"]["text"]["language"] == "en":
        texts = np.array([preprocess_english(text, preprocess_config)])
    else:
        texts = np.array([preprocess_mandarin(text, preprocess_config)])

    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    # 5. Run synthesis and return audio buffer
    control_values = (pitch, energy, duration)
    return synthesize(model, step, configs, vocoder, batchs, control_values)


if __name__ == "__main__":
    main()