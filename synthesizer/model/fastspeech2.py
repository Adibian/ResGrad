import os
import json

import torch.nn as nn
import torch.nn.functional as F

from ..transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from ..utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, config):
        super(FastSpeech2, self).__init__()
        preprocess_config, model_config = config['synthesizer']['preprocess'], config['synthesizer']['model']
        self.model_config = model_config
        self.device = config['synthesizer']['main']['device']

        self.encoder = Encoder(model_config, preprocess_config['preprocessing']['text']['language'])
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, self.device)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if config['main']['multi_speaker']:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len, self.device)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len, self.device)
            if mel_lens is not None
            else None
        )
        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            self.speaker_vec = self.speaker_emb(speakers).unsqueeze(1)
            output = output + self.speaker_vec.expand(-1, max_src_len, -1)     


        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        ) 

        # if self.speaker_emb is not None:
        #     output = output + self.speaker_vec.expand(-1, output.shape[1], -1)     
        
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output
        
        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
        
       

