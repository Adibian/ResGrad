## ResGrad - PyTorch Implementation
[**ResGrad: Residual Denoising Diffusion Probabilistic Models for Text to Speech**](https://arxiv.org/abs/2212.14518)

This is an *unofficial* PyTorch implementation of **ResGrad** as a high-quality speech synthesis model. In short, this model generates the spectrogram using FastSpeech2 and then removes the noise in the spectrogram using the Diffusion method to synthesize high-quality speeches. As mentioned in the paper the implementation is based on FastSpeech2 and Grad-TTS. Also, the HiFiGAN model is used to generate waveforms from synthesized spectrograms.


## Quickstart
Data structures:
`
dataset/data_name/synthesizer_data/
    test_data/
        speaker1/
            sample1.txt
            sample1.wav
            ...
        ...
    train_data/
        ...
    test.txt  (sample1|speaker1|*phoneme_sequence \n ...)
    train.txt (sample1|speaker1|*phoneme_sequence \n ...)
`

Preprocessing:
`
python synthesizer/prepare_align.py config/data_name/config.yaml
python synthesizer/preprocess.py config/data_name/config.yaml
`

Train synthesizer:
`
python train_synthesizer.py --config config/data_name/config.yaml
`

Prepare data for ResGrade:
`
python resgrad_data.py --synthesizer_restore_step 1000000 --data_file_path dataset/data_name/synthesizer_data/train.txt \
                        --config config/data_name/config.yaml
`

Train ResGrade:
`
python train_resgrad.py --config config/data_name/config.yaml
`

Inference:
`
python inference.py --text "phonemes sequence example" \
                    --synthesizer_restore_step 1000000 --regrad_restore_step 1000000 --vocoder_restore_step 2500000 \
                    --config config/data_name/config.yaml --result_dir output/data_name/results
`

## References :notebook_with_decorative_cover:
- [ResGrad: Residual Denoising Diffusion Probabilistic Models for Text to Speech](https://arxiv.org/abs/2212.14518), Z. Chen, *et al*.
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://arxiv.org/abs/2105.06337), V. Popov, *et al*.
