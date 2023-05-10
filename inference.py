from synthesizer.synthesize import infer as synthesizer_infer
from resgrad.inference import infer as resgrad_infer
from vocoder.inference import infer as vocoder_infer
from utils import load_model, save_result, get_synthesizer_configs

import argparse

def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--synthesizer_restore_step", type=int, required=True)
    parser.add_argument("--regrad_restore_epoch", type=int, required=True)
    parser.add_argument("--vocoder_restore_epoch", type=int, default=0 ,required=False)
    parser.add_argument("--result_dir", type=str, default="results", required=False)
    parser.add_argument("--pitch_control", type=float, default=1.0, required=False)
    parser.add_argument("--energy_control", type=float, default=1.0, required=False)
    parser.add_argument("--duration_control", type=float, default=1.0, required=False)
    parser.add_argument("--synthesizer_preprocess_config", type=str, default="synthesizer/config/persian/preprocess.yaml", required=False)
    parser.add_argument("--synthesizer_model_config", type=str, default="synthesizer/config/persian/model.yaml", required=False)
    parser.add_argument("--synthesizer_train_config", type=str, default="synthesizer/config/persian/train.yaml", required=False)
    args = parser.parse_args()

    synthesizer_configs = get_synthesizer_configs(args.synthesizer_preprocess_config, args.synthesizer_model_config, args.synthesizer_train_config)
    
    print("load models...")
    restore_steps = {"synthesizer":args.synthesizer_restore_step, "regrad":args.regrad_restore_epoch, "vocoder":args.vocoder_restore_epoch}
    synthesizer_model, resgrad_model, vocoder_model = load_model(restore_steps, synthesizer_configs)

    ## Synthesizer
    control_values = args.pitch_control, args.energy_control, args.duration_control
    mel_prediction, duration_prediction, pitch_prediction, energy_prediction = synthesizer_infer(synthesizer_model, args.text, control_values, \
                                                                                        synthesizer_configs['preprocess_config'], \
                                                                                        synthesizer_configs['model_config']['device'])

    ## ResGrad
    mel_prediction = resgrad_infer(resgrad_model, mel_prediction, duration_prediction)

    ## Vocoder
    wav = vocoder_infer(vocoder_model, mel_prediction, synthesizer_configs['preprocess_config']["preprocessing"]["audio"]["max_wav_value"])

    ## Save result
    save_result(mel_prediction, wav, pitch_prediction, energy_prediction, synthesizer_configs['preprocess_config'], args.result_dir)

