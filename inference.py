from synthesizer.synthesize import infer as synthesizer_infer
from resgrad.inference import infer as resgrad_infer
from vocoder.inference import infer as vocoder_infer
from utils import load_models, save_result, load_yaml_file, get_file_name

import argparse
import time

def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker_id", type=int, default=None, required=False)
    parser.add_argument("--synthesizer_restore_step", type=int, required=True)
    parser.add_argument("--regrad_restore_step", type=int, required=False)
    parser.add_argument("--vocoder_restore_step", type=int, default=0 ,required=False)
    parser.add_argument("--result_dir", type=str, default="output/Persian/results", required=False)
    parser.add_argument("--result_file_name", type=str, default="", required=False)
    parser.add_argument("--pitch_control", type=float, default=1.0, required=False)
    parser.add_argument("--energy_control", type=float, default=1.0, required=False)
    parser.add_argument("--duration_control", type=float, default=1.0, required=False)
    parser.add_argument("--config", type=str, default='config/Persian/config.yaml', required=False, help="path to config.yaml")
    args = parser.parse_args()
    
    # Read Config
    config = load_yaml_file(args.config)
    
    print("load models...")
    restore_steps = {"synthesizer":args.synthesizer_restore_step, "resgrad":args.regrad_restore_step, "vocoder":args.vocoder_restore_step}
    synthesizer_model, resgrad_model, vocoder_model = load_models(restore_steps, config)

    ## Synthesizer
    control_values = args.pitch_control, args.energy_control, args.duration_control
    start_time = time.time()
    mel_prediction, duration_prediction, pitch_prediction, energy_prediction = synthesizer_infer(synthesizer_model, args.text, control_values, \
                                                                                        config['synthesizer']['preprocess'], \
                                                                                        config['main']['device'], \
                                                                                        speaker = args.speaker_id)
    end_time = time.time()
    FastSpeech_process_time = end_time-start_time
    
    ## Save FastSpeech2 result as wav
    wav = vocoder_infer(vocoder_model, mel_prediction, config['synthesizer']['preprocess']["preprocessing"]["audio"]["max_wav_value"])
    print("Save FastSpeech2 result...")
    file_name = get_file_name(args)
    save_result(mel_prediction, wav, pitch_prediction, energy_prediction, config['synthesizer']['preprocess'], args.result_dir, file_name)

    ## Real-Time factor of FastSpeech2
    wav_length = len(wav)/config['synthesizer']['preprocess']["preprocessing"]["audio"]["sampling_rate"]
    RTF_FastSpeech = FastSpeech_process_time / wav_length
    print("FastSpeech2 RTF: {:.6f}".format(RTF_FastSpeech))

    # ## ResGrad
    # print("Inference from ResGrad...")
    # start_time = time.time()
    # mel_prediction = resgrad_infer(resgrad_model, mel_prediction, duration_prediction, args.speaker_id, config['resgrad'])
    # end_time = time.time()
    # ResGrad_process_time = end_time-start_time

    # ## Vocoder
    # wav = vocoder_infer(vocoder_model, mel_prediction, config['synthesizer']['preprocess']["preprocessing"]["audio"]["max_wav_value"])

    # ## Save result
    # print("Save ResGrad result...")
    # file_name = file_name.replace("FastSpeech", "ResGrad")
    # save_result(mel_prediction.squeeze(), wav, pitch_prediction, energy_prediction, config['synthesizer']['preprocess'], args.result_dir, file_name)

    # ## Real-Time factor of ResGrad
    # wav_length = len(wav)/config['synthesizer']['preprocess']["preprocessing"]["audio"]["sampling_rate"]
    # RTF_ResGrad = ResGrad_process_time / wav_length
    # print("ResGrad RTF: {:.6f}".format(RTF_ResGrad))


if __name__ == "__main__":
    infer()