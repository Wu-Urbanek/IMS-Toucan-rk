import torch
from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "sanity_check_best.pt"

tts = ToucanTTSInterface(device=device, tts_model_path=checkpoint_path)

tts.set_language("dru")

text = "ku ama"
output_path ="kuama.wav"

tts.read_to_file(text_list=[text], file_path=output_path)

print("Saved wav to output_path")