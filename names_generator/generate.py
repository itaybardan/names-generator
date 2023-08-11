import torch
import torchaudio
from names_generator.dataset import NamesDataset
from model import NamesGeneratorModel
from names_generator import CONFIG
import os

dataset = NamesDataset(CONFIG.dataset_root_folder)
model = NamesGeneratorModel(dataset.vocab_size)
model = model.to(CONFIG.device)
saved_state_dict = torch.load(CONFIG.model_path)
model.load_state_dict(saved_state_dict)
model.eval()

with torch.no_grad():
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=CONFIG.device)
    output = model.generate(context, CONFIG.context_length, 3000)[0].tolist()
    decoded = [dataset.index_to_freq[i] for i in output]
    waveform = torch.tensor(decoded).view(1, -1)
    output_path = os.path.join(CONFIG.output_path, 'output.wav')
    torchaudio.save('output.wav', waveform, 1000)
