import os, pdb
from matplotlib import pyplot as plt
import matplotlib as mpl
from colorspacious import cspace_converter
import torchaudio, torch
from torchmetrics.audio import PerceptualEvaluationSpeechQuality 
PESQ = PerceptualEvaluationSpeechQuality(16000, 'wb')

def make_distortion(spec):
	spec=spec.squeeze()
	spec[:20,...] = 0
	wav = torch.istft(spec, 256,128,256,torch.hann_window(256))
	return wav, spec

def from_wav_to_spec(path, map_name, folder):
    wav = torchaudio.load("vis_audio.wav")[0]
    spec = torch.stft(wav, 256, 128, 256, torch.hann_window(256), return_complex=True)
    spec = torch.view_as_real(spec)
    pst = torch.log1p((spec[...,0]**2+spec[...,1]**2)**0.5)
    plt.imshow(torch.log1p(pst).squeeze().T.numpy(), cmap=map_name)
    plt.savefig("./spec/"+folder+"/spectrogram_"+map_name+".png", dpi=600)
    distorted_wav, distorted_spec = make_distortion(spec)
    distorted_wav = distorted_wav/distorted_wav.max()
    torchaudio.save("distorted_20.wav", distorted_wav.squeeze().unsqueeze(0), 16000)
    pesq_score = PESQ(wav.squeeze(), distorted_wav)
    distorted_pst = torch.log1p((distorted_spec[...,0]**2+distorted_spec[...,1]**2)**0.5)
    # pdb.set_trace()
    plt.imshow(torch.log1p(distorted_pst).squeeze().T.numpy(), cmap=map_name)
    plt.savefig("./spec/"+folder+"/spectrogram_"+map_name+"_"+str(pesq_score.item())+".png", dpi=600)

def try_maps(path, list_of_maps, folder):
	for map_name in list_of_maps:
		from_wav_to_spec(path, map_name, folder)

def main(path):
	
	folder = 'Perceptually Uniform Sequential'
	if not os.path.isdir("spec/"+folder):
		os.makedirs("spec/"+folder)
	list_of_maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
	try_maps(path, list_of_maps, folder)
	
	folder = 'Sequential'
	if not os.path.isdir("spec/"+folder):
		os.makedirs("spec/"+folder)
	list_of_maps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 
				 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
				 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
	try_maps(path, list_of_maps, folder)	
	folder = 'Sequential2'
	if not os.path.isdir("spec/"+folder):
		os.makedirs("spec/"+folder)
	list_of_maps = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                      'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                      'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
	try_maps(path, list_of_maps, folder)
	folder = 'Diverging'
	if not os.path.isdir("spec/"+folder):
		os.makedirs("spec/"+folder)
	list_of_maps = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
	try_maps(path, list_of_maps, folder)
	folder = 'Cyclic'
	if not os.path.isdir("spec/"+folder):
		os.makedirs("spec/"+folder)
	list_of_maps = ['twilight', 'twilight_shifted', 'hsv']
	try_maps(path, list_of_maps, folder)
	folder = 'Qualitative'
	if not os.path.isdir("spec/"+folder):
		os.makedirs("spec/"+folder)
	list_of_maps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                      'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                      'tab20c']
	try_maps(path, list_of_maps, folder)		
    
if __name__=='__main__':
	path = #PATH TO GENERATED AUDIO
	main(path)