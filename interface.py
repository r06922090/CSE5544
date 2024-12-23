import torch
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from DNSMOS.dnsmos_local import file_dnsmos
from torchmetrics.audio import SignalNoiseRatio
from torchmetrics.audio import SignalDistortionRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

clean_dir_path = 'audio_files/Clean'
noisy_dir_path = 'audio_files/Noisy'
sampling_rate = 16000

class App(tk.Tk):
    def __init__(self, title, geometry):
        super().__init__()
        # Window parameters
        self.title(title)
        self.geometry(geometry)
        
        ###
        # Left frame 1 contains color map buttons
        ###
        self.left_frame_1 = tk.Frame(self)
        self.left_frame_1.pack(fill='both', side='left')
        
        # Load file button
        self.load_button = tk.Button(self.left_frame_1, text='Load Audio File', command=self.load_file)
        self.load_button.pack(fill='both', pady=10, ipady=5) 

        # Color map buttons 
        self.map1_button = tk.Button(self.left_frame_1, text='Cividis Map', command=lambda: self.color_mapping(1))
        self.map1_button.pack(expand=True, fill='both')
        
        self.map2_button = tk.Button(self.left_frame_1, text='Inferno Map', command=lambda: self.color_mapping(2))
        self.map2_button.pack(expand=True, fill='both')
        
        self.map3_button = tk.Button(self.left_frame_1, text='Magma Map', command=lambda: self.color_mapping(3))
        self.map3_button.pack(expand=True, fill='both')
        
        self.map4_button = tk.Button(self.left_frame_1, text='Plasma Map', command=lambda: self.color_mapping(4))
        self.map4_button.pack(expand=True, fill='both')
        
        self.map5_button = tk.Button(self.left_frame_1, text='Viridis Map', command=lambda: self.color_mapping(5))
        self.map5_button.pack(expand=True, fill='both')


        ###
        # Left frame 2 contains spectrogram
        ###
        self.left_frame_2 = tk.Frame(self)
        self.left_frame_2.pack(expand=True, fill='both', side='left')
        
        # File path label
        self.path_label = tk.Entry(self.left_frame_2, font=('Arial', 12), fg="gray", bg="white", justify='center',
                                  borderwidth=1, relief="solid")
        self.path_label.pack(fill='both', pady=12, ipady=5) 
        self.path_label.insert(tk.INSERT, 'file/path/appears/here')
                
        # Image label
        self.image_label = tk.Label(self.left_frame_2, text='Spectrogram Appears Here', font=('Arial', 12), 
                                    fg="gray", bg="white", borderwidth=1, relief="solid")
        self.image_label.pack(expand=True, fill='both') 
        
        
        ###
        # Left frame 3 contains metric scores
        ###
        self.left_frame_3 = tk.Frame(self)
        self.left_frame_3.pack(fill='both', side='left')
        
        # Play file button
        self.play_button = tk.Button(self.left_frame_3, text='Play Audio File')
        self.play_button.pack(fill='both', pady=10, ipady=5) 
        
        self.subframe1 = tk.Frame(self.left_frame_3)
        self.subframe1.pack(expand=True, fill='both', side='left')
        
        self.subframe2 = tk.Frame(self.left_frame_3)
        self.subframe2.pack(expand=True, fill='both', side='left')
        
        border_width = 1
        entry_width = 8
        
        self.pesq_button = tk.Button(self.subframe1, text='PESQ', command=self.find_pesq)
        self.pesq_button.pack(expand=True, fill='both', side='top', padx=1, pady=1)
        self.pesq_label = tk.Entry(self.subframe2, font=('Arial', 12), fg="gray", bg="white",  justify='center',
                                   width=entry_width, borderwidth=border_width, relief="solid")
        self.pesq_label.pack(expand=True, fill='both', side='top')
        self.pesq_label.insert(tk.INSERT, '-0.5')
     
        self.stoi_button = tk.Button(self.subframe1, text='STOI', command=self.find_stoi)
        self.stoi_button.pack(expand=True, fill='both', side='top', padx=1, pady=1)
        self.stoi_label = tk.Entry(self.subframe2, font=('Arial', 12), fg="gray", bg="white",  justify='center',
                                   width=entry_width, borderwidth=border_width, relief="solid")
        self.stoi_label.pack(expand=True, fill='both', side='top')
        self.stoi_label.insert(tk.INSERT, '0.0')

        self.sdr_button = tk.Button(self.subframe1, text='SDR', command=self.find_sdr)
        self.sdr_button.pack(expand=True, fill='both', side='top', padx=1, pady=1)
        self.sdr_label = tk.Entry(self.subframe2, font=('Arial', 12), fg="gray", bg="white", justify='center',
                                  width=entry_width, borderwidth=border_width, relief="solid")
        self.sdr_label.pack(expand=True, fill='both', side='top')
        self.sdr_label.insert(tk.INSERT, '0dB')

        self.snr_button = tk.Button(self.subframe1, text='SNR', command=self.find_snr)
        self.snr_button.pack(expand=True, fill='both', side='top', padx=1, pady=1)
        self.snr_label = tk.Entry(self.subframe2, font=('Arial', 12), fg="gray", bg="white", justify='center',
                                  width=entry_width, borderwidth=border_width, relief="solid")
        self.snr_label.pack(expand=True, fill='both', side='top')
        self.snr_label.insert(tk.INSERT, '0db')
        
        self.dnsmos_button = tk.Button(self.subframe1, text='DNSMOS', command=self.find_dnsmos)
        self.dnsmos_button.pack(expand=True, fill='both', side='top', padx=1, pady=1)
        self.dnsmos_label = tk.Entry(self.subframe2, font=('Arial', 12), fg="gray", bg="white",  justify='center',
                                     width=entry_width, borderwidth=border_width, relief="solid")
        self.dnsmos_label.pack(expand=True, fill='both', side='top')
        self.dnsmos_label.insert(tk.INSERT, '1.0')


    def load_file(self):
        self.file_path = filedialog.askopenfilename(initialdir=noisy_dir_path) 
        if self.file_path:
            # delete all existing entries
            self.pesq_label.delete(0, tk.END)
            self.stoi_label.delete(0, tk.END)
            self.dnsmos_label.delete(0, tk.END)
            self.sdr_label.delete(0, tk.END)
            self.snr_label.delete(0, tk.END)
            
            # replace old file path with current one
            self.path_label.delete(0, tk.END)
            self.path_label.insert(0, self.file_path)

            # load clean and noisy version of audio
            self.f_name = self.file_path.split('/')[-1]
            self.ref_filepath = clean_dir_path + '/' + self.f_name
            self.deg_filepath = noisy_dir_path + '/' + self.f_name
            
            self.ref, _ = librosa.load(self.ref_filepath, sr=sampling_rate)
            self.deg, _ = librosa.load(self.deg_filepath, sr=sampling_rate)
            self.ref_tensor = torch.from_numpy(self.ref)
            self.deg_tensor = torch.from_numpy(self.deg)
                        
            # display spectrogram in default colormap
            self.compute_spectrogram()  
            self.show_spectrogram()
    

    def compute_spectrogram(self):
        # load and compute power spectrum
        self.selected_cmap = 'magma'
        self.sampled_audio = self.deg
        self.complex_stft = librosa.stft(self.sampled_audio, n_fft=256, hop_length=128, win_length=256)
        self.power_spectrum = librosa.amplitude_to_db(np.abs(self.complex_stft), ref=np.max)


    def show_spectrogram(self):
        # save spectrum in heatmap format
        plt.figure(); librosa.display.specshow(self.power_spectrum, cmap=self.selected_cmap); 
        plt.savefig('Temp/Spectrum.png'); plt.close()
        
        # load image of power spectrogram 
        image = Image.open('Temp/Spectrum.png')

        # resizing spectrogram to fit label widget
        self.image_label.update()
        image_w, image_h = self.image_label.winfo_width(), self.image_label.winfo_height()
        image = image.resize((image_w, image_h))

        # display tk-compatible version on interface
        photo_image = ImageTk.PhotoImage(image=image)
        self.image_label.config(image=photo_image)
        self.image_label.image = photo_image  # Keep a reference to prevent garbage collection
        self.image_label.pack()
    
    
    def color_mapping(self, index):
        self.cmaps = ['cividis', 'inferno', 'magma', 'plasma', 'viridis']
        self.selected_cmap = self.cmaps[index-1]
        self.show_spectrogram()

  
    def find_pesq(self):
        pesq = PerceptualEvaluationSpeechQuality(sampling_rate, 'wb')
        api_value = pesq(self.deg_tensor, self.ref_tensor).numpy()
        self.pesq_value = (3.8224 - np.log((4.999 - api_value) / (api_value - 0.999))) / 1.3669
        self.pesq_label.insert(0, self.pesq_value)
        return
    
    
    def find_stoi(self):
        stoi = ShortTimeObjectiveIntelligibility(sampling_rate)
        self.stoi_value = stoi(self.deg_tensor, self.ref_tensor).numpy()
        self.stoi_label.insert(0, self.stoi_value)
        return
    
    
    def find_sdr(self):  
        sdr = SignalDistortionRatio()
        self.sdr_value = sdr(self.deg_tensor, self.ref_tensor).numpy()
        self.sdr_label.insert(0, self.sdr_value)
        return
    
    
    def find_snr(self):
        snr = SignalNoiseRatio()
        self.snr_value = snr(self.deg_tensor, self.ref_tensor).numpy()
        self.snr_label.insert(0, self.snr_value)
        return
    
    
    def find_dnsmos(self):
        self.dnsmos_value = file_dnsmos(self.deg_filepath, '')
        self.dnsmos_label.insert(0, self.dnsmos_value)
        return
    

app = App(title='Data Visualization for Speech', geometry="800x500")
app.mainloop()