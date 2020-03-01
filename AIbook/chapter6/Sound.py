import librosa
import librosa.display
from matplotlib import  pyplot ,image

relative="content\\genres\\blues\\"

filename = 'blues.00000.wav'
filePath=relative+ filename
y, sr = librosa.load(filePath)
print("number of samples=",str(len(y)))
print("SampleRate=",str(sr))

librosa.display.waveplot(y, sr=sr);
pyplot.show()

import numpy as np
n_fft = 2048
D=librosa.stft(y[:n_fft], n_fft=n_fft,
                       hop_length=n_fft+1)

D1 = np.abs(D)

print(D[:10])
print(D1[:10])

pyplot.plot(D1)
pyplot.show()


import matplotlib.pyplot as plt
hop_length = 512
D = np.abs(librosa.stft(y, n_fft=n_fft,  
                        hop_length=hop_length))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear');
pyplot.colorbar();
pyplot.show()

DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, hop_length=hop_length, 
                         x_axis='time', y_axis='log');
pyplot.colorbar(format='%+2.0f dB');
pyplot.show()


S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, 
                                   hop_length=hop_length
                                  )
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, 
                         x_axis='time', y_axis='mel');
pyplot.colorbar(format='%+2.0f dB');
pyplot.title('melspectogram for blues.00000.wav')
pyplot.show()


relative="content\\genres\\rock\\"
filename = 'rock.00000.wav'
filePath=relative+ filename
y, sr = librosa.load(filePath)
S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, 
                                   hop_length=hop_length
                                  )
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, 
                         x_axis='time', y_axis='mel');
pyplot.colorbar(format='%+2.0f dB');
pyplot.title('melspectogram for rock.00000.wav')
pyplot.show()

filePath=relative+ filename
y, sr = librosa.load(filePath)
sr=16000
S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, 
                                   hop_length=hop_length
                                  )
S_DB = librosa.power_to_db(S[0:1000], ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, 
                         x_axis='time', y_axis='mel');
pyplot.colorbar(format='%+2.0f dB');
pyplot.title('melspectogram for rock.00000.wav 16000Hz')
pyplot.show()
