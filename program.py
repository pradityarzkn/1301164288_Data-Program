# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 19:34:13 2021

@author: Praditya Rizki
"""
from libraries import*
import pandas as pd
import numpy as np
'''
--------------------PETUNJUK --------------------------
1. Jalankan Bagian 1
2. Jalankan Bagian 2

BAGIAN 1 untuk load data dan ekstraksi fitur
- 1A Dekomposisi sinyal
- 1B Plot Peak to Peak Interval
- 1C Time Domain Features
- 1D Shannon Entropy
- 1E Statistical
- 1F Time Domain Features + Shannon Entropy
- 1G Time Domain Features + Statistical
- 1H Seluruh Fitur

BAGIAN 2 untuk UJI Klasifikasi

--------------------------------------------------------
'''

'''
BAGIAN 1A
-----------------------------------------------------------------
Pilih file dan ganti bagian nama_file sesuai data yang diinginkan
----------------------------------------------------------------- 
'''
folder = 'Data'
nama_file = 'FixedPatient1'
anotfile = 'Anotasi'
ppg_signal, ppg_annotation = annotation_to_ppg_signal(folder, anotfile, nama_file)


w = pywt.Wavelet('sym5')    
signal = ppg_signal[2]
list_coeff = pywt.wavedec(signal, 'sym5' , level=5)
reconstruction_plot(pywt.waverec(list_coeff[:-1] + [None] * 1, w))
reconstruction_plot(pywt.waverec(list_coeff[:-2] + [None] * 2, w)) # leaving out detail coefficients up to lvl 4
reconstruction_plot(pywt.waverec(list_coeff[:-3] + [None] * 3, w)) # leaving out detail coefficients up to lvl 3
reconstruction_plot(pywt.waverec(list_coeff[:-4] + [None] * 4, w)) # leaving out detail coefficients up to lvl 2
reconstruction_plot(pywt.waverec(list_coeff[:-5] + [None] * 5, w)) # leaving out detail coefficients up to lvl 2
plt.legend(['1', '2', '3', '4', '5'])

'''
BAGIAN 1B
------------------------------------------------------
Ganti data sesuai yang diinginkan pada bagian signal
------------------------------------------------------
'''
signal = np.load("FixedPatient1.npy", allow_pickle=True)
#signal = pd.read_csv('Data/FixedPatient1.csv', delimiter = ';')
mu, sigma = 0, 0.1 
#Generate Noise
noise = np.random.normal(mu, sigma, signal.shape) 
#Create Noisy Signal
noisy_signal = signal+noise
#Remove baseline using moving average
baseline_removed = baseline_remove_ppg(noisy_signal)
#Remove noise using DWT by removing high frequency si5gnal
clean_ppg = remove_noise(baseline_removed)
#Scale to 0-1
scaled = scaler.fit_transform(clean_ppg.reshape(-1,1))[:,0]
#Get Peaks
peaks = find_signal_peaks(scaled,minimum=0.4)
#Generate peak list
rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=500)

plt.plot(signal)
plt.savefig("sinyal_asli",dpi=1000)
plt.clf()
plt.plot(noisy_signal)
plt.savefig("sinyal_noisy",dpi=1000)
plt.clf()
plt.plot(clean_ppg)
plt.savefig("sinyal_clean",dpi=1000)
plt.clf()
plot_with_rpeaks(clean_ppg,peaks)

'''
Bagian 1C
-----------------------------------------------------------------
Pilih file dan ganti bagian nama_file sesuai data yang diinginkan
-----------------------------------------------------------------
'''
folder = 'Data'
nama_file = 'normal4'
anotfile = 'Anotasi'
ppg_signal, ppg_annotation = annotation_to_ppg_signal(folder, anotfile, nama_file)
x = time_domain_features(ppg_signal)

listtocsv(x, '%s.csv'%(nama_file))

'''
Bagian 1D
-----------------------------------------------------------------
Pilih file dan ganti bagian nama_file sesuai data yang diinginkan
-----------------------------------------------------------------
'''
folder = 'Data'
nama_file = 'normal4'
anotfile = 'Anotasi'
ppg_signal, ppg_annotation = annotation_to_ppg_signal(folder, anotfile, nama_file)
x,y = get_ppg_features_entropy(ppg_signal)
listtocsv(x, '%s.csv'%(nama_file))
'''
Bagian 1E
-----------------------------------------------------------------
Pilih file dan ganti bagian nama_file sesuai data yang diinginkan
-----------------------------------------------------------------
'''
folder = 'Data'
nama_file = 'normal4'
anotfile = 'Anotasi'
ppg_signal, ppg_annotation = annotation_to_ppg_signal(folder, anotfile, nama_file)
x,y = get_ppg_features_statistic(ppg_signal)
listtocsv(x, '%s.csv'%(nama_file))
'''
Bagian 1F
-----------------------------------------------------------------
Pilih file dan ganti bagian nama_file sesuai data yang diinginkan
-----------------------------------------------------------------
'''
folder = 'Data'
nama_file = 'normal4'
anotfile = 'Anotasi'
ppg_signal, ppg_annotation = annotation_to_ppg_signal(folder, anotfile, nama_file)
x = time_domain_features_entropy(ppg_signal)
listtocsv(x, '%s.csv'%(nama_file))
'''
Bagian 1G
-----------------------------------------------------------------
Pilih file dan ganti bagian nama_file sesuai data yang diinginkan
-----------------------------------------------------------------
'''
folder = 'Data'
nama_file = 'normal4'
anotfile = 'Anotasi'
ppg_signal, ppg_annotation = annotation_to_ppg_signal(folder, anotfile, nama_file)
x= time_domain_features_stat(ppg_signal)
listtocsv(x, '%s.csv'%(nama_file))                                 
'''
BAGIAN 2
------------------------------------------------------------------
Ganti Folder File sesuai dengan fitur yang akan di uji klasifikasi
pada bagian folder_file
------------------------------------------------------------------
'''
folder_file = 'Fitur PPI'
fitur_files = glob.glob(folder_file + '/*.csv')
anotasi_folder = 'Anotasi'
anotasi_files = glob.glob(anotasi_folder + '/*.csv')
model = DecisionTreeClassifier()

fit = []
lab = []

for filename in fitur_files:
    df = pd.read_csv(filename, index_col = None, header = None)
    fit.append(df)

datafinal = pd.concat(fit, axis=0, ignore_index=True)

for filename1 in anotasi_files:
    df1 = pd.read_csv(filename1, delimiter = ';')
    ss = df1['event'].values.tolist()
    lab = lab + ss
    
labeltest = 'Normal'
X_train, X_test, y_train, y_test = train_test_split(datafinal, lab, test_size=0.20, random_state=42)
z = run_experiment(model, X_train, y_train, X_test, y_test, labeltest)


