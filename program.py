# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 19:34:13 2021

@author: Praditya Rizki
"""
from libraries import*
#from untitled0 import *
import pandas as pd
import numpy as np


'''
load data dan ekstraksi fitur
'''
folder = 'Data'
nama_file = 'FixedPatient6'
anotfile = 'Anotasi'
ppg_signal, ppg_annotation = annotation_to_ppg_signal(folder, anotfile, nama_file)
x, y = get_ppg_features_statistic_entropy(ppg_signal, ppg_annotation, 'bior3.9')


listtocsv(x, '%s.csv'%(nama_file))
# =============================================================================
# '''
# bikin file npy
# '''
# folder = 'Data'
# nama_file = 'FixedPatient2'
# ppg_signal = load_dataPPG(folder, nama_file)
# =============================================================================

#signal = np.load("FixedPatient1.npy", allow_pickle=True)
# =============================================================================
# signal = np.load("FixedPatient1.npy", allow_pickle=True)
# =============================================================================
# =============================================================================
# '''
# load data dan ekstraksi fitur revisi
# '''
# folder = 'Data'
# nama_file = 'FixedPatient1'
# anotfile = 'Anotasi'
# ppg_signal, ppg_annotation = annotation_to_ppg_signal(folder, anotfile, nama_file)
# x = time_domain_features(ppg_signal)
# #z, y = get_ppg_features_statistic(ppg_signal, ppg_annotation, 'sym8')
# 
# listtocsv(x, '%s.csv'%(nama_file))
# =============================================================================
# =============================================================================
# signal = np.load("FixedPatient2.npy", allow_pickle=True)
# #signal = pd.read_csv('Data/FixedPatient1.csv', delimiter = ';')
# mu, sigma = 0, 0.1 
# #Generate Noise
# noise = np.random.normal(mu, sigma, signal.shape) 
# #Create Noisy Signal
# noisy_signal = signal+noise
# #Remove baseline using moving average
# baseline_removed = baseline_remove_ppg(noisy_signal)
# #Remove noise using DWT by removing high frequency si5gnal
# clean_ppg = remove_noise(baseline_removed)
# #Scale to 0-1
# scaled = scaler.fit_transform(clean_ppg.reshape(-1,1))[:,0]
# #Get Peaks
# peaks = find_signal_peaks(scaled,minimum=0.4)
# #Generate peak list
# rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=500)
# 
# plt.plot(signal)
# plt.savefig("sinyal_asli",dpi=1000)
# plt.clf()
# plt.plot(noisy_signal)
# plt.savefig("sinyal_noisy",dpi=1000)
# plt.clf()
# plt.plot(clean_ppg)
# plt.savefig("sinyal_clean",dpi=1000)
# plt.clf()
# plot_with_rpeaks(clean_ppg,peaks)
# =============================================================================
# =============================================================================
# '''
# KLASIFIKASI & metrik uji
# '''
# folder_file = 'Fitur PPI'
# fitur_files = glob.glob(folder_file + '/*.csv')
# anotasi_folder = 'Anotasi'
# anotasi_files = glob.glob(anotasi_folder + '/*.csv')
# model = DecisionTreeClassifier()
# 
# fit = []
# lab = []
# 
# for filename in fitur_files:
#     df = pd.read_csv(filename, index_col = None, header = None)
#     fit.append(df)
# 
# datafinal = pd.concat(fit, axis=0, ignore_index=True)
# 
# for filename1 in anotasi_files:
#     df1 = pd.read_csv(filename1, delimiter = ';')
#     ss = df1['event'].values.tolist()
#     lab = lab + ss
#     
# labeltest = 'AF'
# X_train, X_test, y_train, y_test = train_test_split(datafinal, lab, test_size=0.20, random_state=42)
# z = run_experiment(model, X_train, y_train, X_test, y_test, labeltest)
# =============================================================================



