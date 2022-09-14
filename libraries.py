# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:56:26 2021

@author: Praditya Rizki
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime as dt
import scipy.io as sio
from sklearn.tree import DecisionTreeClassifier
from sklearn import*
from sklearn.metrics import*
from sklearn.metrics import confusion_matrix
import csv
import os
import pywt
from scipy.fftpack import fft
import time
from IPython.display import display
import scipy.stats
from collections import defaultdict, Counter
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import glob
from scipy.signal import find_peaks, resample
from sklearn.preprocessing import MinMaxScaler


sample_rate = 500
scaler = MinMaxScaler()

'''
Ekstraksi Fitur Statistik
'''
def calculate_statistic(list_values):
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    maximum = np.nanmax(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
#    return[median, mean, std, var]
    return[median, mean, std, var, rms]
    #return [mean]

'''
Ekstraksi Fitur Entropy
'''
def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

'''
Ekstraksi Fitur Level Crossing
'''
def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistic = calculate_statistic(list_values)
    time_domain = feature_extraction_time_domain_features1(list_values)
    return [entropy] + crossings + statistic + time_domain

                                
def get_features_entropy(list_values):
    entropy = calculate_entropy(list_values)
    return [entropy]


def get_features_crossings(list_values):
    crossings = calculate_crossings(list_values)
    return crossings


def get_features_statistic(list_values):
    statistic = calculate_statistic(list_values)
    return statistic


def get_features_statistic_entropy(list_values):
    statistic = calculate_statistic(list_values)
    entropy = calculate_entropy(list_values)
    return [entropy] + statistic


def get_features_statistic_crossings(list_values):
    statistic = calculate_statistic(list_values)
    crossings = calculate_crossings(list_values)
    return statistic + crossings


def get_features_crossings_entropy(list_values):
    crossings = calculate_crossings(list_values)
    entropy = calculate_entropy(list_values)
    return crossings + [entropy]

'''
Dekomposisi sinyal untuk mengambil koefisien fitur gabungan (DWT)
'''
    
def get_ppg_features(dataset, labels, waveletname):
    ppg_features = []
    for signal_no in range(0, len (dataset)):
        features = []
        signal = dataset[signal_no]
        list_coeff = pywt.wavedec(signal, waveletname, level =0)
        for coeff in list_coeff:
            features += get_features(coeff)
        ppg_features.append(features)
    x = np.array(ppg_features)
    y = np.array(labels)
    return x, y

'''
Dekomposisi sinyal untuk mengambil koefisien fitur shanonn entropy (DWT)
'''

def get_ppg_features_entropy(dataset, labels, waveletname):
    ppg_features = []
    for signal_no in range(0, len (dataset)):
        features = []
        signal = dataset[signal_no]
        list_coeff = pywt.wavedec(signal, waveletname, level=0)
        for coeff in list_coeff:
            features += get_features_entropy(coeff)
        ppg_features.append(features)
    x = np.array(ppg_features)
    y = np.array(labels)
    return x, y

'''
Dekomposisi sinyal untuk mengambil koefisien fitur crossings (DWT)
'''

def get_ppg_features_crossings(dataset, labels, waveletname):
    ppg_features = []
    for signal_no in range(0, len (dataset)):
        features = []
        signal = dataset[signal_no]
        list_coeff = pywt.wavedec(signal, waveletname, level=0)
        for coeff in list_coeff:
            features += get_features_crossings(coeff)
        ppg_features.append(features)
    x = np.array(ppg_features)
    y = np.array(labels)
    return x, y

'''
Dekomposisi sinyal untuk mengambil koefisien fitur statistical (DWT)
'''

def get_ppg_features_statistic(dataset, labels, waveletname):
    ppg_features = []
    for signal_no in range(0, len (dataset)):
        features = []
        signal = dataset[signal_no]
        list_coeff = pywt.wavedec(signal, waveletname, level=0)
        for coeff in list_coeff:
            features += get_features_statistic(coeff)
        ppg_features.append(features)
    x = np.array(ppg_features)
    y = np.array(labels)
    return x, y

def get_ppg_features_statistic1(dataset, waveletname):
    ppg_features = []
    for signal_no in range(0, len (dataset)):
        features = []
        signal = dataset[signal_no]
        list_coeff = pywt.wavedec(signal, waveletname, level=0)
        for coeff in list_coeff:
            features += get_features_statistic(coeff)
        ppg_features.append(features)
    x = np.array(ppg_features)
    return x

'''
Dekomposisi sinyal untuk mengambil koefisien fitur statistic dengan shanonn entropy (DWT)
'''

def get_ppg_features_statistic_entropy(dataset, labels, waveletname):
    ppg_features = []
    for signal_no in range(0, len (dataset)):
        features = []
        signal = dataset[signal_no]
        list_coeff = pywt.wavedec(signal, waveletname, level=0)
        for coeff in list_coeff:
            features += get_features_statistic_entropy(coeff)
        ppg_features.append(features)
    x = np.array(ppg_features)
    y = np.array(labels)
    return x, y

'''
Dekomposisi sinyal untuk mengambil koefisien fitur statistical dengan crossings (DWT)
'''

def get_ppg_features_statistic_crossings(dataset, labels, waveletname):
    ppg_features = []
    for signal_no in range(0, len (dataset)):
        features = []
        signal = dataset[signal_no]
        list_coeff = pywt.wavedec(signal, waveletname, level=0)
        for coeff in list_coeff:
            features += get_features_statistic_crossings(coeff)
        ppg_features.append(features)
    x = np.array(ppg_features)
    y = np.array(labels)
    return x, y

def get_ppg_features_crossings_entropy(dataset, labels, waveletname):
    ppg_features = []
    for signal_no in range(0, len (dataset)):
        features = []
        signal = dataset[signal_no]
        list_coeff = pywt.wavedec(signal, waveletname, level=0)
        for coeff in list_coeff:
            features += get_features_crossings_entropy(coeff)
        ppg_features.append(features)
    x = np.array(ppg_features)
    y = np.array(labels)
    return x, y

'''
Memberi label pada sinyal
'''

def annotation_to_ppg_signal(signal_path, annotation_path, number):
    signal_file = "%s/%s.csv" % (signal_path, number)
    anot_file = "%s/%s.csv" % (annotation_path, number)
    signal = pd.read_csv(signal_file, delimiter= ';', low_memory= False).replace(";",100)
    annotation = pd.read_csv(anot_file, delimiter= ';',low_memory=False)
    ppg = signal.iloc[1:,1].values
    ppg_signal = []
    start, stop, signal_class = annotation["indx1"].values, annotation["indx2"].values, annotation["event"].values.tolist()
    for i in range(len(annotation)):
            ppg_cut = ppg[start[i]:stop[i]].astype("float64").tolist()
            ppg_signal.append(ppg_cut)
    return ppg_signal, signal_class


def load_dataPPG(signal_path, number):
    signal_file = "%s/%s.csv" % (signal_path, number)
    signal = pd.read_csv(signal_file, delimiter= ';', low_memory= False).replace(";",100)
    ppg_signal = np.array(signal.iloc[2033:3890,1].astype("float64").tolist())
    np.save('%s.npy'%(number), ppg_signal)
    return ppg_signal

def load1_data(signal_path, number):
    signal_file = '%s\%s.csv' % (signal_path, number)
    df = pd.read_csv(signal_file)
    dataset = []
    ppg_signal = []
    for i in range(len(df)):
        data = df.loc[i].values.astype('float64')
        dataset.append(data)
    ppg_signal.append(dataset)
    return ppg_signal

'''
metrik uji
'''

def run_experiment(model, x_train, y_train, x_test, y_test, label):
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    
    plot_confusion_matrix(model, x_test, y_test, cmap='GnBu')
    plt.show()
    
# =============================================================================
#     precision = precision_score(y_test, y_pred, pos_label = 'AF')
#     recall = recall_score(y_test, y_pred, pos_label = 'AF')
#     f1 = f1_score(y_test, y_pred, pos_label='AF')
#     accuracy = accuracy_score(y_test, y_pred)
# =============================================================================
    print('Precision: %.3f' %precision_score(y_test, y_pred, average='macro'))
    print('Recall: %.3f' % recall_score(y_test, y_pred, average='macro'))
    print('F1: %.3f' % f1_score(y_test, y_pred, average='macro'))
    print('accuracy %.3f' %accuracy_score(y_test, y_pred))
    accuracy = 'selesai'
    return accuracy
# =============================================================================
#     print('Precision : %.3f' % precision_score(y_test, y_pred, pos_label='PVC'))
#     print('Recall : %.3f' % recall_score(y_test, y_pred, pos_label='PVC'))
#     print('F1 : %.3f' % f1_score(y_test, y_pred))
#     print('Accuracy : %.3f' % accuracy_score(y_test, y_pred))
# 
# =============================================================================
'''
split test train data
'''

def get_train_test(df, y_col, x_cols, ratio):
    df_train = df [mask]
    df_test = df[~mask]
    
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test

'''
list to csv
'''

def listtocsv(data, judul):
    np.savetxt(judul, data, delimiter =", ", fmt='%.3f')
    print(judul, 'added')
    
def listtocsv1(data, judul):
    np.savetxt(judul, data)
    print(judul, 'added')
    return

def find_signal_peaks(array_data, minimum=0, maximum=None, freq=500):
    dist = freq/2
    r_peaks = find_peaks(array_data, distance=dist, prominence=(minimum,maximum))
    return r_peaks[0].tolist()

def get_rr(r_peaks, to_sec=False,sample_rate=125):
    rr_list = []
    start_stop = []
    for i in range(len(r_peaks)-2):
        rr_list.append(r_peaks[i+1]-r_peaks[i])
        start_stop.append([r_peaks[i],r_peaks[i+1]])
    if (to_sec):
        rr_list = np.divide(rr_list,sample_rate)
    return rr_list, start_stop

def remove_signal(signal):
    return np.zeros_like(signal)

def remove_noise(noisy_signal):
    denoised = noisy_signal
    for i in range(5):
        denoised = pywt.dwt(denoised,'sym8')[0]
    for i in range(5):
        denoised = pywt.idwt(denoised,remove_signal(denoised),wavelet='sym8')
    return denoised

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def baseline_remove_ppg(signal,frequency=500):
    half_freq = int(frequency/2)
    baseline = resample(moving_average(signal,n=half_freq),len(signal))
    return np.subtract(signal,baseline)

def preprocess_ppg_signal(input_signal,sample_rate=500):
    baseline_removed = baseline_remove_ppg(input_signal,frequency=sample_rate)
    clean_ppg = remove_noise(baseline_removed)
    return clean_ppg

'''
ekstraksi fitur ppi/tdf
'''

def time_domain_features(signal_array,sample_rate=500 , preprocess=False):
    features = []
    i = 0
    for signal in signal_array:
        if (preprocess):
            signal = preprocess_ppg_signal(signal,sample_rate=sample_rate)
            
        peaks = find_signal_peaks(signal,minimum=0.2)
        rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=sample_rate)
        
        rata_ppi = np.mean(rr_list)
        std_ppi = np.std(rr_list)
        rata_sinyal = np.mean(signal)
        std_sinyal = np.std(signal)
        
        if(math.isnan(rata_ppi) or math.isnan(std_ppi) or math.isnan(rata_sinyal) or math.isnan(std_sinyal)):
          print("TRUE")
          print(i)
          rata_ppi = np.nansum(np.mean(rr_list))
          std_ppi = np.nansum(np.std(rr_list))
          rata_sinyal = np.nansum(np.mean(signal))
          std_sinyal = np.nansum(np.std(signal))
          features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal])
        else:
            features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal])
        i = i+1
    return features

def time_domain_features_entropy(signal_array,sample_rate=500 , preprocess=False):
    features = []
    i = 0
    for signal in signal_array:
        if (preprocess):
            signal = preprocess_ppg_signal(signal,sample_rate=sample_rate)
            
        peaks = find_signal_peaks(signal,minimum=0.2)
        rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=sample_rate)
        
        rata_ppi = np.mean(rr_list)
        std_ppi = np.std(rr_list)
        rata_sinyal = np.mean(signal)
        std_sinyal = np.std(signal)
        #shannon entropy ppi
        counter_values = Counter(rr_list).most_common()
        probabilities = [elem[1]/len(rr_list) for elem in counter_values]
        entropy = scipy.stats.entropy(probabilities)
        #shannon entropy sinyal
        counter_values_s = Counter(rr_list).most_common()
        probabilities_s = [elem[1]/len(rr_list) for elem in counter_values_s]
        entropy_s = scipy.stats.entropy(probabilities_s)
        
        if(math.isnan(rata_ppi) or math.isnan(std_ppi) or math.isnan(rata_sinyal) or math.isnan(std_sinyal)):
          print("TRUE")
          print(i)
          rata_ppi = np.nansum(np.mean(rr_list))
          std_ppi = np.nansum(np.std(rr_list))
          rata_sinyal = np.nansum(np.mean(signal))
          std_sinyal = np.nansum(np.std(signal))
          #shannon entropy ppi
          counter_values = Counter(rr_list).most_common()
          probabilities = [elem[1]/len(rr_list) for elem in counter_values]
          entropy=scipy.stats.entropy(probabilities)
          #shannon entropy sinyal
          counter_values_s= Counter(signal).most_common()
          probabilities_s = [elem[1]/len(signal) for elem in counter_values_s]
          entropy_s = scipy.stats.entropy(probabilities_s)
          features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal,entropy,entropy_s])
        else:
            features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal,entropy,entropy_s])
        i = i+1
    return features

def time_domain_features_entropy_crossing_stat(signal_array,sample_rate=500 , preprocess=False):
    features = []
    i = 0
    for signal in signal_array:
        if (preprocess):
            signal = preprocess_ppg_signal(signal,sample_rate=sample_rate)
            
        peaks = find_signal_peaks(signal,minimum=0.2)
        rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=sample_rate)
        
        rata_ppi = np.mean(rr_list)
        std_ppi = np.std(rr_list)
        rata_sinyal = np.mean(signal)
        std_sinyal = np.std(signal)
        median_ppi = np.nanpercentile(rr_list,50)
        var_ppi = np.nanvar(rr_list)
        var = np.nanvar(signal)
        median = np.nanpercentile(signal,50)
        #shannon entropy ppi
        counter_values = Counter(rr_list).most_common()
        probabilities = [elem[1]/len(rr_list) for elem in counter_values]
        entropy = scipy.stats.entropy(probabilities)
        #shannon entropy sinyal
        counter_values_s= Counter(signal).most_common()
        probabilities_s = [elem[1]/len(signal) for elem in counter_values_s]
        entropy_s = scipy.stats.entropy(probabilities_s)
        #level crossing ppi
        zero_crossing_indices = np.nonzero(np.diff(np.array(rr_list) > 0))[0]
        no_zero_crossings = len(zero_crossing_indices)
        mean_crossing_indices = np.nonzero(np.diff(np.array(rr_list) > np.nanmean(rr_list)))[0]
        no_mean_crossings = len(mean_crossing_indices)
        #level crossing sinyal
        zero_crossing_indices_s = np.nonzero(np.diff(np.array(signal) > 0))[0]
        no_zero_crossings_s = len(zero_crossing_indices_s)
        mean_crossing_indices_s = np.nonzero(np.diff(np.array(signal) > np.nanmean(signal)))[0]
        no_mean_crossings_s = len(mean_crossing_indices_s)
        
        if(math.isnan(rata_ppi) or math.isnan(std_ppi) or math.isnan(rata_sinyal) or math.isnan(std_sinyal)):
          print("TRUE")
          print(i)
          rata_ppi = np.nansum(np.mean(rr_list))
          std_ppi = np.nansum(np.std(rr_list))
          rata_sinyal = np.nansum(np.mean(signal))
          std_sinyal = np.nansum(np.std(signal))
          median_ppi = np.nansum(np.nanpercentile(rr_list,50))
          var_ppi = np.nansum(np.nanvar(rr_list))
          var = np.nansum(np.nanvar(signal))
          median = np.nansum(np.nanpercentile(signal,50))
          #shannon entropy ppi
          counter_values = Counter(rr_list).most_common()
          probabilities = [elem[1]/len(rr_list) for elem in counter_values]
          entropy=scipy.stats.entropy(probabilities)
          #shannon entropy sinyal
          counter_values_s= Counter(signal).most_common()
          probabilities_s = [elem[1]/len(signal) for elem in counter_values_s]
          entropy_s = scipy.stats.entropy(probabilities_s)
          #level crossing ppi
          zero_crossing_indices = np.nonzero(np.diff(np.array(rr_list) > 0))[0]
          no_zero_crossings = len(zero_crossing_indices)
          mean_crossing_indices = np.nonzero(np.diff(np.array(rr_list) > np.nanmean(rr_list)))[0]
          no_mean_crossings = len(mean_crossing_indices)
          #level crossing sinyal
          zero_crossing_indices_s = np.nonzero(np.diff(np.array(signal) > 0))[0]
          no_zero_crossings_s = len(zero_crossing_indices_s)
          mean_crossing_indices_s = np.nonzero(np.diff(np.array(signal) > np.nanmean(signal)))[0]
          no_mean_crossings_s = len(mean_crossing_indices_s)
          features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal,median_ppi,var_ppi,median,var,entropy,entropy_s,no_zero_crossings,no_mean_crossings,no_zero_crossings_s,no_mean_crossings_s])
        else:
            features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal,median_ppi,var_ppi,median,var,entropy,entropy_s,no_zero_crossings,no_mean_crossings,no_zero_crossings_s,no_mean_crossings_s])
        i = i+1
    return features

def time_domain_features_crossing(signal_array,sample_rate=500 , preprocess=False):
    features = []
    i = 0
    for signal in signal_array:
        if (preprocess):
            signal = preprocess_ppg_signal(signal,sample_rate=sample_rate)
            
        peaks = find_signal_peaks(signal,minimum=0.2)
        rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=sample_rate)
        
        rata_ppi = np.mean(rr_list)
        std_ppi = np.std(rr_list)
        rata_sinyal = np.mean(signal)
        std_sinyal = np.std(signal)
        #level crossing ppi
        zero_crossing_indices = np.nonzero(np.diff(np.array(rr_list) > 0))[0]
        no_zero_crossings = len(zero_crossing_indices)
        mean_crossing_indices = np.nonzero(np.diff(np.array(rr_list) > np.nanmean(rr_list)))[0]
        no_mean_crossings = len(mean_crossing_indices)
        #level crossing sinyal
        zero_crossing_indices_s = np.nonzero(np.diff(np.array(signal) > 0))[0]
        no_zero_crossings_s = len(zero_crossing_indices_s)
        mean_crossing_indices_s = np.nonzero(np.diff(np.array(signal) > np.nanmean(signal)))[0]
        no_mean_crossings_s = len(mean_crossing_indices_s)
        if(math.isnan(rata_ppi) or math.isnan(std_ppi) or math.isnan(rata_sinyal) or math.isnan(std_sinyal)):
          print("TRUE")
          print(i)
          rata_ppi = np.nansum(np.mean(rr_list))
          std_ppi = np.nansum(np.std(rr_list))
          rata_sinyal = np.nansum(np.mean(signal))
          std_sinyal = np.nansum(np.std(signal))
          #level crossing ppi
          zero_crossing_indices = np.nonzero(np.diff(np.array(rr_list) > 0))[0]
          no_zero_crossings = len(zero_crossing_indices)
          mean_crossing_indices = np.nonzero(np.diff(np.array(rr_list) > np.nanmean(rr_list)))[0]
          no_mean_crossings = len(mean_crossing_indices)
          #level crossing sinyal
          zero_crossing_indices_s = np.nonzero(np.diff(np.array(signal) > 0))[0]
          no_zero_crossings_s = len(zero_crossing_indices_s)
          mean_crossing_indices_s = np.nonzero(np.diff(np.array(signal) > np.nanmean(signal)))[0]
          no_mean_crossings_s = len(mean_crossing_indices_s)
          features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal,no_zero_crossings,no_mean_crossings,no_zero_crossings_s,no_mean_crossings_s])
        else:
            features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal,no_zero_crossings,no_mean_crossings,no_zero_crossings_s,no_mean_crossings_s])
        i = i+1
    return features
'''
plot sinyal
'''
def plot_color_text(filtered,predicted_beats,start_stop):
    minimum = min(filtered)
    maximum = max(filtered)
    lel = [np.arange(data[0],data[1]) for data in start_stop]
    #plt.plot(ecg_raw)
    plt.plot(filtered)
    for i in range(len(predicted_beats)):
        if (predicted_beats[i] != "N"):
            plt.fill_between(lel[i],minimum,maximum,facecolor='black', alpha=0.5)
            plt.text(start_stop[i][0],maximum,predicted_beats[i])
#    plt.scatter(peaks, [filtered[peaks[i]] for i in range(len(peaks))],c='red')
    plt.show()
    
def plot_with_rpeaks(filtered,r_peaks):
    peaks = [filtered[peak] for peak in r_peaks]
    plt.plot(filtered)
    plt.scatter(r_peaks,peaks,c='black')
#    plt.savefig("sinyal_clean_peaks",dpi=1000)
    plt.show()