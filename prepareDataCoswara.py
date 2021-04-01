import os
import librosa
import math
import json
import pdb
import csv
import numpy as np
import pickle

DATASET_PATH = 'covid_shared_data/cough_dataset.csv'
JSON_PATH = 'data.json'
SAMPLE_RATE=22050

def extract_features(file_name, n_mfcc=13, hop_length=512, n_fft=2048):
        try:
            signal, sr = librosa.load(file_name, sr=SAMPLE_RATE)
            # extract the MFCCs  
            mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                            hop_length=hop_length)
        except Exception as e:
            print("Error encountered while parsing file: ", e)
            return None, None

        return mfccs, sr

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):

    dataPackage = {
        "MFCCs": [],
        "labels": [],
    }

    features = []
    with open(DATASET_PATH) as dataset:
            csv_reader = csv.reader(dataset, delimiter=',')
            index = 1

            for row in csv_reader:
                file_properties = row[0]
                file_name = os.getcwd()+'/covid_shared_data/audio/'+file_properties
                class_label = row[1]
                data, sr = extract_features(file_name)
                #print(data)
                if data is not None:
                    dataPackage['MFCCs'].append(data.T.tolist())
                    dataPackage['labels'].append(class_label)

                else:
                    print("Data is empty: ", file_name)

                print("Processed row ", index)
                index = index+1
    
    with open(json_path, "w") as fp:
        json.dump(dataPackage, fp, indent=4)

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)