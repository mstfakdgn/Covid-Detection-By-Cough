import requests
import tensorflow.keras as keras
import librosa
import numpy as np

# URL = "http://127.0.0.1:5050/predict"
URL = "http://127.0.0.1/predict"

TEST_AUDIO_FILE_TEST = "../covid_shared_data/audio/cough-shallow-6T43bddKoKfG7MwnJWvrPZSsyrc2.wav"
mappings = [
        "covid",
        "not_covid"
]

if __name__ == "__main__":

    model = keras.models.load_model('model.h5')

    signal, sr = librosa.load(TEST_AUDIO_FILE_TEST)
    
    # extract MFCC
    MFCCs = librosa.feature.mfcc(signal, sr, n_fft=2048, n_mfcc=13, hop_length=512)
    MFCCs = MFCCs.T
    
    if np.array(MFCCs).shape[0] < 431:
        zeros = np.zeros([431,13])
        zeros[:np.array(MFCCs).shape[0], :np.array(MFCCs).shape[1]] = np.array(MFCCs)
        MFCCs = zeros
    elif np.array(MFCCs).shape[0] > 431:
        mfccArray = np.array(MFCCs)
        MFCCs = mfccArray[:431, :13]

    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

    y_pred = model.predict(MFCCs)
    predicted_index = np.argmax(y_pred[:, 0:2])
    predicted_keyword = mappings[predicted_index]
    print(predicted_keyword)
    # data = y_pred.json()

    # print(f"Predicted Keyword is: {data['key']}")