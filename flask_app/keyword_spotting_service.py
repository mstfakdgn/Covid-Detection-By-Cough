import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = 'model.h5'
NUM_SAMPLES_TO_CONSIDER = 22050 # 1 second

class _Keyword_Spotting_Service:

    model = None
    _mappings_ = [
        "covid",
        "not_covid"
    ]
    _instance = None

    def predict(self, file_path):

        model = keras.models.load_model('model.h5')

        signal, sr = librosa.load(file_path)
        
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

        y_pred = self.model.predict(MFCCs)
        predicted_index = np.argmax(y_pred[:, 0:2])
        predicted_keyword = self._mappings_[predicted_index]
        return  predicted_keyword

    def preprocess(self, file_path, n_fft=2048, n_mfcc=13, hop_length=512):
        
        # laod audio file
        signal, sr = librosa.load(file_path)

        # ensure consistncy in audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCC
        MFCCs = librosa.feature.mfcc(signal, sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)

        return MFCCs.T

def Keyword_Spotting_Service():

    # ensure that we only have one instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)

    return _Keyword_Spotting_Service._instance

if __name__ == "__main__":
    kss = Keyword_Spotting_Service()

    keyword1 = kss.predict("../covid_shared_data/audio/_-_5kbw2Mcw_ 0.000_ 10.000.wav")
    keyword2 = kss.predict("../covid_shared_data/audio/HYGtUoMDukOlfkkQ7rgRPhuKorA3-breathing-deep.wav")

    print(f"Predicted: {keyword1}, {keyword2}")