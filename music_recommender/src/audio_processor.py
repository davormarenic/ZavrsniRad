import librosa
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        y, _ = librosa.effects.trim(y, top_db=60)
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        tempo_val = float(tempo[0])
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        feature_vector = np.hstack(([energy_mean, tempo_val], mfccs_mean))
        return feature_vector.astype(np.float32)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Datoteka nije pronađena: {file_path}")
        return None
    except Exception as e:
        print(f"ERROR: Neuspjela obrada datoteke {file_path}. Detalji: {e}")
        return None

def get_feature_names():
    names = ['energy', 'tempo']
    names.extend([f'mfcc_{i+1}' for i in range(13)])
    return names