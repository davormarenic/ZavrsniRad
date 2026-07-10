import librosa
import numpy as np
import warnings

# Ignoriramo specifična upozorenja Librose oko PySoundFile-a radi čišćeg ispisa u konzoli
warnings.filterwarnings("ignore", category=UserWarning)

def extract_features(file_path):
    try:
        # 1. Učitavanje i Preprocessing
        # Koristimo sr=22050 (standard za analizu) radi konzistentnosti vektora značajki
        y, sr = librosa.load(file_path, sr=22050, mono=True)

        # Uklanjanje tišine s početka i kraja (trimming) - bitno za točan izračun energije.
        # Eksplicitno postavljamo top_db=60 što znači da se reže sve što je 60dB tiše od najglasnijeg dijela.
        y, _ = librosa.effects.trim(y, top_db=60)

        # 2. Energija signala (RMS - Root Mean Square)
        # RMS nam daje informaciju o glasnoći/percipiranoj snazi pjesme
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)

        # 3. Tempo (BPM - Beats Per Minute)
        # Librosa računa tempo pomoću analize onset-a (udaraca)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        tempo_val = float(tempo[0])

        # 4. MFCC (Mel-Frequency Cepstral Coefficients)
        # MFCC su ključni za opisivanje "boje" instrumentacije i vokala.
        # Uzimamo 13 koeficijenata koji su standard u glazbenoj informacijskoj znanosti (MIR).
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Računamo prosjek za svaki od 13 koeficijenata kroz cijelu pjesmu
        # axis=1 znači da agregiramo vremensku komponentu
        mfccs_mean = np.mean(mfccs, axis=1)

        # 5. Agregacija u Vektor Značajki (Feature Vector)
        # Vektor se sastoji od: [energija, tempo, mfcc_1, ..., mfcc_13] -> Ukupno 15 elemenata
        feature_vector = np.hstack(([energy_mean, tempo_val], mfccs_mean))

        # Eksplicitna konverzija u float32 radi uštede memorije i kompatibilnosti
        return feature_vector.astype(np.float32)

    except FileNotFoundError:
        print(f"CRITICAL ERROR: Datoteka nije pronađena: {file_path}")
        return None
    except Exception as e:
        print(f"ERROR: Neuspjela obrada datoteke {file_path}. Detalji: {e}")
        return None

def get_feature_names():
    """Pomoćna funkcija za imenovanje stupaca u bazi podataka."""
    names = ['energy', 'tempo']
    names.extend([f'mfcc_{i+1}' for i in range(13)])
    return names