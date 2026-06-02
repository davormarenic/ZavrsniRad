import os
import pandas as pd
import time
from src.audio_processor import extract_features, get_feature_names

def build_database(data_path, output_csv):
    print(f"\n[INFO] Započinjem izgradnju baze podataka iz mape: '{data_path}'")
    if not os.path.exists(data_path):
        print(f"[ERROR] Direktorij '{data_path}' ne postoji!")
        return False
    wav_files = [f for f in os.listdir(data_path) if f.lower().endswith('.wav')]
    total_files = len(wav_files)
    if total_files == 0:
        print(f"[WARNING] Nisu pronađene .wav datoteke u mapi '{data_path}'.")
        return False
    data_list = []
    successful_extractions = 0
    start_time = time.time()
    for index, file in enumerate(wav_files, start=1):
        file_path = os.path.join(data_path, file)
        print(f"[{index}/{total_files}] Obrađujem: {file} ...", end=" ", flush=True)
        features = extract_features(file_path)
        if features is not None:
            data_list.append([file] + features.tolist())
            successful_extractions += 1
            print("OK")
        else:
            print("FAILED")
    if successful_extractions == 0:
        print("[ERROR] Niti jedna datoteka nije uspješno obrađena. Baza nije kreirana.")
        return False
    columns = ['filename'] + get_feature_names()
    df = pd.DataFrame(data_list, columns=columns)
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Kreiran izlazni direktorij: '{output_dir}'")
    try:
        df.to_csv(output_csv, index=False)
        elapsed_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"[SUCCESS] Baza uspješno kreirana: {output_csv}")
        print(f"[STATISTIKA] Obrađeno {successful_extractions}/{total_files} datoteka.")
        print(f"[STATISTIKA] Vrijeme izvršavanja: {elapsed_time:.2f} sekundi.")
        print("="*50)
        return True
    except Exception as e:
        print(f"\n[ERROR] Neuspjelo zapisivanje CSV datoteke: {e}")
        return False