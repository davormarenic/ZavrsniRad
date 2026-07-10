import os
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from src import extract_features, calculate_cosine_similarity, build_database

def main():
    db_path = 'output/music_database.csv'
    data_dir = 'data/'

    print("=== Sustav za preporuku glazbe (DSS) ===")

    # 1. Prvo provjeravamo postoji li baza uopće
    if not os.path.exists(db_path):
        print(f"[INFO] Baza podataka nije pronađena na {db_path}.")
        confirm = input(f"Želite li sada generirati bazu iz mape '{data_dir}'? (d/n): ")
        if confirm.lower() == 'd':
            build_database(data_dir, db_path)
        else:
            print("Sustav se gasi. Baza je neophodna za rad.")
            return

    try:
        # 2. Učitavanje podataka
        df = pd.read_csv(db_path)

        # --- PROAKTIVNI INŽENJERSKI KORAK: Provjera sinkronizacije ---
        # Brojimo .wav datoteke u mapi
        wav_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.wav')]

        # Uspoređujemo s brojem pjesama u bazi
        if len(wav_files) != len(df):
            print(f"\n[UPOZORENJE] Detektirana je nesukladnost!")
            print(f"Broj pjesama u bazi: {len(df)}")
            print(f"Broj pjesama u mapi '{data_dir}': {len(wav_files)}")

            update = input("Želite li ažurirati bazu podataka s novim pjesmama? (d/n): ")
            if update.lower() == 'd':
                # Ponovno gradimo bazu (prepisujemo stari CSV)
                build_database(data_dir, db_path)
                # Ponovno učitavamo svježu bazu
                df = pd.read_csv(db_path)
            else:
                print("[INFO] Nastavljam rad sa starom bazom podataka.\n")
        # -------------------------------------------------------------

        if df.empty:
            print("[ERROR] Pogreška: CSV baza je prazna.")
            return

        filenames = df['filename'].values
        features_matrix = df.drop(columns=['filename']).values

        # 3. Normalizacija (Min-Max skaliranje)
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(features_matrix)

        # 4. Unos referentne pjesme [cite: 2799, 2954]
        print("\n--- Unos referentne pjesme ---")
        test_song_path = input("Povucite .wav datoteku ovdje ili unesite putanju: ").strip().replace("'", "").replace('"', "")

        if not os.path.exists(test_song_path):
            print(f"Pogreška: Datoteka '{test_song_path}' ne postoji.")
            return

        print(f"Analiziram: {os.path.basename(test_song_path)}...")
        ref_features = extract_features(test_song_path)

        if ref_features is not None:
            # Normalizacija referentnog vektora istim scalerom [cite: 2953]
            ref_normalized = scaler.transform(ref_features.reshape(1, -1))

            # 5. Računanje kosinusne sličnosti [cite: 2881, 2919]
            scores = calculate_cosine_similarity(ref_normalized, features_normalized)

            # Dodavanje rezultata u DataFrame za lakše sortiranje
            df['similarity'] = scores

            # 6. Prikaz rezultata (Top 5 najsličnijih) [cite: 2886, 2912, 2956]
            # Sortiramo od najveće sličnosti prema najmanjoj
            # head(6) jer će prva pjesma često biti ona ista ako je u bazi
            recommendations = df.sort_values(by='similarity', ascending=False).head(6)

            print("\n" + "="*40)
            print(f"TOP 5 PREPORUKA ZA: {os.path.basename(test_song_path)}")
            print("="*40)

            count = 1
            for _, row in recommendations.iterrows():
                # Preskačemo pjesmu ako je identična ulaznoj (similarity ~ 1.0)
                if row['filename'] == os.path.basename(test_song_path):
                    continue
                if count > 5: break

                percentage = row['similarity'] * 100
                print(f"{count}. {row['filename']} | Sličnost: {percentage:.2f}%")
                count += 1
            print("="*40)

    except Exception as e:
        print(f"Došlo je do neočekivane pogreške: {e}")

if __name__ == "__main__":
    main()