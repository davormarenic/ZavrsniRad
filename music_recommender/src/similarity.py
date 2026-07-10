import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(reference_vector, database_vectors):

    try:
        # 1. Validacija ulaza (Proaktivan inženjerski korak)
        if reference_vector is None or database_vectors is None:
            raise ValueError("Vektori ne smiju biti None.")

        if len(database_vectors) == 0:
            raise ValueError("Baza vektora je prazna. Nema se s čime usporediti.")

        # 2. Provjera i čišćenje NaN/Inf vrijednosti
        # (Čest problem kod obrade tihih ili oštećenih audio zapisa)
        if np.isnan(reference_vector).any() or np.isnan(database_vectors).any():
            print("[WARNING] Detektirane NaN vrijednosti u vektorima. Normaliziram u nule.")
            reference_vector = np.nan_to_num(reference_vector)
            database_vectors = np.nan_to_num(database_vectors)

        # 3. Prilagodba oblika (Reshaping) za scikit-learn
        # Sklearn očekuje 2D niz oblika (broj_uzoraka, broj_značajki)
        ref_2d = reference_vector.reshape(1, -1)

        # 4. Izračun kosinusne sličnosti
        # cosine_similarity vraća matricu (1, M), uzimamo prvi redak [0] da dobijemo 1D niz
        scores = cosine_similarity(ref_2d, database_vectors)[0]

        # 5. Ograničavanje vrijednosti (Clipping)
        # Zbog "floating-point" grešaka računala, rezultat može biti npr. 1.0000000000000002
        # Ograničavamo striktno na [-1.0, 1.0]
        scores = np.clip(scores, -1.0, 1.0)

        return scores

    except Exception as e:
        print(f"[ERROR] Greška pri izračunu kosinusne sličnosti: {e}")
        # U slučaju kritične greške, vraćamo niz nula kako ne bismo srušili main.py
        return np.zeros(len(database_vectors)) if database_vectors is not None else np.array([])