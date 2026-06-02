import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(reference_vector, database_vectors):
    try:
        if reference_vector is None or database_vectors is None:
            raise ValueError("Vektori ne smiju biti None.")
        if len(database_vectors) == 0:
            raise ValueError("Baza vektora je prazna. Nema se s čime usporediti.")
        if np.isnan(reference_vector).any() or np.isnan(database_vectors).any():
            print("[WARNING] Detektirane NaN vrijednosti u vektorima. Normaliziram u nule.")
            reference_vector = np.nan_to_num(reference_vector)
            database_vectors = np.nan_to_num(database_vectors)
        ref_2d = reference_vector.reshape(1, -1)
        scores = cosine_similarity(ref_2d, database_vectors)[0]
        scores = np.clip(scores, -1.0, 1.0)
        return scores
    except Exception as e:
        print(f"[ERROR] Greška pri izračunu kosinusne sličnosti: {e}")
        return np.zeros(len(database_vectors)) if database_vectors is not None else np.array([])