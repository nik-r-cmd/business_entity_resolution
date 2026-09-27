"""Multilingual name-embedding similarity feature — targets records whose name is transliterated into a
different script (Kannada/Hindi/Punjabi/etc. vs Latin), where NO string-similarity feature can ever match,
since the two strings share zero characters. A pretrained sentence encoder embeds semantically-equivalent
names close together across scripts, with no fine-tuning needed (inference only)."""
import numpy as np


def add_embedding_feature(rec, df, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                           batch_size=256, device=None):
    """Adds an in-place 'name_embed_sim' column to df. rec: the record table from build_records.
    df: must have integer columns 'a','b' (row indices into rec). VERIFY the model's license tag on its
    Hugging Face model card yourself (challenge rule: MIT/Apache 2.0, <=8B params) before using it for real."""
    import torch
    from sentence_transformers import SentenceTransformer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] using device={device}")
    model = SentenceTransformer(model_name, device=device)
    a, b = df["a"].to_numpy(), df["b"].to_numpy()
    uniq = np.unique(np.concatenate([a, b]))
    texts = rec["business_name"].to_numpy()[uniq].tolist()   # RAW name (real script), not the Latin-normalised name_core
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    pos = {r: i for i, r in enumerate(uniq)}
    ai = np.fromiter((pos[x] for x in a), dtype=np.int64, count=len(a))
    bi = np.fromiter((pos[x] for x in b), dtype=np.int64, count=len(b))
    df["name_embed_sim"] = (embs[ai] * embs[bi]).sum(axis=1).astype(np.float32)   # cosine, since normalised
    return df
