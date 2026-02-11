import numpy as np
import pandas as pd
from joblib import load
from scipy import sparse
from pathlib import Path

from config import (
    ARTIFACTS_DIR, META_PATH,
    VECTORIZER_VEC_PATH, VECTORS_VEC_PATH,
    VECTORIZER_KW_PATH,  VECTORS_KW_PATH,
    CANDIDATE_K, TOP_K, ALPHA,
    EVAL_QUERIES_PATH, RUN_PATH,
)
from utils import normalize_text


def topk_indices_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.size)
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def cosine_scores(qv, X):
    return (X @ qv.T).toarray().ravel()


def retrieve_hybrid(query, vectorizer_kw, X_kw, vectorizer_vec, X_vec, candidate_k, top_k, alpha):
    qn = normalize_text(query)

    q_kw = vectorizer_kw.transform([qn])
    kw_scores_all = cosine_scores(q_kw, X_kw)
    cand_idx = topk_indices_from_scores(kw_scores_all, candidate_k)
    if cand_idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    kw_scores_cand = kw_scores_all[cand_idx]

    q_vec = vectorizer_vec.transform([qn])
    X_cand = X_vec[cand_idx]
    vec_scores = cosine_scores(q_vec, X_cand)

    fusion_scores = alpha * vec_scores + (1.0 - alpha) * kw_scores_cand

    top_local = topk_indices_from_scores(fusion_scores, top_k)
    top_idx = cand_idx[top_local]
    top_scores = fusion_scores[top_local]
    return top_idx, top_scores


def main():
    # artifacts
    vectorizer_kw = load(VECTORIZER_KW_PATH)
    X_kw = sparse.load_npz(VECTORS_KW_PATH)
    vectorizer_vec = load(VECTORIZER_VEC_PATH)
    X_vec = sparse.load_npz(VECTORS_VEC_PATH)
    meta = pd.read_parquet(META_PATH)

    # queries
    qdf = pd.read_csv(EVAL_QUERIES_PATH)
    if not {"qid", "query"}.issubset(qdf.columns):
        raise ValueError("eval_queries.csv must have columns: qid, query")

    rows = []
    for _, qr in qdf.iterrows():
        qid = str(qr["qid"])
        query = str(qr["query"])

        idxs, scores = retrieve_hybrid(
            query=query,
            vectorizer_kw=vectorizer_kw, X_kw=X_kw,
            vectorizer_vec=vectorizer_vec, X_vec=X_vec,
            candidate_k=CANDIDATE_K,
            top_k=TOP_K,
            alpha=float(ALPHA),
        )

        for rank, (idx, score) in enumerate(zip(idxs, scores), start=1):
            m = meta.iloc[int(idx)].to_dict() if int(idx) < len(meta) else {}
            posting_id = m.get("posting_id", m.get("row_index", int(idx)))
            rows.append({
                "qid": qid,
                "posting_id": str(posting_id),
                "rank": int(rank),
                "score": float(score),
            })

    Path(RUN_PATH).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RUN_PATH, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved run: {RUN_PATH} (rows={len(rows)})")
    print(f"- candidate_k={CANDIDATE_K}, alpha={ALPHA}, top_k={TOP_K}")
    print(f"- artifacts: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
