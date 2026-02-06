import numpy as np
import pandas as pd
from joblib import load
from scipy import sparse

from config import (
    ARTIFACTS_DIR, META_PATH, TOP_K,
    VECTORIZER_VEC_PATH, VECTORS_VEC_PATH,
    VECTORIZER_KW_PATH,  VECTORS_KW_PATH,
    CANDIDATE_K,
)
from utils import normalize_text

def topk_indices_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.size)
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, kth=k-1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx

def cosine_scores(qv, X):
    return (X @ qv.T).toarray().ravel()

def main():
    needed = [
        VECTORIZER_KW_PATH, VECTORS_KW_PATH,
        VECTORIZER_VEC_PATH, VECTORS_VEC_PATH,
        META_PATH
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print("[ERROR] artifacts가 없습니다. 먼저 인덱스를 생성하세요:")
        print("  python src/build_index_v02.py")
        for p in missing:
            print(" - missing:", p)
        return

    vectorizer_kw = load(VECTORIZER_KW_PATH)
    X_kw = sparse.load_npz(VECTORS_KW_PATH)

    vectorizer_vec = load(VECTORIZER_VEC_PATH)
    X_vec = sparse.load_npz(VECTORS_VEC_PATH)

    meta = pd.read_parquet(META_PATH)

    print("하이브리드 채용공고 검색 (종료: exit / quit)")
    print(f"artifacts: {ARTIFACTS_DIR}")
    print(f"1차 후보: {CANDIDATE_K}, 최종 TOP_K: {TOP_K}\n")

    while True:
        q = input("검색어> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        qn = normalize_text(q)

        # 1) keyword 후보 축소 (word TF-IDF)
        q_kw = vectorizer_kw.transform([qn])
        kw_scores = cosine_scores(q_kw, X_kw)
        cand_idx = topk_indices_from_scores(kw_scores, CANDIDATE_K)

        if cand_idx.size == 0:
            print("후보가 없습니다.\n")
            continue

        # 2) vector 재랭킹 (char TF-IDF)
        q_vec = vectorizer_vec.transform([qn])
        X_cand = X_vec[cand_idx]
        vec_scores = cosine_scores(q_vec, X_cand)

        # 최종 top_k
        top_local = topk_indices_from_scores(vec_scores, TOP_K)
        top_idx = cand_idx[top_local]

        print("\n=== TOP {} (hybrid) ===".format(TOP_K))
        for rank, idx in enumerate(top_idx, start=1):
            score = float(vec_scores[top_local[rank-1]])  # 2차 점수

            row = meta.iloc[idx].to_dict() if idx < len(meta) else {"row_index": int(idx)}
            title = row.get("job_title", "")
            pid = row.get("posting_id", "")
            exp = row.get("experience_Level", "")
            loc = row.get("location", "")
            period = row.get("posting_period", "")
            url = row.get("url", "")

            print(f"{rank:2d}. ({score:.4f}) {title} | {pid} | {exp} | {loc} | {period}")
            if isinstance(url, str) and url.strip():
                print(f"    {url}")
        print()

if __name__ == "__main__":
    main()
