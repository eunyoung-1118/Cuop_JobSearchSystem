import numpy as np
import pandas as pd
from joblib import load
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from config import ARTIFACTS_DIR, VECTORIZER_PATH, VECTORS_PATH, META_PATH, TOP_K
from utils import normalize_text

def main():
    # artifacts 존재 확인
    if not VECTORIZER_PATH.exists() or not VECTORS_PATH.exists() or not META_PATH.exists():
        print("[ERROR] artifacts가 없습니다. 먼저 인덱스를 생성하세요:")
        print("  python src/build_index.py")
        return

    vectorizer = load(VECTORIZER_PATH) # 쿼리 벡터화 기준
    X = sparse.load_npz(VECTORS_PATH) # 공고 벡터들 (인덱스)
    meta = pd.read_parquet(META_PATH) # 출력용 정보

    print("채용공고 검색 시스템 (종료: exit / quit)")
    print(f"artifacts: {ARTIFACTS_DIR}\n")

    while True:
        q = input("검색어> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        qn = normalize_text(q)
        qv = vectorizer.transform([qn]) # 쿼리 벡터화
        scores = cosine_similarity(qv, X).ravel() # 유사도 계산

        topk = np.argsort(-scores)[:TOP_K] # 점수가 큰 순서대로 인덱스 추출

        print("\n=== TOP {} ===".format(TOP_K))
        for rank, idx in enumerate(topk, start=1):
            score = float(scores[idx])

            row = meta.iloc[idx].to_dict() if idx < len(meta) else {"row_index": idx}
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
