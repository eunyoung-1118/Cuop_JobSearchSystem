import os
import time
import argparse
import pandas as pd
import numpy as np
from joblib import load
from scipy import sparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

from openai import OpenAI

from config import (
    CSV_PATH, META_PATH, TEXT_COLS,
    VECTORIZER_VEC_PATH, VECTORS_VEC_PATH,
    VECTORIZER_KW_PATH,  VECTORS_KW_PATH,
    CANDIDATE_K,
    ALPHA,
)
from utils import normalize_text, safe_read_csv


DEFAULT_EVAL_QUERIES = "eval/eval_queries.csv"
DEFAULT_OUT = "eval/eval_dataset.csv"


def require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY 환경변수가 없습니다.\n"
            "PowerShell에서 다음처럼 설정하고 다시 실행하세요:\n"
            '  $env:OPENAI_API_KEY="your_key_here"\n'
            "또는 영구 설정:\n"
            '  setx OPENAI_API_KEY "your_key_here"\n'
        )
    return key


def build_jd_text(row: pd.Series, text_cols: list[str], max_chars: int = 6000) -> str:
    parts = []
    for c in text_cols:
        v = row.get(c, "")
        if pd.isna(v):
            v = ""
        v = str(v).strip()
        if v:
            parts.append(f"[{c}]\n{v}")
    jd = "\n\n".join(parts)
    if len(jd) > max_chars:
        jd = jd[:max_chars] + "\n\n...(truncated)"
    return jd


def openai_label(client, query: str, jd_text: str, model: str, sleep_s: float = 0.0) -> int:
    prompt = f"""
너는 채용공고 검색/추천 시스템의 평가자다.
아래 '쿼리'에 대해 '채용공고'가 얼마나 관련 있는지 relevance를 0/1/2 중 하나로만 판단해라.

[채점 기준]
2 = 매우 관련: 쿼리 의도에 직접적으로 부합. 스택/역할/요구가 명확히 맞음.
1 = 관련: 일부는 맞지만 애매하거나 부분 일치(우대/보조 역할 등).
0 = 무관: 쿼리와 거의 관계 없음.

[출력 형식]
숫자 하나만 출력 (0 또는 1 또는 2)

[쿼리]
{query}

[채용공고]
{jd_text}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    txt = resp.choices[0].message.content.strip()

    for ch in txt:
        if ch in "012":
            if sleep_s > 0:
                time.sleep(sleep_s)
            return int(ch)

    raise ValueError(f"LLM output parse failed: {txt}")


def topk_indices_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.size)
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def cosine_scores(qv, X):
    # TF-IDF 기본 norm='l2'라 dot == cosine
    return (X @ qv.T).toarray().ravel()


def hybrid_retrieve_indices(
    query: str,
    top_n: int,
    vectorizer_kw, X_kw,
    vectorizer_vec, X_vec,
    candidate_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    return:
      top_idx: 원본 문서 인덱스 (len=top_n)
      top_scores: 2차(char) 점수 (len=top_n)
    """
    qn = normalize_text(query)

    # 1차: word 후보
    q_kw = vectorizer_kw.transform([qn])
    kw_scores = cosine_scores(q_kw, X_kw)
    cand_idx = topk_indices_from_scores(kw_scores, candidate_k)

    if cand_idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    # 2차: char rerank
    q_vec = vectorizer_vec.transform([qn])
    X_cand = X_vec[cand_idx]
    vec_scores = cosine_scores(q_vec, X_cand)

    # fusion score
    kw_scores_cand = kw_scores[cand_idx]
    alpha = float(ALPHA)
    fusion_scores = alpha * vec_scores + (1.0 - alpha) * kw_scores_cand

    top_local = topk_indices_from_scores(fusion_scores, top_n)
    top_idx = cand_idx[top_local]
    top_scores = fusion_scores[top_local]
    return top_idx, top_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=DEFAULT_EVAL_QUERIES, help="eval/eval_queries.csv 경로")
    ap.add_argument("--out", default=DEFAULT_OUT, help="eval/eval_dataset.csv 출력 경로")
    ap.add_argument("--top_n", type=int, default=20, help="쿼리당 TopN (10쿼리면 200개)")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    ap.add_argument("--max_chars", type=int, default=6000, help="JD 텍스트 최대 길이")
    ap.add_argument("--sleep", type=float, default=0.0, help="호출 간 sleep(초)")
    ap.add_argument("--resume", action="store_true", help="중간에 끊겨도 이어서 실행")
    args = ap.parse_args()

    require_api_key()
    client = OpenAI()

    # artifacts 로드 (v0.2)
    vectorizer_kw = load(VECTORIZER_KW_PATH)
    X_kw = sparse.load_npz(VECTORS_KW_PATH)

    vectorizer_vec = load(VECTORIZER_VEC_PATH)
    X_vec = sparse.load_npz(VECTORS_VEC_PATH)

    meta = pd.read_parquet(META_PATH)

    # 쿼리 로드
    qdf = pd.read_csv(args.queries)
    if not {"qid", "query"}.issubset(set(qdf.columns)):
        raise ValueError("eval_queries.csv는 반드시 qid,query 컬럼이 있어야 합니다.")

    # 원본 CSV 로드 (JD 텍스트용)
    df = safe_read_csv(CSV_PATH)
    has_posting_id = "posting_id" in df.columns

    # resume 로드
    rows = []
    done = set()
    if args.resume and os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        rows = prev.to_dict("records")
        for _, r in prev.iterrows():
            rel = str(r.get("relevance", "")).strip()
            if rel != "":
                done.add((str(r["qid"]), str(r["posting_id"])))
        print(f"[RESUME] loaded {len(prev)} rows, done={len(done)}")

    label_cache = {}

    for _, qr in qdf.iterrows():
        qid = str(qr["qid"])
        query = str(qr["query"])

        top_idx, top_scores = hybrid_retrieve_indices(
            query=query,
            top_n=args.top_n,
            vectorizer_kw=vectorizer_kw, X_kw=X_kw,
            vectorizer_vec=vectorizer_vec, X_vec=X_vec,
            candidate_k=CANDIDATE_K,
        )

        for rank, (idx, score2) in enumerate(zip(top_idx, top_scores), start=1):
            m = meta.iloc[idx].to_dict() if idx < len(meta) else {"row_index": int(idx)}
            posting_id = m.get("posting_id", m.get("row_index", idx))
            key = (qid, str(posting_id))

            if args.resume and key in done:
                continue

            # JD row 찾기
            if has_posting_id:
                hit = df[df["posting_id"].astype(str) == str(posting_id)]
                if len(hit) == 0:
                    row_index = m.get("row_index", idx)
                    jd_row = df.iloc[int(row_index)] if 0 <= int(row_index) < len(df) else pd.Series({})
                else:
                    jd_row = hit.iloc[0]
            else:
                row_index = m.get("row_index", idx)
                jd_row = df.iloc[int(row_index)] if 0 <= int(row_index) < len(df) else pd.Series({})

            jd_text = build_jd_text(jd_row, TEXT_COLS, max_chars=args.max_chars)

            cache_key = (str(posting_id), query)
            if cache_key in label_cache:
                rel = label_cache[cache_key]
            else:
                rel = openai_label(client, query=query, jd_text=jd_text, model=args.model, sleep_s=args.sleep)
                label_cache[cache_key] = rel

            rows.append({
                "qid": qid,
                "query": query,
                "rank": rank,
                "posting_id": posting_id,
                "score": float(score2),   # 2차(char) 점수 저장
                "relevance": int(rel),
            })

        pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved eval dataset: {args.out} (rows={len(rows)})")
    print("→ 평가 실행: python src/evaluate.py")


if __name__ == "__main__":
    main()
