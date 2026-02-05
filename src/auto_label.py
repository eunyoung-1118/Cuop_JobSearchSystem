import os
import time
import argparse
import pandas as pd
import numpy as np
from joblib import load
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

from openai import OpenAI

from config import CSV_PATH, VECTORIZER_PATH, VECTORS_PATH, META_PATH, TEXT_COLS
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

    # 숫자 하나만 파싱
    for ch in txt:
        if ch in "012":
            if sleep_s > 0:
                time.sleep(sleep_s)
            return int(ch)

    raise ValueError(f"LLM output parse failed: {txt}")


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

    # API 키 확인
    require_api_key()
    client = OpenAI()

    # artifacts 로드
    vectorizer = load(VECTORIZER_PATH)
    X = sparse.load_npz(VECTORS_PATH)
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

    # (posting_id, query)별 캐시: 같은 공고가 여러 번 나오면 비용 줄임
    label_cache = {}

    for _, qr in qdf.iterrows():
        qid = str(qr["qid"])
        query = str(qr["query"])

        qv = vectorizer.transform([normalize_text(query)])
        scores = cosine_similarity(qv, X).ravel()
        top_idx = np.argsort(-scores)[: args.top_n]

        for rank, idx in enumerate(top_idx, start=1):
            m = meta.iloc[idx].to_dict() if idx < len(meta) else {"row_index": idx}
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
                "score": float(scores[idx]),
                "relevance": int(rel),
            })

        # 중간 저장
        pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved eval dataset: {args.out} (rows={len(rows)})")
    print("→ 평가 실행: python src/evaluate.py")


if __name__ == "__main__":
    main()
