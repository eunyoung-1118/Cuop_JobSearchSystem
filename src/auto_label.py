import os
import time
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from config import CSV_PATH, TEXT_COLS
from utils import safe_read_csv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


def require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY 환경변수가 없습니다.\n"
            "PowerShell에서 다음처럼 설정하고 다시 실행하세요:\n"
            '  $env:OPENAI_API_KEY="your_key_here"\n'
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/eval_queries.csv", help="qid,query CSV")
    ap.add_argument("--out", default="eval/qrels.csv", help="output qrels path")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max_chars", type=int, default=6000)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true", help="중단 후 이어서")
    ap.add_argument("--save_every", type=int, default=200, help="중간 저장 주기(행 수)")
    args = ap.parse_args()

    require_api_key()
    client = OpenAI()

    # queries
    qdf = pd.read_csv(args.queries)
    if not {"qid", "query"}.issubset(qdf.columns):
        raise ValueError("eval_queries.csv는 반드시 qid,query 컬럼이 있어야 합니다.")
    qdf["qid"] = qdf["qid"].astype(str)
    qdf["query"] = qdf["query"].astype(str)
    qids = qdf["qid"].tolist()
    qmap = dict(zip(qdf["qid"], qdf["query"]))

    # postings
    df = safe_read_csv(CSV_PATH)
    if "posting_id" not in df.columns:
        raise ValueError("CSV에 posting_id 컬럼이 필요합니다.")
    df["posting_id"] = df["posting_id"].astype(str)

    # dedup + index
    df = df.drop_duplicates(subset=["posting_id"], keep="first")
    jd_by_pid = df.set_index("posting_id")
    posting_ids = jd_by_pid.index.tolist()

    total_pairs = len(qids) * len(posting_ids)
    print(f"[POOL] all pairs = {total_pairs} (q={len(qids)} x docs={len(posting_ids)})")

    # out dir ensure
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # resume
    done = set()
    rows = []
    if args.resume and os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        rows = prev.to_dict("records")
        for _, r in prev.iterrows():
            done.add((str(r["qid"]), str(r["posting_id"])))
        print(f"[RESUME] loaded {len(prev)} qrels, done={len(done)}")

    # cache jd_text (posting_id당 1번만 생성)
    print("[PREP] caching jd_text...")
    jd_text_by_pid = {
        pid: build_jd_text(jd_by_pid.loc[pid], TEXT_COLS, max_chars=args.max_chars)
        for pid in posting_ids
    }
    print(f"[PREP] cached {len(jd_text_by_pid)} postings")

    # label cache (같은 쿼리+공고는 1번만 호출)
    label_cache = {}

    for qid in qids:
        query = qmap[qid]
        for pid in posting_ids:
            key = (qid, pid)
            if args.resume and key in done:
                continue

            jd_text = jd_text_by_pid.get(pid)
            if jd_text is None:
                continue

            cache_key = (pid, query)
            if cache_key in label_cache:
                rel = label_cache[cache_key]
            else:
                rel = openai_label(client, query=query, jd_text=jd_text, model=args.model, sleep_s=args.sleep)
                label_cache[cache_key] = rel

            rows.append({"qid": qid, "posting_id": pid, "relevance": int(rel)})

            if args.save_every > 0 and len(rows) % args.save_every == 0:
                pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
                print(f"[SAVE] {len(rows)}/{total_pairs}")

    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved qrels: {args.out} (rows={len(rows)})")


if __name__ == "__main__":
    main()
