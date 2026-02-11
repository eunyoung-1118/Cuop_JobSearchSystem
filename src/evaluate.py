import argparse
import math
import pandas as pd
from config import RUN_PATH, QRELS_PATH

def dcg(rels):
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(rels, k):
    rels_k = rels[:k]
    ideal = sorted(rels, reverse=True)[:k]
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(rels_k) / denom


def precision_at_k(rels, k):
    rels_k = rels[:k]
    return sum(1 for r in rels_k if r > 0) / k


def mrr_at_k(rels, k):
    for i, r in enumerate(rels[:k], start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN_PATH),
                    help="eval/run.csv (default from config)")
    ap.add_argument("--qrels", default=str(QRELS_PATH),
                    help="eval/qrels.csv (default from config)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ignore_unjudged", action="store_true",
                    help="qrels에 없는 문서는 평가에서 제외 (기본은 0으로 처리)")
    args = ap.parse_args()

    run = pd.read_csv(args.run)
    qrels = pd.read_csv(args.qrels)

    # 필수 컬럼 체크
    required_run = {"qid", "posting_id", "rank"}
    required_qrels = {"qid", "posting_id", "relevance"}

    missing_run = required_run - set(run.columns)
    missing_qrels = required_qrels - set(qrels.columns)

    if missing_run:
        raise ValueError(f"run missing columns: {sorted(missing_run)}")
    if missing_qrels:
        raise ValueError(f"qrels missing columns: {sorted(missing_qrels)}")

    # 타입 정리
    run["qid"] = run["qid"].astype(str)
    run["posting_id"] = run["posting_id"].astype(str)
    run["rank"] = run["rank"].astype(int)

    qrels["qid"] = qrels["qid"].astype(str)
    qrels["posting_id"] = qrels["posting_id"].astype(str)
    qrels["relevance"] = qrels["relevance"].astype(int)

    # join
    merged = run.merge(qrels, on=["qid", "posting_id"], how="left")

    if args.ignore_unjudged:
        # qrels 없는 문서는 제외
        before = len(merged)
        merged = merged.dropna(subset=["relevance"]).copy()
        after = len(merged)
        print(f"[INFO] ignore_unjudged enabled: dropped {before - after} rows")
    else:
        # qrels 없는 문서는 0점 처리
        merged["relevance"] = merged["relevance"].fillna(0).astype(int)

    # rank 순 정렬
    merged = merged.sort_values(["qid", "rank"])

    # metrics
    metrics = []
    for qid, g in merged.groupby("qid"):
        rels = g["relevance"].tolist()
        metrics.append({
            "qid": qid,
            "P@10": precision_at_k(rels, args.k),
            "MRR@10": mrr_at_k(rels, args.k),
            "nDCG@10": ndcg_at_k(rels, args.k),
        })

    mdf = pd.DataFrame(metrics)

    print("\n=== Per-query metrics ===")
    print(mdf.to_string(index=False))

    print("\n=== Overall (mean) ===")
    print(f"P@10   : {mdf['P@10'].mean():.4f}")
    print(f"MRR@10 : {mdf['MRR@10'].mean():.4f}")
    print(f"nDCG@10: {mdf['nDCG@10'].mean():.4f}")

    # coverage 정보(참고)
    judged = run.merge(qrels, on=["qid", "posting_id"], how="left")["relevance"].notna().mean()
    print(f"\n[INFO] judged coverage in run: {judged*100:.2f}%")
    print(f"[INFO] run rows: {len(run)}, qrels rows: {len(qrels)}")


if __name__ == "__main__":
    main()
