import math
import pandas as pd

EVAL_DATASET_PATH = "eval/eval_dataset.csv"
K = 10


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
    df = pd.read_csv(EVAL_DATASET_PATH)

    required = {"qid", "rank", "relevance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["relevance"].isna().any():
        raise ValueError("relevance에 NaN이 있습니다. auto_label이 정상 완료됐는지 확인하세요.")

    df["relevance"] = df["relevance"].astype(int)
    df = df.sort_values(["qid", "rank"])

    metrics = []
    for qid, g in df.groupby("qid"):
        rels = g["relevance"].tolist()
        metrics.append({
            "qid": qid,
            "P@10": precision_at_k(rels, K),
            "MRR@10": mrr_at_k(rels, K),
            "nDCG@10": ndcg_at_k(rels, K),
        })

    mdf = pd.DataFrame(metrics)

    print("\n=== Per-query metrics ===")
    print(mdf.to_string(index=False))

    print("\n=== Overall (mean) ===")
    print(f"P@10   : {mdf['P@10'].mean():.4f}")
    print(f"MRR@10 : {mdf['MRR@10'].mean():.4f}")
    print(f"nDCG@10: {mdf['nDCG@10'].mean():.4f}")


if __name__ == "__main__":
    main()


