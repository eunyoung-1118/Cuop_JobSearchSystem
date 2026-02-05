import numpy as np
import pandas as pd
from joblib import dump
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from config import CSV_PATH, ARTIFACTS_DIR, VECTORIZER_PATH, VECTORS_PATH, META_PATH, TEXT_COLS
from utils import safe_read_csv, build_full_text

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = safe_read_csv(CSV_PATH)
    df["full_text"] = build_full_text(df, TEXT_COLS)

    # TF-IDF: 한국어 형태소 없이도 시작하기 좋은 char n-gram
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
    )

    X = vectorizer.fit_transform(df["full_text"].values)

    # 저장(인덱스 결과물)
    dump(vectorizer, VECTORIZER_PATH)
    sparse.save_npz(VECTORS_PATH, X)

    # 검색 결과 출력용 메타만 따로 저장 (없으면 빈 값 처리)
    meta_cols = []
    for c in ["posting_id", "job_title", "experience_Level", "location", "posting_period", "url"]:
        if c in df.columns:
            meta_cols.append(c)

    if not meta_cols:
        meta = pd.DataFrame({"row_index": np.arange(len(df))})
    else:
        meta = df[meta_cols].copy()
        meta.insert(0, "row_index", np.arange(len(df)))

    meta.to_parquet(META_PATH, index=False)

    print("[OK] Index built.")
    print(f"- Vectorizer: {VECTORIZER_PATH}")
    print(f"- Vectors:    {VECTORS_PATH}  (shape={X.shape})")
    print(f"- Meta:       {META_PATH}  (rows={len(meta)})")

if __name__ == "__main__":
    main()
