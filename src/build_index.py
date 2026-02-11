import numpy as np
import pandas as pd
from joblib import dump
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    CSV_PATH, ARTIFACTS_DIR,
    META_PATH, TEXT_COLS,
    VECTORIZER_VEC_PATH, VECTORS_VEC_PATH,
    VECTORIZER_KW_PATH,  VECTORS_KW_PATH,
)
from utils import safe_read_csv, build_full_text, kiwi_tokenize

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = safe_read_csv(CSV_PATH)
    df["full_text"] = build_full_text(df, TEXT_COLS).fillna("")

    texts = df["full_text"].values

    # (2차) 벡터 인덱스: char n-gram (기존 v0.1)
    vectorizer_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
    )
    X_vec = vectorizer_vec.fit_transform(texts)

    # (1차) 키워드 인덱스: word n-gram
    vectorizer_kw = TfidfVectorizer(
        tokenizer=kiwi_tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_kw = vectorizer_kw.fit_transform(texts)

    # 저장
    dump(vectorizer_vec, VECTORIZER_VEC_PATH)
    sparse.save_npz(VECTORS_VEC_PATH, X_vec)

    dump(vectorizer_kw, VECTORIZER_KW_PATH)
    sparse.save_npz(VECTORS_KW_PATH, X_kw)

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

    print("[OK] Hybrid index built.")
    print(f"- KW Vectorizer: {VECTORIZER_KW_PATH}")
    print(f"- KW Vectors:    {VECTORS_KW_PATH}  (shape={X_kw.shape})")
    print(f"- Vec Vectorizer:{VECTORIZER_VEC_PATH}")
    print(f"- Vec Vectors:   {VECTORS_VEC_PATH} (shape={X_vec.shape})")
    print(f"- Meta:          {META_PATH}  (rows={len(meta)})")

if __name__ == "__main__":
    main()
