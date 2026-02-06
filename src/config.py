from pathlib import Path

# Paths
ARTIFACTS_DIR = Path("artifacts")
CSV_PATH = Path("data/jobs_merged.csv")

VECTORIZER_VEC_PATH = ARTIFACTS_DIR / "vectorizer_vec.joblib"
VECTORS_VEC_PATH     = ARTIFACTS_DIR / "vectors_vec.npz"

VECTORIZER_KW_PATH = ARTIFACTS_DIR / "vectorizer_kw.joblib"
VECTORS_KW_PATH     = ARTIFACTS_DIR / "vectors_kw.npz"

META_PATH = ARTIFACTS_DIR / "meta.parquet"

# Index design
# full_text를 만들 때 사용할 컬럼 목록
TEXT_COLS = [
    "job_title",
    "responsibilities",
    "requirements",
    "preferred_qualifications",
    "tech_stack",
]

# Search params
TOP_K = 10
CANDIDATE_K = 2000
