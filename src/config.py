from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CSV_PATH = DATA_DIR / "jobs_merged.csv"

VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
VECTORS_PATH = ARTIFACTS_DIR / "job_vectors.npz"
META_PATH = ARTIFACTS_DIR / "row_meta.parquet"

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
