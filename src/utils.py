import re
import pandas as pd
from kiwipiepy import Kiwi
kiwi = Kiwi()

# csv 읽기
def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        raise e

# 텍스트 정규화
def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

# full_text 만들기
def build_full_text(df: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    # 없는 컬럼은 빈 값으로 만들어둠(실행 안정성)
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""

    temp = df[text_cols].fillna("")

    # 가중치 없음
    def row_to_text(row):
        parts = [normalize_text(row[c]) for c in text_cols]
        return " ".join([p for p in parts if p])

    return temp.apply(lambda r: row_to_text(r), axis=1)


STOP_POS = {"JKS","JKC","JKG","JKO","JKB","JKV","JKQ","JX","JC"}  # 조사 (약어 표기)
STOPWORDS = {"그리고","또한","하지만","매우","정말"}

def kiwi_tokenize(text):
    tokens = []
    for token, pos, _, _ in kiwi.analyze(text)[0][0]:
        if pos in STOP_POS:  # 조사 제거
            continue
        if token in STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        tokens.append(token.lower())
    return tokens
