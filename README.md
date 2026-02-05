# Job Search System

채용공고 CSV(`jobs_merged.csv`)를 기반으로, 사용자가 터미널에 검색어를 입력하면 
유사도 기반으로 Top-K 채용공고를 추천하는 검색 시스템

## Project Structure
```text
job-search/
  src/
    config.py
    utils.py
    build_index.py
    search.py
  data/          # (gitignore) 원본 CSV 위치
  artifacts/     # (gitignore) 인덱스 결과물 저장
  requirements.txt
  README.md
  VERSIONING.md
```

## Setup
```terminal
pip install -r requirements.txt
```

## Build Index (1회 / 데이터 업데이트 시)
`data/jobs_merged.csv`가 준비되어 있어야 함.

```terminal
python src/build_index.py
```

## Search (반복 실행)
```terminal
python src/search.py
```

`검색어>` 부분에 원하는 공고의 핵심 단어를 입력하면 추천공고가 나열됨.


## Current Status
- Version: v0.1 (TF-IDF Search System)
- Evaluation: 진행 예정
