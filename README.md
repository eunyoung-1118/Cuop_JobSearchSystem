# Job Search System

채용공고 CSV(`jobs_merged.csv`)를 기반으로, 사용자가 터미널에 검색어를 입력하면 
유사도 기반으로 Top-K 채용공고를 추천하는 검색 시스템

## Project Structure
```text
job-search/
  src/
    config.py
    utils.py
    build_index.py  # 검색 인덱스 생성
    search.py       # 터미널 검색 인터페이스
    auto_label.py   # 자동 relevance labeling + eval dataset 생성
    evaluate.py     # retrieval 성능 평가
  data/             # (gitignore) 원본 CSV 위치
  artifacts/        # (gitignore) 인덱스 결과물 저장
  eval/             # 평가 관련 파일 (dataset 일부 ignore)
  requirements.txt
  README.md
  VERSIONING.md
```

## Setup
```terminal
pip install -r requirements.txt
```

`.env` 파일 생성:
```
OPENAI_API_KEY=your_api_key_here
```
자동 라벨링 평가에 사용됨.


## Build Index (1회 / 데이터 업데이트 시)
`data/jobs_merged.csv`가 준비되어 있어야 함.

```terminal
python src/build_index.py
```

## Search (반복 실행)
```terminal
python src/search.py
```

`검색어>` 부분에 원하는 공고의 핵심 키워드를 입력하면 추천공고 Top-K가 출력됨.

## Evaluation

### 1. 자동 relevance labeling dataset 생성 (1회 / 쿼리 변경 시, 데이터 업데이트 시)
```
python src/auto_label.py
```
- OpenAI API를 이용해 relevance(0/1/2) 자동 라벨링
- eval/qrels.csv 생성

### 2. 검색 결과 생성
```
python src/build_run.py
```
- 현재 검색 모델로 Top-K 추출
- eval/run.csv 생성

### 3. 검색 성능 평가
```
python src/evaluate.py
```
다음 지표가 계산됨:
- Precision@10
- MRR@10
- nDCG@10

## Current Status
- Version: v0.3 — morph-tokenizer
- Evaluation: 진행 예정
