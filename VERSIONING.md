# Version History

## v0.1 — TF-IDF Search System
### Features
- 채용공고 CSV 기반 검색 시스템 구축
- full_text 생성:
  - job_title
  - responsibilities
  - requirements
  - preferred_qualifications
  - tech_stack
- TF-IDF (char ngram) 기반 벡터 인덱싱
- cosine similarity Top-K 추천
- 터미널 검색 인터페이스 구현
- 자동 relevance labeling pipeline 구축
- Retrieval 평가 스크립트 구현

### Evaluation Setup
- Queries: 10 (multi-domain)
- Top20 retrieval per query
- Auto-labeled relevance via LLM
- Compute metrics.

### Metrics:
- P@10   : 0.7100
- MRR@10 : 0.7367
- nDCG@10: 0.6227

TF-IDF 기반 baseline으로서 상위 추천 정확도(P@10)와 첫 추천 품질(MRR)은 비교적 양호한 수준을 보였으며,
ranking quality(nDCG)는 중간 수준으로 나타남.

### Notes
Evaluation relevance는 LLM 자동 라벨링 기반으로 생성되었기 때문에 일부 노이즈가 포함될 수 있음.