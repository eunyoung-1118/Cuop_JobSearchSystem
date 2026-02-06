# Version History

## v0.2 — Hybrid TF-IDF Search System

### Features
- Hybrid 검색 구조 도입:
  - 1차 후보 축소: word TF-IDF similarity
  - 2차 정밀 랭킹: char TF-IDF similarity
- dual vector index 구조:
  - word ngram index
  - char ngram index
- 검색 속도 개선 (전체 similarity → 후보 subset similarity)
- 기존 terminal search 인터페이스 유지

### Metrics:
- P@10   : 0.6900
- MRR@10 : 0.7367
- nDCG@10: 0.6112

**P@10 (0.71 → 0.69)**: Top-10 내 관련 공고 비율이 약간 감소하였다.  
  1차 word TF-IDF 후보 축소 단계에서 일부 relevant 공고가 후보군에서 제외되었을 가능성이 있다.

**MRR@10 (0.7367 → 0.7367)**: 첫 번째 relevant 결과 위치는 거의 동일하게 유지되었다.  
  즉, 상위 1순위 추천 품질에는 큰 변화가 없음을 의미한다.

**nDCG@10 (0.6227 → 0.6112)**: ranking quality가 소폭 감소하였다.  
  이는 reranking 과정에서 relevance가 높은 문서의 상대적 순서가 일부 변경된 영향으로 해석된다.

Hybrid 구조는 검색 속도 개선 및 후보 축소 효율 측면에서는 장점이 있으나,  
현재 설정에서는 ranking precision 측면에서 일부 trade-off가 발생한 것으로 보인다.

향후 개선 방향:
- candidate_k 증가를 통한 recall 보완
- word/char score fusion 적용
- word vectorizer 튜닝
---

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
