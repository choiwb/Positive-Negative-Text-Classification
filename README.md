Penta Systems Technology Inc. Big Data Team member

# Positive-Negative-Text-ClassificationMinin
- in 2018/10/24 ~ 2018/11/30 project
- 사용 데이터: 미국 시카고 소재 20개 호텔의 리뷰(댓글) 총 1600개 (긍정: 800개, 부정: 800개)
# Text Mining 과정
- 데이터 전처리: 불필일요한 기호, 공백 제거, 알파벳 소문자화, 불용어 제거
- EDA:Top Down 방식의 용어 빈도 도출 및 Word Cloud 시각화
- TF-IDF 방식의 DTM 생성
# Modeling
- LSTM 및 CNN+LSTM 2가지 모델 성능 비교 -> CNN+LSTM이 성능은 상대적으로 떨어지지만 과적합 가능성이 낮고, 속도가 월등히 빠르므로 더 우수
- 모델 성능지표로, 정확도, 정밀도, 특이도, 민감도의 Confusion Matrix 도출 및, ROC 커브 시각화를 통한 AUC 값 도출
- 추가적으로, 모델 성능 향상을 위해 추가적인 하이퍼 파라미터 적용 과정 필요.
# Advanced Process
- Image Caption 진행 예정

