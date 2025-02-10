# 🏠 부동산 허위 매물 분류 모델 (Real Estate Fake Listing Classification)

🚀 **부동산 매물 데이터를 분석하여 허위 매물을 탐지하는 머신러닝 모델을 개발하는 프로젝트입니다.**  
데이터 전처리부터 모델 학습, 최적화, 평가까지의 전 과정을 포함합니다.

---

## 📌 프로젝트 개요

### 🎯 목적
- 부동산 플랫폼에서 **허위 매물(Scam Listing)**을 탐지하는 AI 모델 개발
- 사용자가 신뢰할 수 있는 부동산 정보를 얻을 수 있도록 **데이터 정제 및 필터링**
- 부동산 시장의 투명성을 높이고 소비자 피해 방지

### 🎯 목표
1. **부동산 데이터 분석 및 전처리**  
   - 결측치 처리, 이상치 제거, 데이터 증강
   - 주요 피처(보증금, 월세, 중개사무소, 제공 플랫폼 등) 탐색 및 시각화  
   
2. **머신러닝 & 딥러닝 모델 개발**  
   - XGBoost, LightGBM, CatBoost, RandomForest 등 비교  
   - 최적의 하이퍼파라미터 탐색 (Optuna 활용)  

3. **모델 평가 및 최적화**  
   - F1-score, Precision-Recall Curve 분석  
   - 허위 매물 탐지 성능 극대화  

---

## 📂 프로젝트 구조

📦 real-estate-fake-listing-detection
 ┣ 📂 data  
 ┃ ┣ 📜 train.csv             
 ┃ ┣ 📜 test.csv  
 ┃ ┣ 📜 sample_submission.csv  
 ┣ 📂 notebooks  
 ┃ ┣ 📜 01_data_analysis.ipynb  
 ┃ ┣ 📜 02_feature_engineering.ipynb  
 ┃ ┣ 📜 03_model_training.ipynb  
 ┃ ┣ 📜 04_hyperparameter_tuning.ipynb  
 ┣ 📂 models  
 ┃ ┣ 📜 best_xgboost_model.pkl  
 ┃ ┣ 📜 best_lightgbm_model.pkl  
 ┣ 📂 src                 
 ┃ ┣ 📜 preprocess.py       
 ┃ ┣ 📜 train.py            
 ┃ ┣ 📜 inference.py       
 ┣ 📜 README.md             
 ┣ 📜 requirements.txt  

 ---

## 📊 데이터 설명

| 컬럼명 | 설명 |
|--------|------|
| `보증금` | 매물의 보증금 (단위: 만원) |
| `월세` | 매물의 월세 (단위: 만원) |
| `전용면적` | 매물의 실면적 (단위: m²) |
| `해당층` | 해당 매물의 층수 |
| `총층` | 건물의 전체 층수 |
| `방수` | 방 개수 |
| `욕실수` | 욕실 개수 |
| `주차가능여부` | 주차 가능 여부 (1: 가능, 0: 불가능) |
| `제공플랫폼` | 매물을 제공한 플랫폼 (A, B, C 등) |
| `중개사무소` | 중개사무소 식별값 |
| `허위매물여부` | **타겟 변수 (1: 허위 매물, 0: 실제 매물)** |

---

## ⚙️ 모델 개발 과정

### 1️⃣ 데이터 전처리
- 결측치 처리 (전용면적, 해당층 등)
- 이상치 탐지 및 제거
- 새로운 피처 생성 (보증금/월세 비율, 주차 가능 여부 등)
- 허위 매물 특징 분석 및 시각화

### 2️⃣ 모델 학습
- XGBoost, LightGBM, CatBoost, RandomForest 비교
- Optuna를 활용한 **하이퍼파라미터 최적화**
- K-fold Cross Validation 적용

### 3️⃣ 모델 평가
- **Macro F1-score** 최적화
- Precision-Recall Curve 분석
- Feature Importance 해석

---

## 🚀 사용 방법

### 1️⃣ 환경 설정
필요한 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
