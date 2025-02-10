import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
import optuna
import joblib

# 🚀 데이터 전처리 모듈 import
from preprocessor import main as DataPreprocessor

# 데이터 전처리 함수
def preprocess_data(data_path, target_column, features):
    """ 데이터 로드 후 전처리 수행 """
    raw_data = pd.read_csv(data_path)
    
    # 🛠 DataPreprocessor 적용
    train, _ = DataPreprocessor(raw_data, raw_data)  # train 데이터만 전처리

    # 특성과 타겟 분리
    X = train[features]
    y = train[target_column]

    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# Optuna를 사용한 XGBoost 하이퍼파라미터 튜닝
def optimize_xgb(X, y):
    def objective(trial):
        class_counts = np.bincount(y)
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_pos_weight
        }
        model = XGBClassifier(**params, eval_metric="logloss", enable_categorical=False, n_jobs=-1)

        cv_results = cross_validate(model, X, y, cv=5, scoring="f1_macro", n_jobs=-1, return_train_score=True)
        return np.mean(cv_results['test_score'])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params


# Optuna를 사용한 LightGBM 하이퍼파라미터 튜닝
def optimize_lgbm(X, y):
    def objective(trial):
        class_counts = np.bincount(y)
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "verbosity": -1
        }
        model = LGBMClassifier(**params)

        score = cross_val_score(model, X, y, cv=5, scoring="f1_macro", n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study.best_params


# 모델 정의 함수
def define_models(X, y):
    best_xgb_params = optimize_xgb(X, y)
    best_lgbm_params = optimize_lgbm(X, y)

    base_models = {
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=300, class_weight="balanced"),
        "ExtraTrees": ExtraTreesClassifier(random_state=42, n_estimators=300, class_weight="balanced"),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', **best_xgb_params),
        "LightGBM": LGBMClassifier(random_state=42, class_weight="balanced", verbosity=-1, **best_lgbm_params),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0, iterations=300, loss_function="Logloss")
    }
    return base_models


# VotingClassifier 생성 함수
def get_voting_classifier(models):
    return VotingClassifier(estimators=[(name, model) for name, model in models.items()], voting='soft')


# StackingClassifier 생성 함수
def get_stacking_classifier(models):
    return StackingClassifier(estimators=[(name, model) for name, model in models.items()], final_estimator=LogisticRegression())


# 모델 학습 및 평가
def train_and_evaluate_models(models, X, y, test_size=0.2):
    results = []
    best_model = None
    best_f1 = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        results.append({"Model": name, "Macro F1 Score": macro_f1})

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = model

    results_df = pd.DataFrame(results)
    return results_df, best_model


# 모델 학습 및 저장
def train_and_save_model(data_path, target_column, features, model_save_path, scaler_save_path):
    X, y, scaler = preprocess_data(data_path, target_column, features)

    # 모델 정의
    models = define_models(X, y)

    # Voting & Stacking 모델 추가
    models["VotingClassifier"] = get_voting_classifier(models)
    models["StackingClassifier"] = get_stacking_classifier(models)

    # Stratified K-Fold 평가
    results_df, best_model = train_and_evaluate_models(models, X, y)

    # 최적 모델 저장
    best_model.fit(X, y)
    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    print(f"Best model: {best_model.__class__.__name__}")
    print(f"Best model saved to {model_save_path}")

    return results_df, best_model.__class__.__name__


# 모델 성능 시각화
def visualize_model_performance(results_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="Macro F1 Score", y="Model", palette="Blues_r")
    plt.xlabel("Macro F1 Score")
    plt.ylabel("Model")
    plt.title("Model Performance Comparison")
    plt.xlim(0, 1)
    plt.show()


# 실행
if __name__ == "__main__":
    data_path = "data/train.csv"
    target_column = "허위매물여부"
    model_save_path = "model_list/ML/model/model_t_20.joblib"
    scaler_save_path = "model_list/ML/scaler/scaler_t_20.joblib"

    features = ['보증금', '월세', '총층', '주차가능여부', '관리비', '제공플랫폼', 
                '전용면적_결측치_여부', '해당층_결측치_여부', 'G52중개사무소여부', 
                '게재연', '게재월', '게재분기']

    results, best_model_name = train_and_save_model(data_path, target_column, features, model_save_path, scaler_save_path)
    visualize_model_performance(results)