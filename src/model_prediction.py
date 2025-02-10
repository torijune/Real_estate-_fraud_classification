import torch
import pandas as pd
import joblib
from preprocessor import main as DataPreprocessor  # ✅ 새로운 전처리 코드 import

# PyTorch 딥러닝 모델 클래스 (저장된 모델 로드를 위해 동일한 클래스 정의 필요)
class MLPModel(torch.nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)  # 이진 분류를 위해 출력 노드는 1개
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def load_and_predict(model_path, scaler_path, test_data_path, train_data_path, model_type, target_column="허위매물여부", device="cpu"):
    # 스케일러 로드
    scaler = joblib.load(scaler_path)

    # 데이터 로드
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    # ✅ 새로운 전처리 코드 적용
    train_data, test_data = DataPreprocessor(train_data, test_data)

    # Train 때 사용한 변수들만 활용
    features = ['보증금', '월세', '총층', '주차가능여부', '관리비', '제공플랫폼', 
                '전용면적_결측치_여부', '해당층_결측치_여부', 'G52중개사무소여부', 
                '게재연', '게재월', '게재분기']

    test_features = test_data[features]

    # 스케일링 적용
    test_features_scaled = scaler.transform(test_features)

    # 모델 로드 및 예측
    if model_type == "ML":
        model = joblib.load(model_path)
        predictions = model.predict(test_features_scaled)

    elif model_type == "DL":
        input_size = len(features)
        model = MLPModel(input_size).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_tensor = torch.tensor(test_features_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = model(test_tensor).cpu().numpy()
            predictions = (predictions > 0.5).astype(int).flatten()
    else:
        raise ValueError("Invalid model type. Choose 'ML' or 'DL'.")

    # 결과 추가
    test_data["Predicted"] = predictions
    return test_data

def get_model_path(model, version):
    if model == 'ML':
        return f'model_list/{model}/model/model_{version}.joblib', f'model_list/{model}/scaler/scaler_{version}.joblib'
    elif model == 'DL':
        return f'model_list/{model}/model/model_{version}.pth', f'model_list/{model}/scaler/scaler_{version}.joblib'
    else:
        raise ValueError(f"Invalid model type: {model}. Please choose 'ML' or 'DL'.")

if __name__ == "__main__":
    # MPS 또는 CPU 장치 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = input("사용할 모델 종류를 입력하세요 (ML / DL):").strip()
    version = input("사용할 모델 버전을 입력하세요 (예: t_1): ").strip()
    test_data_path = 'data/test.csv'
    train_data_path = 'data/train.csv'
    submission_template_path = 'data/sample_submission.csv'

    try:
        model_path, scaler_path = get_model_path(model, version)
        print(f"Model path: {model_path}")
    except ValueError as e:
        print(e)
        exit()

    output_path = f'prediction_list/final_predictions_{model}_{version}.csv'

    # 예측 수행
    predictions = load_and_predict(model_path, scaler_path, test_data_path, train_data_path, model_type=model, device=device)
    submission = pd.read_csv(submission_template_path)
    submission["허위매물여부"] = predictions["Predicted"].values[:len(submission)]

    # 결과 저장
    submission.to_csv(output_path, index=False)
    print(f"Predictions for version {version} saved to {output_path}")
