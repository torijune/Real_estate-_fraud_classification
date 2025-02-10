import os
import torch
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from preprocessor import train_main as preprocessor


# 글로벌 스케일러
scaler = StandardScaler()

# 데이터 로드 및 전처리
def load_data(train_path, features, device):
    df = pd.read_csv(train_path)

    preprocessed_df = preprocessor(df)

    X = preprocessed_df.drop(columns=['허위매물여부'])
    X = X[features]
    y = preprocessed_df['허위매물여부']

    # 데이터 정규화
    global scaler
    X_scaled = scaler.fit_transform(X)

    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Tensor로 변환 및 MPS로 전송
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    return X_train, X_test, y_train, y_test

# 모델 정의
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
        x = self.sigmoid(x)  # Sigmoid 활성화 함수
        return x

# 학습 함수
def training(X_train, y_train, input_size, device, batch_size=16, epochs=100, lr=0.001):
    model = MLPModel(input_size).to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 데이터셋을 미니 배치로 나누기
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 학습
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # 순전파
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 에포크 손실 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    return model

# 평가 함수 (Macro F1 Score 계산)
def evaluation(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_classes = (y_pred > 0.5).float()

        # Tensor를 NumPy로 변환
        y_test_np = y_test.cpu().numpy()
        y_pred_classes_np = y_pred_classes.cpu().numpy()

        # Macro F1 Score 계산
        macro_f1 = f1_score(y_test_np, y_pred_classes_np, average='macro')
        print(f"Macro F1 Score: {macro_f1:.4f}")

# 실행 코드
if __name__ == "__main__":
    # MPS 장치 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 경로
    train_path = "data/train.csv"

    # 데이터 로드
    # ['보증금', '월비합계', '전용면적', '중개사무소', '제공플랫폼', '주차가능여부_가능', '주차가능여부_불가능', '매물확인방식', '게재연', '게재월', '결측치_유무']
    features = ['보증금', '월비합계', '전용면적', '중개사무소', '제공플랫폼', '게재연', '게재월', '결측치_유무']
    X_train, X_test, y_train, y_test = load_data(train_path, features, device)

    # 모델 학습
    input_size = X_train.shape[1]
    model = training(X_train, y_train, input_size, device)

    # 모델 평가
    evaluation(model, X_test, y_test)

    # 모델 및 스케일러 저장
    scaler_save_dir = "model_list/DL/scaler"
    model_save_dir = "model_list/DL/model"

    model_path = os.path.join(model_save_dir, "model_t_3.pth")
    scaler_path = os.path.join(scaler_save_dir, "scaler_t_3.joblib")

    # 모델 저장
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 스케일러 저장
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")