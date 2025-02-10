import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans

def drop_unnecessary_columns(df):
    """ 불필요한 열을 제거하는 함수 """
    drop_cols = ['매물확인방식', '방향', '방수', '욕실수', '총주차대수']
    df = df.drop(columns=drop_cols, errors='ignore')
    return df

def handle_missing_values(df):
    """ 결측값 처리 함수 """
    # 전용면적 결측치 여부 컬럼 생성
    df['전용면적_결측치_여부'] = df['전용면적'].isnull().astype(int)
    df = df.drop(columns=['전용면적'], errors='ignore')

    df['해당층_결측치_여부'] = df['해당층'].isnull().astype(int)
    df = df.drop(columns=['해당층'], errors='ignore')
    
    return df

def convert_currency_units(df):
    """ 보증금, 월세, 관리비 단위 변환 """
    df['보증금'] = df['보증금'] / 10000  # 만원 단위 변환
    df['월세'] = df['월세'] / 10000
    df['관리비'] = df['관리비'] / 10
    return df

def process_real_estate_office(df):
    """ 중개사무소 정보를 'G52중개사무소여부' 컬럼으로 변환 """
    df['G52중개사무소여부'] = df['중개사무소'].astype(str).str.startswith('G52').astype(int)
    df = df.drop(columns=['중개사무소'], errors='ignore')
    return df

def process_dates(df):
    """ '게재일' 컬럼을 '연-월' 형태로 변환하고, 연/월 분리 및 분기 컬럼 추가 """
    df['게재일'] = pd.to_datetime(df['게재일'], format='%Y-%m-%d', errors='coerce')
    df['게재연'] = df['게재일'].dt.year  # 연도 분리
    df['게재월'] = df['게재일'].dt.month  # 월 분리

    # 분기 컬럼 추가 (1~3월: 1분기, 4~6월: 2분기, 7~9월: 3분기, 10~12월: 4분기)
    df['게재분기'] = ((df['게재월'] - 1) // 3 + 1).astype(int)

    df = df.drop(columns=['게재일'], errors='ignore')
    return df

def encode_platform(df):
    """ 제공 플랫폼을 Label Encoding """
    if '제공플랫폼' in df.columns:
        le = LabelEncoder()
        df['제공플랫폼'] = le.fit_transform(df['제공플랫폼'].astype(str))
    return df

def train_kmeans_and_impute(train):
    """ Train 데이터에서만 KMeans 클러스터링을 수행하고 결측값을 채움 """
    
    features = ['보증금', '관리비', '월세', '주차가능여부']
    
    train['주차가능여부'] = train['주차가능여부'].map({'가능': 1, '불가능': 0})

    # 스케일링
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(train[features])

    # KMeans 클러스터링 수행 (train 데이터에서만!)
    kmeans = KMeans(n_clusters=5, random_state=7)
    train['cluster'] = kmeans.fit_predict(scaled_features)

    # 클러스터별 '총층' 중앙값 계산
    cluster_medians = train.groupby('cluster')['총층'].median()

    # 결측값을 클러스터 중앙값으로 대체하는 함수
    def impute_missing_values(df, feature, cluster_medians, kmeans, scaler):
        missing_data = df[df[feature].isnull()].copy()
        if missing_data.empty:
            return df
        
        scaled_missing_features = scaler.transform(missing_data[features])
        missing_data['cluster'] = kmeans.predict(scaled_missing_features)

        for idx, row in missing_data.iterrows():
            cluster = row['cluster']
            df.loc[idx, feature] = cluster_medians.get(cluster, np.nan)
        
        return df

    # Train 데이터의 결측값 채우기
    train = impute_missing_values(train, '총층', cluster_medians, kmeans, scaler)

    return train, kmeans, scaler, cluster_medians

def apply_kmeans_to_test(test, kmeans, scaler, cluster_medians):
    """ Test 데이터에 Train의 KMeans 결과를 적용하여 결측값을 채움 """
    
    features = ['보증금', '관리비', '월세', '주차가능여부']
    test['주차가능여부'] = test['주차가능여부'].map({'가능': 1, '불가능': 0})

    def impute_missing_values(df, feature, cluster_medians):
        missing_data = df[df[feature].isnull()].copy()
        if missing_data.empty:
            return df

        scaled_missing_features = scaler.transform(missing_data[features])
        missing_data['cluster'] = kmeans.predict(scaled_missing_features)

        for idx, row in missing_data.iterrows():
            cluster = row['cluster']
            df.loc[idx, feature] = cluster_medians.get(cluster, np.nan)
        
        return df

    # Test 데이터의 결측값 채우기
    test = impute_missing_values(test, '총층', cluster_medians)

    return test

def main(train, test):
    """ 전체 전처리 과정 실행 """
    
    train = drop_unnecessary_columns(train)
    test = drop_unnecessary_columns(test)

    train = handle_missing_values(train)
    test = handle_missing_values(test)

    train = convert_currency_units(train)
    test = convert_currency_units(test)

    train = process_real_estate_office(train)
    test = process_real_estate_office(test)

    train = process_dates(train)
    test = process_dates(test)

    train = encode_platform(train)
    test = encode_platform(test)

    # Train 데이터에서만 클러스터링 학습
    train, kmeans, scaler, cluster_medians = train_kmeans_and_impute(train)

    # Test 데이터에 학습된 KMeans 적용
    test = apply_kmeans_to_test(test, kmeans, scaler, cluster_medians)

    return train, test

# 데이터 로드
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 전처리 실행
train, test = main(train, test)
print("train 데이터 셋: \n ",train.head(2))
print("\n test 데이터 셋: \n ",test.head(2))
print("전처리 완료!")
