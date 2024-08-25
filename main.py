import scipy.io as sio
import numpy as np
import pandas as pd
import glob

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def load_mat_file(filepath):
    mat_data = sio.loadmat(filepath)
    # 'DE_time'이라는 키를 사용해 데이터를 추출합니다.
    for key in mat_data.keys():
        if 'DE_time' in key:
            return mat_data[key].flatten()

# 신호 데이터를 일정 시간 구간으로 나누고, 각 구간에 대해 통계적 변수를 추출하는 함수
def extract_segment_features(data, segment_length):
    num_segments = len(data) // segment_length
    features_list = []

    for i in range(num_segments):
        segment = data[i * segment_length:(i + 1) * segment_length]
        features = {
            'mean': np.mean(segment),
            'std': np.std(segment),
            'max': np.max(segment),
            'min': np.min(segment),
            'skewness': pd.Series(segment).skew(),
            'kurtosis': pd.Series(segment).kurt()
        }
        features_list.append(features)

    return pd.DataFrame(features_list)

# 샘플링 주파수 설정
sampling_frequency = 12000  # 12000 sps
segment_time = 1  # 분석할 시간 구간 (단위: 초)
segment_length = sampling_frequency * segment_time  # 구간 당 샘플 수

if __name__ == "__main__":
    normal_files = glob.glob('./Data/Normal/*.mat')
    normal_features_list = []

    for file in normal_files:
        data = load_mat_file(file)
        segment_features = extract_segment_features(data, segment_length=12000)  # 1초 구간
        segment_features['label'] = 0  # 정상 데이터의 레이블은 0
        normal_features_list.append(segment_features)

    # 모든 정상 데이터 통합
    normal_features_df = pd.concat(normal_features_list, ignore_index=True)
    normal_features_df['label'] = 0 

    faulty_files = glob.glob('./Data/12k_DE/*.mat')[:14]
    faulty_features_list = []

    for file in faulty_files:
        data = load_mat_file(file)
        segment_features = extract_segment_features(data, segment_length=12000)  # 1초 구간
        segment_features['label'] = 1  # 불량 데이터의 레이블은 1
        faulty_features_list.append(segment_features)

    # 모든 불량 데이터 통합
    faulty_features_df = pd.concat(faulty_features_list, ignore_index=True)

    combined_df = pd.concat([normal_features_df, faulty_features_df], ignore_index=True)

    # 데이터셋을 특성과 레이블로 분리
    X = combined_df.drop('label', axis=1)
    y = combined_df['label']


    # 학습 데이터와 테스트 데이터로 분리 (80% 학습, 20% 테스트)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVM 모델 생성 및 학습
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측
    y_pred = model.predict(X_test)

    # 모델 평가
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))