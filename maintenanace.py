import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="기계 장비 예지보전 시스템", layout="centered")
st.title("🛠️ 기계 장비 고장 위험 실시간 진단")
st.write("주요 센서 데이터를 조절하여 장비의 실시간 고장 확률을 확인하세요.")

# 2. 저장해둔 랜덤 포레스트 모델 불러오기 (실제 파일명으로 수정 완료!)
@st.cache_resource
def load_model():
    # 왼쪽 탐색기에 있는 maintenance.pkl 파일을 바로 로드합니다.
    return joblib.load("maintenance.pkl")

try:
    rf_model_eng = load_model()
except Exception as e:
    st.error("⚠️ 모델 파일(maintenance.pkl)을 찾을 수 없습니다. 파일이 동일한 폴더에 있는지 확인해 주세요.")
    st.stop()

st.divider()

# 3. 실시간 센서 데이터 입력 컴포넌트 배치
st.subheader("📊 실시간 센서 데이터 입력")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input("현재 대기 온도 (K)", min_value=200.0, max_value=400.0, value=298.5, step=0.1)
    proc_temp = st.number_input("현재 공정 온도 (K)", min_value=200.0, max_value=400.0, value=309.1, step=0.1)
    speed = st.number_input("현재 모터 회전속도 (rpm)", min_value=0.0, max_value=5000.0, value=1500.0, step=10.0)

with col2:
    torque = st.number_input("현재 가해지는 토크 (Nm)", min_value=0.0, max_value=100.0, value=42.5, step=0.5)
    wear = st.slider("현재 누적 공구마모도 (min)", min_value=0, max_value=300, value=60)

st.divider()

# 4. 데이터프레임 변환 및 파생 변수 계산
input_data = pd.DataFrame(
    [[air_temp, proc_temp, speed, torque, wear]],
    columns=['대기온도', '공정온도', '회전속도', '토크', '공구마모도']
)

input_data['온도차'] = input_data['공정온도'] - input_data['대기온도']
input_data['동력'] = input_data['토크'] * (input_data['회전속도'] * 2 * np.pi / 60)
input_data['과열위험'] = (input_data['온도차'] >= 10).astype(int)
input_data['마모위험'] = (input_data['공구마모도'] >= 200).astype(int)


# 5. 진단 및 예측 실행 버튼
if st.button("🚀 실시간 상태 진단 실행", use_container_width=True):
    # 스케일링이 빠진 모델이므로 input_data 원본을 그대로 예측합니다.
    predicted = rf_model_eng.predict(input_data)
    prob = rf_model_eng.predict_proba(input_data)
    
    failure_probability = prob[0][1] * 100

    # 6. 진단 결과 시각화 출력
    st.subheader("🔍 진단 결과 보고서")
    
    if predicted[0] == 1:
        st.error(f"🚨 **고장 위험 상태 (주의 요망)**")
    else:
        st.success(f"✅ **정상 가동 중 (안정)**")
        
    st.write(f"**장비 고장 및 오작동 발생 확률:** {failure_probability:.2f}%")
    st.progress(int(failure_probability))
    
    st.info(f"💡 현재 내부 발열 온도차는 **{input_data['온도차'][0]:.1f} K** 이며, 계산된 실시간 동력은 **{input_data['동력'][0]:.2f} W** 입니다.")
