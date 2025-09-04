import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
  page_title = "klasifikasi tomat",
  page_icon = "🍅"     
)

model = joblib.load("model_klasifikasi_tomat.joblib")
scaler = joblib.load("scaler_klasifikasi_tomat.joblib")

st.title("Klasifikasi Tomat")
st.markdown("Aplikasi machine learning untuk klasifikasi tomat termasuk kategori *Ekspor, Lokal Premium, atau Industri*")

berat = st.slider("Berat Tomat", 50, 200, 80)
kekenyalan = st.slider("Tingkat Kekenyalan",2.0, 10.0, 4.2)
kadar_gula = st.slider("Kadar Gula", 1.0, 10.0, 4.2)
tebal_kulit = st.slider("Tebal Kulit", 0.0, 1.0, 0.8)

if st.button("Prediksi"):
	data_baru = pd.DataFrame([[berat,kekenyalan,kadar_gula,tebal_kulit]],
		columns=["berat","kekenyalan","kadar_gula","tebal_kulit"])

	data_baru_scaled = scaler.transform(data_baru)
	prediksi = model.predict(data_baru_scaled)[0]
	presentase = max(model.predict_proba(data_baru_scaled)[0])
	st.success(f"Prediksi {prediksi} keyakinan {presentase*100:.2f}%")
	st.balloons()
	st.caption("Dibuat dengan :heart: oleh Fikri Val Nabil Sugiarto")

