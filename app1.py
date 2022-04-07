# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

pclass_d = {0:"First",1:"Second", 2:"Third"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
sex_d = {0:"Female", 1:"Male"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="Titanic")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://ocdn.eu/pulscms-transforms/1/g6gk9kpTURBXy8wOThjMDY4ZmU3NTY4MTdmZmRhZmI5ZGU0ODRjZmQwMC5qcGeTlQMHzLzNBUbNAviVAs0DBwDDw5MJpmQ4YTg0ZQaBoTAB/kadr-z-filmu-titanic.jpg")

	with overview:
		st.title("Would you survive on Titanic?")

	with left:
		sex_radio = st.radio( "Sex", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		embarked_radio = st.radio( "Port of embarkment", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )
		pclass_radio = st.radio("Travel class", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])

	with right:
		age_slider = st.slider("Age", value=18, min_value=1, max_value=80)
		sibsp_slider = st.slider("Number of siblings and/or spouse", min_value=0, max_value=8)
		parch_slider = st.slider("Number of parents and/or children", min_value=0, max_value=6)
		fare_slider = st.slider("Ticket price", min_value=0, max_value=512, step=1)

	data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Would such a person survive the catastrophe?")
		st.subheader(("Yes" if survival[0] == 1 else "No"))
		st.write("Prediction certainty {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
