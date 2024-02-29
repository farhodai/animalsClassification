import streamlit as st
from fastai.vision.all import *
import plotly.express as px

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Handle potential path discrepancies between operating systems
if str(pathlib.Path().root) == '\\':  # Assuming Windows
    pathlib.PosixPath = pathlib.WindowsPath

#title
st.title("Ҳайвонҳоро классификатсия мекардагии модел (танҳо сурати ҳайвонҳоро ҷойгир кунед.)")

#rasmni joylash
file=st.file_uploader('Ҷойгир кардани расм', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    try:
        st.image(file)
        #PIL convert
        img = PILImage.create(file)
        
        # model
        model = load_learner('animal_model.pkl')

        # prediction
        pred, pred_id, probs = model.predict(img)
        st.success(f'Пешгӯйӣ: {pred}')
        st.info(f'Эҳтимолият: {probs[pred_id]*100:.1f}%')

        #plotting
        fig = px.bar(x=probs*100, y=model.dls.vocab)
        st.plotly_chart(fig)

    except Exception as e:  # Catch generic exceptions
        st.error("Хато: Танҳо сурати ҳайвонҳо, хирс ва моҳиҳоро.")
        print(f"Error: {e}")  # Print detailed error message for debugging

# Informative message when no image is uploaded
if not file:
    st.info("Илтимос сурати ҳайвонҳо, хирс ва моҳиҳоро дохил кунед.")