import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Função para carregar modelos
def load_model(model_name):
    if model_name == 'Modelo S - Scratch sem Data Augmentation':
        model = tf.keras.models.load_model('C:/Users/Fernando Correia/Desktop/projetoDL/modelos/model_s_without_data_augmentation.h5')
    elif model_name == 'Modelo S - Scratch com Data Augmentation':
        model = tf.keras.models.load_model('C:/Users/Fernando Correia/Desktop/projetoDL/modelos/model_s_with_data_augmentation.h5')
    elif model_name == 'Modelo T - Feature Extraction sem Data Augmentation':
        model = tf.keras.models.load_model('C:/Users/Fernando Correia/Desktop/projetoDL/modelos/model_t_feature_extraction_without_data_augmentation.h5')
    elif model_name == 'Modelo T - Feature Extraction com Data Augmentation':
        model = tf.keras.models.load_model('C:/Users/Fernando Correia/Desktop/projetoDL/modelos/model_t_feature_extraction_with_data_augmentation.h5')
    elif model_name == 'Modelo T - Fine Tuning com Data Augmentation':
        model = tf.keras.models.load_model('C:/Users/Fernando Correia/Desktop/projetoDL/modelos/model_t_fine_tuning_with_data_augmentation.h5')

    return model

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def carregar_e_preprocessar_imagem(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.set_page_config(page_title="Classificação de Vegetais", page_icon="🥦")

st.title('Classificação de Vegetais')

model_option = st.selectbox(
    'Escolha o modelo que deseja usar:',
    ('Modelo S - Scratch sem Data Augmentation', 'Modelo S - Scratch com Data Augmentation', 
     'Modelo T - Feature Extraction sem Data Augmentation', 'Modelo T - Feature Extraction com Data Augmentation', 'Modelo T - Fine Tuning com Data Augmentation')
)

uploaded_files = st.file_uploader("Escolha imagens...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Lista de categorias
categories = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 
              'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

if uploaded_files:
    images = [Image.open(file) for file in uploaded_files]

    if st.button('Submeter'):
        model = load_model(model_option)
        with st.spinner('Classificando as imagens...'):
            dados_de_entrada = np.vstack([carregar_e_preprocessar_imagem(image) for image in images])

            previsoes = model.predict(dados_de_entrada)

            classes_preditas_indices = np.argmax(previsoes, axis=1)
            classes_preditas_nomes = [categories[i] for i in classes_preditas_indices]

            for image, classe in zip(images, classes_preditas_nomes):
                st.image(image, caption=f'A imagem parece ser um {classe}', width=150)
else:
    st.button('Submeter', disabled=True)
    st.write("Por favor, carregue uma ou mais imagens para clicar no botão Submeter.")