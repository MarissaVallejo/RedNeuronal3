import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from model import create_model, train_model
from preprocessing import preprocess_image, load_images_from_folder
from utils import plot_training_history, plot_confusion_matrix

st.set_page_config(page_title="Clasificador de Imágenes CNN", layout="wide")

st.markdown('''
<style>
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
''', unsafe_allow_html=True)

def main():
    st.title("Clasificador de Imágenes CNN")
    
    menu = ["Entrenamiento", "Predicción", "Predicción por Video"]
    choice = st.sidebar.selectbox("Seleccione Modo", menu)
    
    if choice == "Predicción por Video":
        st.header("Predicción por Video")
        
        if not os.path.exists('modelo_cnn.pth'):
            st.error("No se encontró un modelo entrenado. Por favor, entrene el modelo primero.")
            return
            
        uploaded_file = st.file_uploader("Cargar video para predicción", type=['mp4', 'avi'])
        
        if uploaded_file:
            # Guardar el video temporalmente
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Abrir el video con OpenCV
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Slider para navegar por los frames
            frame_idx = st.slider("Seleccionar frame", 0, total_frames-1, 0)
            
            # Cargar modelo y nombres de clases
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load('modelo_cnn.pth')
            class_names = checkpoint['class_names']
            
            model = create_model(len(class_names))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Ir al frame seleccionado
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convertir BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Crear una imagen PIL
                pil_image = Image.fromarray(frame_rgb)
                
                # Procesar imagen y hacer predicción
                with torch.no_grad():
                    processed_image = preprocess_image(pil_image).to(device)
                    outputs = model(processed_image)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Mostrar resultados
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(frame_rgb, caption=f"Frame {frame_idx}", use_container_width=True)
                
                with col2:
                    st.subheader("Predicción:")
                    pred_class = torch.argmax(probabilities).item()
                    confidence = probabilities[0][pred_class].item()
                    
                    st.write(f"Clase predicha: {class_names[pred_class]}")
                    st.write(f"Confianza: {confidence:.2%}")
                    
                    # Gráfico de barras de probabilidades
                    fig, ax = plt.subplots()
                    probs = probabilities[0].cpu().numpy()
                    sns.barplot(x=class_names, y=probs)
                    plt.xticks(rotation=45)
                    plt.title("Probabilidades por clase")
                    plt.xlabel("Clase")
                    plt.ylabel("Probabilidad")
                    plt.tight_layout()
                    st.pyplot(fig)
            
            cap.release()
            # Eliminar archivo temporal
            os.remove(temp_path)

    elif choice == "Entrenamiento":
        st.header("Entrenamiento del Modelo")
        
        # Input para la ruta de la carpeta
        folder_path = st.text_input("Ingrese la ruta de la carpeta con las imágenes de entrenamiento",
                                  help="La carpeta debe contener 6 subcarpetas, una por cada clase")
        
        if folder_path and os.path.exists(folder_path):
            if st.button("Entrenar Modelo"):
                with st.spinner("Preparando datos..."):
                    X_train, y_train, class_names = load_images_from_folder(folder_path)
                    
                st.info(f"Clases detectadas: {class_names}")
                st.info(f"Total de imágenes cargadas: {len(X_train)}")
                
                with st.spinner("Entrenando modelo..."):
                    model = create_model(len(class_names))
                    history = train_model(model, X_train, y_train)
                    
                    # Guardar modelo y nombres de clases
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'class_names': class_names
                    }, 'modelo_cnn.pth')
                    st.success("Modelo guardado exitosamente!")
                    
                    # Visualizar métricas
                    fig = plot_training_history(history)
                    st.pyplot(fig)
        else:
            if folder_path:
                st.error("La ruta especificada no existe")
            
    else:
        st.header("Predicción")
        
        if not os.path.exists('modelo_cnn.pth'):
            st.error("No se encontró un modelo entrenado. Por favor, entrene el modelo primero.")
            return
            
        uploaded_file = st.file_uploader("Cargar imagen para predicción", 
                                      type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", width=300)
            
            # Cargar modelo y nombres de clases
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load('modelo_cnn.pth')
            class_names = checkpoint['class_names']
            
            model = create_model(len(class_names))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            with torch.no_grad():
                processed_image = preprocess_image(image).to(device)
                outputs = model(processed_image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
            # Mostrar resultados
            st.subheader("Resultados de la predicción:")
            pred_class = torch.argmax(probabilities).item()
            confidence = probabilities[0][pred_class].item()
            
            st.write(f"Clase predicha: {class_names[pred_class]}")
            st.write(f"Confianza: {confidence:.2%}")
            
            # Gráfico de barras de probabilidades
            fig, ax = plt.subplots()
            probs = probabilities[0].cpu().numpy()
            sns.barplot(x=class_names, y=probs)
            plt.xticks(rotation=45)
            plt.title("Probabilidades por clase")
            plt.xlabel("Clase")
            plt.ylabel("Probabilidad")
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
