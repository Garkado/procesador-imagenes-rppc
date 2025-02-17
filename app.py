import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from skimage import filters

# Título de la aplicación
st.title("Procesador de Imágenes para Gestión Documental - RPPC Jalisco")

# Función para verificar si una imagen contiene texto
def contiene_texto(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, umbral = cv2.threshold(gris, 150, 255, cv2.THRESH_BINARY_INV)
    return np.sum(umbral) > 1000

# Función para verificar la resolución de la imagen
def verificar_resolucion(imagen):
    dpi = imagen.info.get('dpi', (72, 72))
    return 150 <= dpi[0] <= 250 and 150 <= dpi[1] <= 250

# Función para detectar imágenes borrosas
def es_borrosa(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gris, cv2.CV_64F).var()
    return laplacian_var < 100

# Función para detectar imágenes encimadas o cortadas
def es_encimada_o_cortada(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    bordes = filters.sobel(gris)
    return np.mean(bordes) < 0.1

# Función para procesar las imágenes
def procesar_imagenes(carpeta):
    reporte = []
    for root, _, files in os.walk(carpeta):
        for file in files:
            if file.lower().endswith('.jpg'):
                ruta_imagen = os.path.join(root, file)
                try:
                    with Image.open(ruta_imagen) as img:
                        imagen = cv2.imread(ruta_imagen)
                        if imagen is None:
                            continue

                        # Validar contenido de texto
                        if not contiene_texto(imagen):
                            continue

                        # Validar resolución
                        if not verificar_resolucion(img):
                            reporte.append([file, root, "Resolución fuera de rango"])
                            continue

                        # Validar borrosidad
                        if es_borrosa(imagen):
                            reporte.append([file, root, "Imagen borrosa"])
                            continue

                        # Validar encimada o cortada
                        if es_encimada_o_cortada(imagen):
                            reporte.append([file, root, "Imagen encimada o cortada"])
                            continue

                        # Si pasa todas las validaciones
                        reporte.append([file, root, "Válida"])

                except Exception as e:
                    reporte.append([file, root, f"Error al procesar: {str(e)}"])

    return reporte

# Interfaz para cargar imágenes
st.header("Cargar Imágenes")
carpeta = st.text_input("Ingresa la ruta de la carpeta con las imágenes (por ejemplo, C:\\ruta\\a\\carpeta):")

if st.button("Procesar Imágenes"):
    if os.path.exists(carpeta):
        st.write(f"Procesando imágenes en la carpeta: {carpeta}")
        reporte = procesar_imagenes(carpeta)
        df = pd.DataFrame(reporte, columns=["Archivo", "Carpeta", "Estado"])
        st.write("Reporte de Imágenes:")
        st.dataframe(df)

        # Guardar el reporte en un archivo CSV
        output_csv = "reporte_imagenes.csv"
        df.to_csv(output_csv, index=False)
        st.write(f"Reporte guardado en {output_csv}")

        # Botón para descargar el reporte
        with open(output_csv, "rb") as file:
            st.download_button(
                label="Descargar Reporte",
                data=file,
                file_name=output_csv,
                mime="text/csv"
            )
    else:
        st.error("La carpeta no existe. Por favor, verifica la ruta.")