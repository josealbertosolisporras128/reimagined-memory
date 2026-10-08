# deteccion de lineas
# jose solis nc 1472

import cv2
import numpy as np
import os

# Obtener la ruta absoluta del directorio donde reside este script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definir la ruta de la imagen 'loro.jpg' dentro de la carpeta 'imagenes'
ruta_imagen = os.path.join(script_dir, "..", "imagenes", "loro.jpg")

# Cargar la imagen del loro
imagen = cv2.imread(ruta_imagen)

# Verificar la carga de la imagen
if imagen is None:
    print(f"Error: no se pudo cargar la imagen en: {os.path.abspath(ruta_imagen)}")
    print("Asegúrate de que el archivo 'loro.jpg' esté dentro de la carpeta 'imagenes/'.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a tipo float32 para cv2.cornerHarris
gris_float = np.float32(gris)

# Detectar esquinas mediante Harris
esquinas = cv2.cornerHarris(gris_float, 2, 3, 0.04)

# Dilatar los puntos detectados para que se vean claramente
esquinas = cv2.dilate(esquinas, None)

# Crear una copia para marcar los resultados
resultado = imagen.copy()

# Definir el umbral para las esquinas con respuesta fuerte
umbral = 0.01 * esquinas.max()

# Marcar las esquinas encontradas en rojo (BGR)
resultado[esquinas > umbral] = [0, 0, 255]

# Mostrar las ventanas de resultados con "1472" al final del título
cv2.imshow("Imagen Original - 1472", imagen)
cv2.imshow("Esquinas Detectadas (Loro) - 1472", resultado)

# Crear directorio de resultados si no existe
carpeta_resultados = os.path.join(script_dir, "..", "resultados")
os.makedirs(carpeta_resultados, exist_ok=True)

# 1. Guardar la imagen original
ruta_original = os.path.join(carpeta_resultados, "loro_original_1472.jpg")
cv2.imwrite(ruta_original, imagen)

# 2. Guardar la imagen con las esquinas detectadas
ruta_resultado = os.path.join(carpeta_resultados, "ejemplo2_esquinas_1472.jpg")
cv2.imwrite(ruta_resultado, resultado)

# Mostrar contadores y confirmación en consola
cantidad_esquinas = np.sum(esquinas > umbral)
print("Detección de esquinas terminada.")
print(f"Cantidad aproximada de puntos detectados: {cantidad_esquinas}")
print("Imágenes guardadas en la carpeta 'resultados':")
print(f" - Original: {os.path.abspath(ruta_original)}")
print(f" - Resultado: {os.path.abspath(ruta_resultado)}")

# Esperar interacción de teclado para salir
cv2.waitKey(0)
cv2.destroyAllWindows()

print("jose solis nc 1472")