# deteccion de lineas
# jose solis nc 1472

import cv2
import numpy as np
import os

# Obtener la ruta del directorio donde está guardado este script
dir_script = os.path.dirname(__file__)

# Construir la ruta absoluta hacia la imagen
ruta_imagen = os.path.join(dir_script, "../imagenes/loro.jpg")

# Cargar imagen
imagen = cv2.imread(ruta_imagen)

# Verificar que la imagen exista
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    print("Ruta intentada:", os.path.abspath(ruta_imagen))
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a tipo float32
gris_float = np.float32(gris)

# Detectar esquinas mediante Harris
esquinas = cv2.cornerHarris(
    gris_float,
    2,
    3,
    0.04
)

# Dilatar para hacer visibles las esquinas
esquinas = cv2.dilate(
    esquinas,
    None
)

# Crear copia
resultado = imagen.copy()

# Umbral para identificar esquinas
umbral = 0.01 * esquinas.max()

# Marcar esquinas
resultado[esquinas > umbral] = [0, 0, 255]

# Mostrar resultados
cv2.imshow(
    "Imagen Original - 1472",
    imagen
)

cv2.imshow(
    "Esquinas Detectadas (Loro) - 1472",
    resultado
)

# Rutas para guardar
ruta_original = os.path.join(dir_script, "../resultados/loro_original_1472.jpg")
ruta_resultado = os.path.join(dir_script, "../resultados/ejemplo2_esquinas_1472.jpg")

# Guardar resultados
cv2.imwrite(
    ruta_original,
    imagen
)

cv2.imwrite(
    ruta_resultado,
    resultado
)

# Contar esquinas aproximadas
cantidad_esquinas = np.sum(
    esquinas > umbral
)

# Impresiones en consola
print("Deteccion de esquinas terminada.")
print("Cantidad aproximada de puntos detectados:",
      cantidad_esquinas)

print("Resultados guardados en:")
print(ruta_original)
print(ruta_resultado)
print("jose solis nc 1472")

# Esperar una tecla
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()

print("jose solis nc 1472")