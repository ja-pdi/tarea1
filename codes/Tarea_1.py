# Procesamiento Digital de Imagenes.
# Tarea 1.
# Jeckson Jaimes. 12-10446.
# Andres Suarez. 12-10925.

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Leer imagen
img = cv2.imread('../img/text1.png')

# Convertir a escala de grises
imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Se le aplica dilatación y erosión para disminuir el ruido
imgDi = cv2.dilate(imgGRAY, np.ones((3, 1), np.uint8), iterations=1)
imgEr = cv2.erode(imgDi, np.ones((3, 1), np.uint8), iterations=1)
# blur = cv2.GaussianBlur(imgGRAY,(3,1),0)

#  Se le aplica un treshold para obtener la imagen binaria 
thresh = cv2.adaptiveThreshold(imgEr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# thresh = cv2.Canny(thresh,100,200)
# laplacian = cv2.Laplacian(thresh,cv2.CV_8UC1)
# sobelx = cv2.Sobel(thresh,cv2.CV_64F,1,0,ksize=5)  # x
# sobely = cv2.Sobel(thresh,cv2.CV_64F,0,1,ksize=5)  # y

im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
r = 0
bandera = False
for i in range(len(contours)):
    cnt=contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    # Estoy intentando este if para que cuando detecte una i no pinte el otro pedazo y que el contador no se 
    # vuelva mierda pero no esta funcionando!
    if x+w<21:
        if bandera==False:
            bandera = True 
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            r = r + 1
        else: 
            bandera = False
    else:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        r = r + 1

print("El número de carácteres en la imagen es: " + str(r))

plt.subplot(1,3,1),
plt.imshow(imgGRAY,'gray')
plt.title('Imagen Original')
plt.subplot(1,3,2),
plt.imshow(img,'inferno')
plt.title('Imagen letras detectadas')
plt.subplot(1,3,3),
plt.imshow(thresh,'gray')
plt.title('Imagen Binaria sin ruido')
plt.show()
