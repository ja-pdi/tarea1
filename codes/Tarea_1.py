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

# Filtro Gaussiano
# Obtencion de los coeficientes
ksize = 3
sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8

imgFILT	=	cv2.GaussianBlur(imgGRAY, (3,3), 0)
plt.figure()
plt.subplot(1,6,1),
plt.imshow(imgGRAY,'gray')
plt.title('Imagen Original')
plt.subplot(1,6,2),
plt.imshow(imgFILT,cmap='Greys')
plt.title('Gauss')


# Thres OTSU GAUS
# thresh = cv2.threshold(imgFILT,0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)
# print(type(thresh))
# Bilateral Filtering


# ----------------------EROSION DILATACION

# # Erosión
# erosion = np.ones((2, 1), np.uint8)
# imgEr = cv2.erode(imgFILT, erosion, iterations=3)
# plt.subplot(1,6,3),
# plt.imshow(imgEr,cmap='Greys')
# plt.title('Erosionada')
# # Dilatacion 
# dilatacion = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
# imgDi = cv2.dilate(imgEr, dilatacion, iterations=1)
# plt.subplot(1,6,4),
# plt.imshow(imgDi,cmap='Greys')
# plt.title('Dilatada')

# imgDiEr = imgDi

# ---------------------DILATACION EROSION

# Dilatacion 
dilatacion = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
imgDi = cv2.dilate(imgFILT, dilatacion, iterations=1)
plt.subplot(1,6,3),
plt.imshow(imgDi,cmap='Greys')
plt.title('Dilatada')
# Erosión
erosion = np.ones((1, 2), np.uint8)
imgEr = cv2.erode(imgDi, erosion, iterations=1)
plt.subplot(1,6,4),
plt.imshow(imgEr,cmap='Greys')
plt.title('Erosionada')
 
imgDiEr = imgEr


# ESTO DE CLOSING ES PARA RELLENAR LAS LETRAS PERO NO ES NECESARIO
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# plt.subplot(1,5,5),
# plt.imshow(closing,cmap='Greys')
# plt.title('Closing')
# Closing

thresh = cv2.adaptiveThreshold(imgDiEr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
plt.subplot(1,6,5),
plt.imshow(thresh,cmap='Greys')
plt.title('Threshold')


# imgDi = cv2.dilate(imgGRAY, np.ones((3, 1), np.uint8), iterations=1)
# imgEr = cv2.erode(imgDi, np.ones((3, 1), np.uint8), iterations=1)
# blur = cv2.GaussianBlur(imgGRAY,(3,1),0)
#Bilateral Filter
# imgEr = cv2.bilateralFilter(imgDi,9,75,75)
#  Se le aplica un treshold para obtener la imagen binaria 
# thresh = cv2.threshold(imgEr,0,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)
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
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    # Estoy intentando este if para que cuando detecte una i no pinte el otro pedazo y que el contador no se 
    # vuelva mierda pero no esta funcionando!
    # if x+w<19:
    #     if bandera==False:
    #         bandera = True 
    #         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    #         r = r + 1
    #     else: 
    #         bandera = False
    # else:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    #     r = r + 1

plt.subplot(1,6,6),
plt.imshow(img,cmap='Greys')
plt.title('Contours')


print("El número de carácteres en la imagen es: " + str(len(contours)))
plt.show()

# plt.figure()
# plt.subplot(1,3,1),
# plt.imshow(imgGRAY,'gray')
# plt.title('Imagen Original')
# plt.subplot(1,3,2),
# plt.imshow(img,'inferno')
# plt.title('Imagen letras detectadas')
# plt.subplot(1,3,3),
# plt.imshow(thresh,'gray')
# plt.title('Imagen Binaria sin ruido')
# plt.show()
