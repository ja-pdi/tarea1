# Procesamiento Digital de Imagenes.
# Laboratorio II.
# Jeckson Jaimes. 12-10446.
# Andres Suarez. 12-10925


import cv2
import matplotlib.pyplot as plt
import numpy as np

# Leer imagen
imgBGR = cv2.imread('C:/Users/Andres/Documents/PDI/tarea1/img/Prueba12.png')
imgBGR = imgBGR[0:205,...]

# Convertir a escala de grises
imgGREY = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
plt.figure(),
plt.imshow(imgGREY,cmap='Greys')
plt.title('Gris')

# Binarizacion de la img
ret,thresh = cv2.threshold(imgGREY, 127,255,0)
plt.figure(),
plt.imshow(thresh,cmap='Greys')
plt.title('Threshold')

#Erosion
erosion = cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
imgEr = cv2.erode(thresh, erosion, iterations=1)
plt.figure(),
plt.imshow(imgEr,cmap='Greys')
plt.title('Erosionada')

# Laplacian Filter
imgFILT = cv2.Laplacian(imgEr,cv2.CV_8U)
plt.figure(),
plt.imshow(imgFILT,cmap='Greys')
plt.title('Laplacian')

# Dilatacion 
dilatacion = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
imgDi = cv2.dilate(imgFILT, dilatacion, iterations=1)
plt.figure(),
plt.imshow(imgDi,cmap='Greys')
plt.title('Dilatada')

img, contours, hierarchy = cv2.findContours(imgDi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

i = 0
for i in range(len(contours)):
    cnt=contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(imgBGR,(x,y),(x+w,y+h),(255,0,0),1)

plt.figure(),
plt.imshow(imgBGR)
plt.title('Contours')
print("El número de carácteres en la imagen es: " + str(len(contours)))

plt.show()


