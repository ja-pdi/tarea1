# Procesamiento Digital de Imagenes.
# Laboratorio II.
# Jeckson Jaimes. 12-10446.
# Andres Suarez. 12-10925


import cv2
import matplotlib.pyplot as plt
import numpy as np

imgGREY = cv2.imread('/media/andres/Archivos/Personal/PDI/Tarea1/letra1.png',cv2.IMREAD_GRAYSCALE)
imgBGR = cv2.imread('/media/andres/Archivos/Personal/PDI/Tarea1/letra1.png')
#imgRGB = imgBGR[:,:,::-1]
#imgGREY	=	cv2.cvtColor(imgRGB,cv2.COLOR_RGB2GRAY)
print(imgGREY)
print(imgBGR)
plt.figure()

plt.title('imgRGB')
plt.imshow(imgGREY,cmap='Greys')


plt.show()


