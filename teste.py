import cv2
import numpy as np

# Lê as imagens
img1 = cv2.imread('static/photos/foto_1cfe9770-d8c2-46bf-a12c-0285e336480b.png')
img2 = cv2.imread('static/photos/foto_2c35c0a7-994a-4e63-88b5-2169071b2ef5.png')
img3 = cv2.imread('static/photos/foto_3f1c1847-092e-4ec7-8ed1-1c85570c220c.png')
img4 = cv2.imread('static/photos/foto_4ac68086-2aae-4f5d-8417-4d8f0295dd14.png')

# Redimensiona as imagens para o mesmo tamanho
img1 = cv2.resize(img1, (200, 200))
img2 = cv2.resize(img2, (200, 200))
img3 = cv2.resize(img3, (200, 200))
img4 = cv2.resize(img4, (200, 200))

# Cria uma imagem em branco para o painel de fotos
panel = np.zeros((410, 410, 3), dtype=np.uint8)

# Adiciona as imagens ao painel
panel[5:205, 5:205] = img1
panel[5:205, 205:405] = img2
panel[205:405, 5:205] = img3
panel[205:405, 205:405] = img4

# Mostra o painel de fotos
cv2.imshow('Panel', panel)
cv2.waitKey(0)
