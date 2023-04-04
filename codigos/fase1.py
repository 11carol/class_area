from osgeo import gdal
import numpy as np
import matplotlib as plt
import cv2
import csv

dataset = gdal.Open('./fase/data/recorte_det1.png', gdal.GA_ReadOnly)



cols = dataset.RasterXSize
rows = dataset.RasterYSize
band = dataset.RasterCount
driver = dataset.GetDriver().LongName 
print('Imagem POLIGONO')
print(cols,rows,band,driver)

#Acessa o arquivo por banda
band_1 = dataset.GetRasterBand(1) 
band_2 = dataset.GetRasterBand(2) 
band_3 = dataset.GetRasterBand(3)

b1 = band_1.ReadAsArray()  
b2 = band_2.ReadAsArray()  
b3 = band_3.ReadAsArray()
img_1 = np.dstack((b1, b2, b3))
print(img_1.shape) 

geotransform = dataset.GetGeoTransform()
x_geo = geotransform[0]
y_geo = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]

print('Primeiro valor da contagem de pixel no eixo X =' , x_geo)
print('Primeiro valor da contagem de pixel no eixo Y =' ,y_geo)


arq = open("./fase/csv/data_shpTest.csv", "w", newline='', encoding='utf-8')
arq.write("X, Y, Cor(Classificação) \n")

for i in range(rows):
    for j in range(cols):
        if ((img_1[i][j][0] == 255) and (img_1[i][j][1] == 255) and (img_1[i][j][2] == 255)):
            #print("ok")
            arq.write("%s, %s, Branco \n"% (i,j));
        if ((img_1[i][j][0] == 0) and (img_1[i][j][1] == 1) and (img_1[i][j][2] == 135)):
            arq.write("%s, %s, Azul(Igapó) \n"% (i,j));
        if ((img_1[i][j][1] == 101) and (img_1[i][j][2] == 183) and (img_1[i][j][3] == 14)):
            arq.write("%s, %s, Verde(Floresta) \n"% (i,j));
            
        if ((img_1[i][j][1] == 196) and (img_1[i][j][2] == 127) and (img_1[i][j][3] == 146)):
            arq.write("%s, %s, Rosa(Zona de transição) \n"% (i,j));
        
        if ((img_1[i][j][1] == 131) and (img_1[i][j][2] == 0) and (img_1[i][j][3] == 0)):
            arq.write("%s, %s, Vermelho(Várzea) \n"% (i,j));
            #arq.write("%s, %s, %s, %s, %s \n"% (i, j, img_1[i][j][0], img_1[i][j][1], img_1[i][j][2]));

arq.close