# class_area
 

# Classificação de Áreas


## Importando Bibliotecas



```python
from osgeo import gdal
gdal.UseExceptions()
import numpy as np
import matplotlib as plt
#import matplotlib.image as mpimg
from sklearn.tree import DecisionTreeClassifier
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\least_angle.py:30: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      method='lar', copy_X=True, eps=np.finfo(np.float).eps,
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\least_angle.py:167: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      method='lar', copy_X=True, eps=np.finfo(np.float).eps,
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\least_angle.py:284: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_Gram=True, verbose=0,
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\least_angle.py:862: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_X=True, fit_path=True,
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\least_angle.py:1101: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_X=True, fit_path=True,
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\least_angle.py:1127: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, positive=False):
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\least_angle.py:1362: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      max_n_alphas=1000, n_jobs=None, eps=np.finfo(np.float).eps,
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\least_angle.py:1602: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      max_n_alphas=1000, n_jobs=None, eps=np.finfo(np.float).eps,
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\least_angle.py:1738: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_X=True, positive=False):
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\decomposition\online_lda.py:29: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      EPS = np.finfo(np.float).eps
    

## Abrindo as amostras para classificação


```python
filenametiff = "data/recorte_tiff.png"
filenameshp = "data/recorte_shp.png"
datasettiff = gdal.Open(filenametiff, gdal.GA_ReadOnly)
datasetshp = gdal.Open(filenameshp, gdal.GA_ReadOnly) 

cols_tiff = datasettiff.RasterXSize
rows_tiff = datasettiff.RasterYSize
bands_tiff = datasettiff.RasterCount
driver_tiff = datasettiff.GetDriver().LongName #Verif
print('Imagem TIFF')
print(cols_tiff,rows_tiff,bands_tiff,driver_tiff)

cols_shp = datasetshp.RasterXSize
rows_shp = datasetshp.RasterYSize
bands_shp = datasetshp.RasterCount
driver_shp = datasetshp.GetDriver().LongName #Verif
print('Imagem SHAPEFILE')
print(cols_shp,rows_shp,bands_shp,driver_shp)


```

    Imagem TIFF
    2412 2340 4 Portable Network Graphics
    Imagem SHAPEFILE
    2392 2380 4 Portable Network Graphics
    


```python
geotransformtiff = datasettiff.GetGeoTransform() #tranforma as coordenadas de pixel em espaço geográfico 
print(geotransformtiff)

geotransformshp = datasetshp.GetGeoTransform() 
print(geotransformshp)
```

    (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    


```python
originX = geotransformtiff[0]
originY = geotransformtiff[3]
pixelWidth = geotransformtiff[1]
pixelHeight = geotransformtiff[5]

print('OriginX=' , originX)
print('OriginY=' ,originY)
print('pixelWidth=' ,pixelWidth)
print('pixelHeight=' ,pixelHeight)

originX = geotransformshp[0]
originY = geotransformshp[3]
pixelWidth = geotransformshp[1]
pixelHeight = geotransformshp[5]

print('OriginX=' , originX)
print('OriginY=' ,originY)
print('pixelWidth=' ,pixelWidth)
print('pixelHeight=' ,pixelHeight)

```

    OriginX= 0.0
    OriginY= 0.0
    pixelWidth= 1.0
    pixelHeight= 1.0
    OriginX= 0.0
    OriginY= 0.0
    pixelWidth= 1.0
    pixelHeight= 1.0
    


```python
#Separando as bandas da imagem tiff
band1_tiff = datasettiff.GetRasterBand(1)
data1_tiff = band1_tiff.ReadAsArray(0, 0, cols_tiff, rows_tiff)

band2_tiff = datasettiff.GetRasterBand(2)
data2_tiff = band2_tiff.ReadAsArray(0, 0, cols_tiff, rows_tiff)

band3_tiff = datasettiff.GetRasterBand(3)
data3_tiff = band3_tiff.ReadAsArray(0, 0, cols_tiff, rows_tiff)
#Montando o array com os valores do RGB
rgb_tiff = np.dstack((data1_tiff, data2_tiff, data3_tiff))

print (rgb_tiff)
```

    [[[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     ...
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]]
    


```python
#Separando as bandas da imagem shp
band1_shp = datasetshp.GetRasterBand(1)
data1_shp = band1_shp.ReadAsArray(0, 0, cols_shp, rows_shp)

band2_shp = datasetshp.GetRasterBand(2)
data2_shp = band2_shp.ReadAsArray(0, 0, cols_shp, rows_shp)

band3_shp = datasetshp.GetRasterBand(3)
data3_shp = band3_shp.ReadAsArray(0, 0, cols_shp, rows_shp)
#Montando o array com os valores do RGB
rgb_shp = np.dstack((data1_shp,data2_shp,data3_shp))

print (rgb_shp)
```

    [[[255 255 255]
      [114 210  19]
      [114 210  19]
      ...
      [ 94 225   9]
      [ 94 225   9]
      [ 94 225   9]]
    
     [[255 255 255]
      [114 210  19]
      [114 210  19]
      ...
      [ 94 225   9]
      [ 94 225   9]
      [ 94 225   9]]
    
     [[255 255 255]
      [114 210  19]
      [114 210  19]
      ...
      [ 94 225   9]
      [ 94 225   9]
      [ 94 225   9]]
    
     ...
    
     [[255 255 255]
      [170 215  74]
      [170 215  74]
      ...
      [ 92 222   0]
      [ 92 222   0]
      [ 92 222   0]]
    
     [[255 255 255]
      [165 225  64]
      [165 225  64]
      ...
      [ 87 220   0]
      [ 87 220   0]
      [ 87 220   0]]
    
     [[255 255 255]
      [165 225  64]
      [165 225  64]
      ...
      [ 87 220   0]
      [ 87 220   0]
      [ 87 220   0]]]
    


```python
#np.savetxt('test.txt', rgb,fmt='%s',delimiter=",")
arq = open("data.csv", "w")
arq.write("X, Y, B1_ORIGINAL, B2_ORIGINAL, B3_ORIGINAL, CLASSIFICACAO \n")

preservada = 0; #false

for i in range(rows_tiff):
    for j in range(cols_shp):
        #if((rgb_tiff[i][j][0] == 255) and (rgb_tiff[i][j][0] == 255) and (rgb_tiff[i][j][0] == 255)):
        #arq.write("%s, %s, [%s-%s-%s]\n"% (i,j,rgb_tiff[i][j][0],rgb_tiff[i][j][1],rgb_tiff[i][j][2]));
        if ((rgb_tiff[i][j][0] != 255) and (rgb_tiff[i][j][1] != 255) and (rgb_tiff[i][j][2] != 255)):
            
            if((rgb_shp[i][j][0] == 9) and (rgb_shp[i][j][1] == 1) and (rgb_shp[i][j][2] == 255)):
                arq.write("%s, %s, %s, %s, %s,Água \n"% (i,j,rgb_tiff[i][j][0],rgb_tiff[i][j][1],rgb_tiff[i][j][2]));
                preservada = 1; # Area preservada
                #print(preservada)
                
            '''if ((rgb_shp[i][j][0] == 1) and (rgb_shp[i][j][1] == 2) and (rgb_shp[i][j][2] == 14)):
                arq.write("%s, %s, %s, %s, %s, %s,Várzea \n"% (i,j,rgb_tiff[i][j][0],rgb_tiff[i][j][1],rgb_tiff[i][j][2]));

            elif ((rgb_shp[i][j][0] == 0) and (rgb_shp[i][j][1] == 54) and (rgb_shp[i][j][2] == 210)):
                arq.write("%s, %s, %s, %s, %s, %s,Igapó \n"% (i,j,rgb_tiff[i][j][0],rgb_tiff[i][j][1],rgb_tiffl[i][j][2]));

            elif ((rgb_shp[i][j][0] == 6) and (rgb_shp[i][j][1] == 9) and (rgb_shp[i][j][2] == 49)):
                arq.write("%s, %s, %s, %s, %s, %s,Zona de Trnsição \n"% (i,j,rgb_tiff[i][j][0],rgb_tiff[i][j][1],rgb_tiff[i][j][2]));'''

            if ((rgb_shp[i][j][0] == 252) and (rgb_shp[i][j][1] == 158) and (rgb_shp[i][j][2] == 8)):
                arq.write("%s, %s, %s, %s, %s,Solo exposto \n"% (i,j,rgb_tiff[i][j][0],rgb_tiff[i][j][1],rgb_tiff[i][j][2]));
                
            if ((rgb_shp[i][j][0] == 0) and (rgb_shp[i][j][1] == 86) and (rgb_shp[i][j][2] == 44)):
                arq.write("%s, %s, %s, %s, %s,Floresta \n"% (i,j,rgb_tiff[i][j][0],rgb_tiff[i][j][1],rgb_tiff[i][j][2]));

        preservada = 0

arq.close()

#plt.figure(1)
#plt.imshow(rgb_Original, interpolation='nearest')
#plt.grid()
#plt.figure(2)
#plt.imshow(rgb_TC, interpolation='nearest')
#plt.grid()
#plt.figure(3)
#plt.imshow(rgb_Passada, interpolation='nearest')
#plt.grid()
#plt.show()


```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-18-6c7b19261706> in <module>
         11         if ((rgb_tiff[i][j][0] != 255) and (rgb_tiff[i][j][1] != 255) and (rgb_tiff[i][j][2] != 255)):
         12 
    ---> 13             if((rgb_shp[i][j][0] == 9) and (rgb_shp[i][j][1] == 1) and (rgb_shp[i][j][2] == 255)):
         14                 arq.write("%s, %s, %s, %s, %s,Água \n"% (i,j,rgb_tiff[i][j][0],rgb_tiff[i][j][1],rgb_tiff[i][j][2]));
         15                 preservada = 1; # Area preservada
    

    IndexError: index 742 is out of bounds for axis 0 with size 742



```python


```


```python

```


```python

```
