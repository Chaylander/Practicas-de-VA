# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

Letra = "R"

contador=0

while contador<11:
    opcion = input("Escribe la inicial de mi nombre ")
    #print(opcion)
    
    if opcion == "R":
        
        contador=contador+1
        print (contador)
        
        if contador != 12:
            img1 = cv2.imread('monokuma_res.jpg', cv2.IMREAD_GRAYSCALE)
            cv2.imshow('Imagen 1',img1)
            img2 = cv2.imread('thewitcher_res.jpg', cv2.IMREAD_GRAYSCALE)
            cv2.imshow('Imagen 2',img2)
            cv2.moveWindow('Imagen 1', -16,0)
            cv2.moveWindow('Imagen 2', 1325,0)
            
        
            
            if contador == 1:
                print("Suma de imagenes")
                ressuma = cv2.addWeighted(img1,0.5,img2,0.5,1)
                cv2.imshow('suma',ressuma)
                
                #img = plt.imread('monokuma_res.jpg')
                plt.hist(img1.ravel())
                plt.show()
                thismanager = get_current_fig_manager()
                thismanager.window.SetPosition((50000, 111500))
                
                img1 = cv2.equalizeHist(img1)
                cv2.imshow('imagen ecualizada', img1)

                plt.hist(img1.ravel())
                plt.show()

                              
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
            elif contador == 2:
                print("Resta de imagenes")
                resresta = cv2.absdiff(img1,img2)
                print('img1[0,0]= ', img1[0,0])
                print('img2[0,0]= ',img2[0,0])
                print('resultado[0,0]= ',resresta[0,0])
                            
                cv2.imshow('resta',resresta)
                cv2.imwrite('resta.jpg',resresta)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            elif contador == 3:
                print("Division")
                
                #resdiv = cv2.divide(img1,0.5,img2,0.1,1)
                resdiv=cv2.divide(img1,img2)
                
                print('img1[0,0]= ', img1[0,0])
                print('img2[0,0]= ',img2[0,0])
                print('resultado[0,0]= ',resdiv[0,0])
                
                cv2.imshow('division',resdiv)
                cv2.imwrite('division.jpg',resdiv)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            elif contador == 4:
                print("Multiplicacion")
            
                resmul=cv2.multiply(img1,img2)
                
                print('img1[0,0]= ', img1[0,0])
                print('img2[0,0]= ',img2[0,0])
                print('resultado[0,0]= ',resmul[0,0])
                
                cv2.imshow('multiplicacion',resmul)
                cv2.imwrite('multiplicacion.jpg',resmul)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            elif contador == 5:
                print("Conjuncion")
            
                resconjuncion=cv2.bitwise_and (img1, img2)
                
                print('img1[0,0]= ', img1[0,0])
                print('img2[0,0]= ',img2[0,0])
                print('resconjuncion[0,0]= ',resmul[0,0])
                
                cv2.imshow('Conjuncion',resconjuncion)
                cv2.imwrite('Conjuncion.jpg',resconjuncion)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            elif contador == 6:
                print("Disyuncion")
    
                resdisyuncion=cv2.bitwise_or(img1, img2)
                
                print('img1[0,0]= ', img1[0,0])
                print('img2[0,0]= ',img2[0,0])
                print('resultado[0,0]= ',resmul[0,0])
                
                cv2.imshow('disyuncion',resdisyuncion)
                cv2.imwrite('disyuncion.jpg',resdisyuncion)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            elif contador == 7:
                print("Negacion")
    
                resnegacion=cv2.bitwise_not(img1)
                
                print('img1[0,0]= ', img1[0,0])
                print('resnegacion[0,0]= ',resnegacion[0,0])
                
                cv2.imshow('negacion',resnegacion)
                cv2.imwrite('negacion.jpg',resnegacion)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            elif contador == 8:
                print("Traslacion")
    
                img = cv2.imread('monokuma_res.jpg',0)
                rows,cols = img.shape
                
                M = np.float32([[1,0,210],[0,1,20]])
                dst = cv2.warpAffine(img,M,(cols,rows))
                
                cv2.imshow('img',dst)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            elif contador == 9:
                print("Escalado")
                img = cv2.imread("monokuma_res.jpg")
                newImg = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
                cv2.imwrite('escalado.jpg',newImg)
                cv2.imshow('Resized Cuadrito1', newImg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            
            elif contador ==10:
                print("Rotacion")
                img = cv2.imread('monokuma_res.jpg',0)
                rows,cols = img.shape
                
                M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
                dst = cv2.warpAffine(img,M,(cols,rows))
                cv2.imshow('Rotacion', dst)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            else :
                print('Traslacion a fin')
                
                
                img = cv2.imread('monokuma_res.jpg')
                rows,cols,ch = img.shape
            
    
                pts1 = np.float32([[100,400],[400,100],[100,100]])
                pts2 = np.float32([[50,300],[400,200],[80,150]])
    
    
                M = cv2.getAffineTransform(pts1,pts2)
                dst = cv2.warpAffine(img,M,(cols,rows))
    
                plt.subplot(121),plt.imshow(img),plt.title('Entrada')
                plt.subplot(122),plt.imshow(dst),plt.title('Salida')
                plt.show()
            
print("Ha terminado")
    
      
         
