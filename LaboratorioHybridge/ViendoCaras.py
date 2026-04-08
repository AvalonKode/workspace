#Comunidad
#openCV --- VISION POR COMPUTADDORA MUY SENCILLa
#importamos la libreria cv2 con la palabra reservada import
import cv2
print(cv2.__version__)

# Clasificador de características Haar, este clasificador lo vamos a cargar desde nuestra libreria cv2. Haar es el etipo de clasificador facial que usaremos

clasificador_facial = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#de la libreira le decimos que empice a capturar video por medio de la camar aintegrada a eso se refiere el 0 
#0 camara por defecto. en caso de que no tengamos camara integrada, debemos probar cambiando el id 0 a  o 2 etc 
#captura = cv2.VideoCapture(0) #Estamos capturando un unico frame(estamos tomando una foto) y lo guardamos en una variable de nombre captura 

#print(captura)
#ahora podemos obtener los datos RGB como ¿? 
#creamos una nueva variable 
#contenido_real = captura.read()
#print(contenido_real)





# Instalar la librería que ibamos a utilizar opencv-python

# Importante: Instalar != Utilizar

# Importación completa
# Importación por módulos


# Sabe qué: Treaete esta librería al código
import cv2
print(cv2.__version__)
# Clasificador de características Haar, este clasificador lo vamos a cargar desde nuestra libreria cv2. Haar es el etipo de clasificador facial que usaremos

clasificador_facial = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# De la librería, le decimos que empiece a capturar video por medio de la cámara
# integrada

# 0: Cámara por defecto

while True:
    # Capturar una foto
    captura = cv2.VideoCapture(0) # -> Estamos capturando un único frame (Estamos tomando una foto)

    contenido_real = captura.read()

    # Contenido real es una tupla

    #print(contenido_real[1]) #con este print, podemos visualizar l oque gusradamos en nuestra variable contenido_real que es igual a lo que visualiza la camar aen tiempo real. tampoco es necesario para el ejercicio asi que va. acomentario
    # Lo que cv2 Captura son dos valores
    # Bool                -> Si la imagen se capturó correctamente
    # Matríz de pixeles   -> La información que mi computadora captura de la cámara
    #negativo = 255 - contenido_real[1] #Con esto lo que hacemos es restarle 255 pixeles a nuestra imagen real y con ello obtendremos la imagen en negativo. es mero ejemplo no es necesario para lo que requerimos asi que lo pondre en comentario
    
    
    imagen_bn = cv2.cvtColor(contenido_real[1], cv2.COLOR_BGR2GRAY) #convertimos a BN lo cual es necesario para el reconocimineto facial


    #señalamos los requisitos para que el reconociemineto funcione y no no s arroje falsos positivos por lo menos deben de ser 5
    rostro_detectado = clasificador_facial.detectMultiScale(
        imagen_bn,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60,60)
    )

    # La imagen donde lo va a detectar (ByN)
    # La escala (scalaFactor)
    # Número mínimo de vecinos (minNeighbors)
    # Tamaño mínimo del rostro (minSize)
    
    #print(rostro_detectado) una vez que convertimos los "datos" obtenidos ya no es necesario este print, ya que el otro nos los da de manera ordenada, asi que lo comentamos 
    #rosotro_detectado nos arroja como resultado una lista con 4 valores 
     #print(rostro_detectado)
    # rostro_detectado nos arroja como resultado una lista con 4 valores:
    # 1.- La coordenada en x del rostro
    # 2.- La coordenada en y del rostro
    # 3.- El largo de rostro (weidht)
    # 4.- El alto del rostros (height)
    #asi se ve impreso en la terminal 
    # [[851 300 472 472]]
    
    #estos datos son arrojados como matricez: 
     #Una matriz es una lsita de listas
#matriz = [[1,2],
#         [3,4],
#         [5,6],
#         [7,8],
#         [9,10],
#         [11,12]]

# El tamaño de la lista general es el número de filas

# El tamaño de cada lista individual es el número de columnas


# Diseñen un programa que obtenga la salida
#1
#2
#3
#4

#for valor_1, valor_2 in matriz:
#   print(valor_1)
#   print(valor_2)
    
    #con este ciclo For podemos obtener las salidas pedidas, en este ejemplo lo estamos aplicando con la lista de la matriz.
    
    
    
    for (x, y, largo, alto) in rostro_detectado: #con este ciclo y prints obtendremos los datos que captura la camara de manera mas ordenada
        print(f'Coordenada en x {x}')
        print(f'Coordenada en y {y}')
        print('---------------------')
        print(f'Largo del rostro {largo}')
        print(f'Alto del rostro {alto}')
        print('&&&&&&&&&&&&&&&&&&&&&&')
        
        
        #con esta linea creamos el rectangulo el cual delimita el rostro a capturar
        cv2.rectangle(contenido_real[1], #la imagen es decir nuestra variable que conteiene la imagen capturada
            (x, y), #punto 1
            (x+largo, y+alto), #punto 2
            (255, 0, 0), 2) # color ---- RGB / BGR
    
    #en python el plano cartesiano cambia un poco a como lo conocemos tal cual, en el eje de las y en lugar de irse a lo negativo mientras bajamos aumenta asi funciona en python (0, -10) seria en el convencional en python seria(0,10)

    # Vamos a utilizar una función que hace la recontrucción de manera automática
    cv2.imshow('Captura tomada', contenido_real[1])
  
    
    
    # Detecta si el usuario presiona la tecla "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    
    
    

# El nombre de la imagen
# La información de la imagen


# Image Show


# RGB -> Red Green Blue



