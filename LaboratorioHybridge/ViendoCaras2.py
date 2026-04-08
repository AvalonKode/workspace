#Importamos la libreria de OpenCV para poder usar sus funciones y herramientas para el procesamiento de imagenes y video.

import cv2

#Con esta linea de codigo imprimimos la version de OpenCV que tenemos instalada, esto es util para verificar que tenemos la version correcta y para solucionar problemas de compatibilidad.
print(cv2.__version__)

#Primero cargamos el clasificador que usaremos para este proyecto, existen varios pero usaremos el clasificador de caracteristas Haar
#Creamos las variables que guardaran los clasificadores
clasificador_facial = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
# Cargamos el clasificador preentrenado de deteccion facial (Haar cASCADE)DESDE LA RUTA incluida en openCV, 
# contiene la ruta de la carpeta en donde esta guardado el clasificador.
#Basicamente estamos diciendo ve a la carpetade haarcascade de OpenCV y carga este archivo. 

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
#Esta linea es basicamento lo anterior, solo que cargamos el clasificador para la sonrisa.

clasificador_ojos = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#este es nuestro clasificador para los ojos.

#El siguiente paso consiste en indicar que la captura de video se inicialce desde la camar aintegrada o la que tengamos conctada.
#para ello, del mismo modo tenemos que hacerlo con la misma libreria y guardarlo en una variable. 

captura = cv2.VideoCapture(0)
#el 0 representa la camara que vamos a usar, en algunas ocasiones la camara no es la 0 por defecto, asi que tendremos quejugar un poco 
#con ese digito: puede ser 1, 2 etc. 

#Ahora usaremos un ciclo para tener control de la captura de la camara en general nos dice "Sigue tomando fotos TODO el tiempo, hasta que algo falle."
#recordemos que el video es una secuencia de imagenes o fotos tomadas.
while True:
    
    #captura = cv2.VideoCapture(0)
#el 0 representa la camara que vamos a usar, en algunas ocasiones la camara no es la 0 por defecto, asi que tendremos quejugar un poco 
#con ese digito: puede ser 1, 2 etc. 

#Ahora usaremos un ciclo para tener control de la captura de la camara en general nos dice "Sigue tomando fotos TODO el tiempo, hasta que algo falle."
#recordemos que el video es una secuencia de imagenes o fotos tomadas.

    exito, frame = captura.read()
    #tenemos nuestra variable "exito,frame + la funcion capture.read()" la funcion nos retorna dos valores:
    # "exito" = (True/False) que nos indicara si la camara capturo bien la imagen.
    #"frame" = (matriz de pixeles) la cual nos proporcionara la foto (la matriz se refiere a los valores que onbtendremos y podemos visualizar en la terminal.)
    #En resumen:
    #print(frame)
    #Lo que OpenCV captura son dos valores
    #Bool = True/False (exito) --> si la imagen se capturo correctamente
    #Matriz de pixeles (frame) ---> informacion que mi computadora captura con la camara. 
    
    if not exito or frame is None:
        print("Error al obtener imagen de la camara. Revisa permisos")
        break
    #esta condicion aplica si no tenemos exito(booleano) o frame (matriz de pixeles) es decir si la camara falla-no autoriza o no hay imagen 
    imagen_bn = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #CONVERTIMOS la imagen a blanco y negro, pues nuestro clasificador haar 
    #solo funciona con imagenes blanco y negro.
    
    #Determinamos las especificaciones para que nuestro clasificador sea mas preciso y evitemos los falsos positivos.
    rostro_detectado = clasificador_facial.detectMultiScale( #detectMultiScale busca objetos(caras) en diferentes tamanos dentro de la imagen
        imagen_bn,             #La imagen donde lo va a detectar (la variable)
        scaleFactor=1.1,       #La escala (scaleFactor) (que tan detallado miras)
        minNeighbors=5,        #Numero minimo de vecinos (que tan seguro estas)
        minSize=(60,60)        #Tama;o minimo del rostro. (que tan grandes deben de ser)
    )
    
    #scalefactor = controla como cambia el tamano de busqueda 1.0 --> el mismo tamano, 1.1 --> reduce en 10%(preciso pero lento), 1.2% --> reduce mas(rapido pero menos preciso).
    #1.1 ES EL ESTANDAR
    #minNeighbors = Controla que tan seguro debe estar el sistema antes de decir 'esto es una cara', si hay suficientes coincdencias lo acepta
    #Bajo "3" detecta mas cosas esto uncluye falsos positivos / alto "8" mas preciso, pro puede ignorar caras reales. RECOMENDACION: 5
    #minSize = tamano minimo del objeto a detectar es decir en esta caso ignorara objetos mas pequenos que 60x60 pixeles.
    #30x30 detecta caras lejanas pero es mas lento.
    #100x100 solo cara cercanas y es mas rapido.
    #60x60 es lo recomendable apra camaras web
    
    #TIP No detecta caras → baja minNeighbors / Detecta cosas raras → súbelo / No detecta caras lejanas → baja minSize
    
    #print(rostro_detectado) si usaramos este print obtendriamo una lsita con 4 valores o datos (matriz)
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
    #1. la coordenada en x del rostro 
    #2. la coordenada en y del rostro 
    #3. el largo del rostro (width)
    #4. el alto del rostro (height)
    
    #Recordemos que la imagen se percibe en un plano cartesiano  
    #en python el plano cartesiano cambia un poco a como lo conocemos tal cual,
    #en el eje de las y en lugar de irse a lo negativo mientras bajamos aumenta asi funciona en python (0, -10) seria en el convencional en python seria(0,10)
    
    for (x, y, w, h) in rostro_detectado:
        #eSTE CICLO FOR ES EL siguiente paso para la deteccion de rostros, lo que hace es recorrer la lista de rostros detectados y extraer las coordenadas y dimensiones de cada rostro.
        #Basicamente dice lo siguiente: Para cada cara deetectada, tomas sus coordenadas (x,y) y sus dimensiones (w,h) y haz algo con ellas.
        #volvamos a recordar que un video es una secuencia de imagenes, entonces cada vez que se detecta un rostro en una imagen, se ejecuta este ciclo for para procesar esa imagen en particular.
        
        
        #con estos print podriamos visualiazar las matricez separadas.
        # print(f'Coordenada en x {x}')
        # print(f'Coordenada en y {y}')
        # print('---------------------')
        # print(f'Largo del rostro {w}') --> largo
        # print(f'Alto del rostro {h}'). --> alto
        # print('&&&&&&&&&&&&&&&&&&&&&&')
        
        cv2.rectangle(
            frame,#la imagen donde se dibujara el rectangulo (en este caso el frame original a color)
            (x, y), #coordenadas del rectangulo (esquina superior izquierda)
            (x + w, y + h), #coordenadas de la esquina inferior derecha
            (255,23,129), #color del rectangulo en formato BGR (en este caso es un rosa fuerte. Color RGB --> BGR Recordemos que OpenCV usa el formato BGR en lugar de RGB osea al reves.
            10 #grosor del rectangulo (en este caso 10 pixeles)
        )
        
        roi_gray = imagen_bn[y:y+h, x:x+w] #region de interes en gris (la parte del rostro detectado)
        roi_color = frame[y:y+h, x:x+w] #region de interes a color (la parte del rostro detectado)
        
        # Roi = Region of Interest (Region de interes) es un termino que se usa para referirse a una parte especifica de la imagen que queremos analizar o procesar, en este caso es el rostro detectado.
        # Basicamente recorta la cara detectada para analizarla por separado, esto es util para mejorar la precision de los clasificadores de sonrisa y ojos, ya que se enfocan solo en la region del rostro. 
        
        # Sabemos que: x,y son las coordenadas de la esquina superior izquierda del rostro detectado, w es el ancho del rostro y h es el alto del rostro.
        #Por lo tanto con imagen[y:y+alto, x:x+largo] le estamos diciendo: Dame la parte de la imagen que va desde la coordenada y hasta y+h (alto) y desde la coordenada x hasta x+w (largo). El rectangulo donde esta la cara.
        
        # Y porque dos variables? roi_gray y roi_color? Porque el clasificador de sonrisa y ojos funciona mejor con imagenes en blanco y roi_color la usamos para dibujar con un color el rectangulo.
        
        
        
        #Creamos un par de variables para detectar sonrisas y ojos dentro de la region de interes (el rostro) usando los clasificadores correspondientes, y los caules ya cargamos al inicio del codigo.
        
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=20) #detectamos sonrisas dentro de la region de interes (el rostro)
        # smile = Usa clasificador de sonrisa.Busca sonrisas dentro del rostro detectado y guarda las coordenadas detectadas en la variable "smiles"
        #Sonrisas scaleFactor = 1.8 alto ---> menos escalas ----> mas rapido | Solo detecta sonrisas muy claras y evita falsos positivos. minNeighbors = 30 alto ---> mas preciso | Solo detecta sonrisas muy claras y evita falsos positivos.
        
        
        ojos = clasificador_ojos.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5) #detectamos ojos dentro de la region de interes (el rostro)
        # Ojos = Usa clasificador de ojos. Busca ojos dentro del rostro detectado y guarda las coordenadas detectadas en la variable "ojos"
        # Ojos scaleFactor = 1.5 alto ---> menos escalas ----> mas rapido | Solo detecta ojos muy claros y evita falsos positivos. minNeighbors = 15 alto ---> mas preciso | Solo detecta ojos muy claros y evita falsos positivos.
        #Los ojos son mas faciles de detectar que las sonrisas, por eso podemos usar un scaleFactor y minNeighbors mas bajos.
        
        
        
        
        #Los Siguientes ciclos for tienen una funcion muy parecida al ciclo for anterior, es decir, al de los rostros. pero este ciclo esta dentro del ciclo rotro, es decir que va a detectar algo unicamente dentro del 
        #rostro detectado, por eso estan dentro del ciclo for de rostros.
        
        for (sx, sy, sw,sh) in smiles:
            #Para cada sonrisa detectada (sx, sy ---> posicion | sw, sh --->tamano) en smiles que es la variable que guarda las coordenadas de las sonrisas detectadas, nuestra variable anteriormente creada. 
            cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh),(0,0,255),2)
            # cv2.rectangle Dibuja un rectángulo en la imagen
            # roi_color es la imagen donde se dibujara el rectangulo (en este caso la region de interes a color, es decir el rostro detectado)
            # (sx,sy) son las coordenadas de la esquina superior izquierda del rectangulo
            # (sx+sw,sy+sh) son las coordenadas de la esquina inferior derecha del rectangulo   
            # (0,0,255) es el color del rectangulo en formato BGR (en este caso es rojo)
            # 2 es el grosor del rectangulo (en este caso 2 pixeles
                   # = con este ciclo for lo que hacemos es dibujar un rectangulo rojo alrededor de cada sonrisa detectada dentro del rostro detectado.
            
            
        for (ox, oy, ow, oh) in ojos: 
            # Para cada ojo detectado (ox, oy ---> posicion | ow, oh --->tamano) en ojos que es la variable que guarda las coordenadas de los ojos detectados, nuestra variable anteriormente creada.
            cv2.rectangle(roi_color,(ox, oy), (ox+ow, oy+oh), (0,255,0,),2)    
            # cv2.rectangle Dibuja un rectángulo en la imagen
            # roi_color es la imagen donde se dibujara el rectangulo (en este caso la region de interes a color, es decir el rostro detectado)
            # (ox,oy) son las coordenadas de la esquina superior izquierda del rectangulo
            # (ox+ow,oy+oh) son las coordenadas de la esquina inferior derecha del rectangulo   
            # (0,255,0) es el color del rectangulo en formato BGR (en este caso es verde)
            # 2 es el grosor del rectangulo (en este caso 2 pixeles
            # = con este ciclo for lo que hacemos es dibujar un rectangulo verde alrededor de cada ojo detectado dentro del rostro detectado.
            
            #RECORDEMOS QUE CON roi_color Solo estamos dibujando dentro del rostro detectado, es decir que si hay una sonrisa o un ojo fuera del rostro detectado, no se dibujara nada.
            
    cv2.imshow('Captura tomada', frame) #Muestra la imagen con los rectangulos dibujados en una ventana llamada "Captura tomada"
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #Espera a que el usuario presione la tecla 'q' para salir del ciclo y cerrar la ventana.
        break
    
    
captura.release() #Libera la captura de video, es decir, libera la camara para que pueda ser usada por otras aplicaciones.
cv2.destroyAllWindows() #Cierra todas las ventanas abiertas por OpenCV. 
    
    
    
                   #Flujo del programa:
                     #Detecta rostro 👤
                           #↓
                      #Recorta ROI ✂️
                           #↓
                     #Detecta ojos 👀
                           #↓
                     #Detecta sonrisa 😄
                           #↓
                 #for → dibuja cuadros 🟩🟥
                 
