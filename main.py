from flask import Flask, render_template, send_file, Response, request, jsonify
import cv2
import uuid
import qrcode
from io import BytesIO
import numpy as np
import mediapipe as mp
import time

app = Flask(__name__)

# Inicializa a captura da webcam
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Definir a largura e a altura da imagem em pixels
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Inicializa o rastreador de mãos
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Criar um objeto para desenhar as anotações nas mãos detectadas
mp_drawing = mp.solutions.drawing_utils

# Criar um objeto DrawingSpec para os landmarks
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

# Criar um objeto DrawingSpec para as conexões
connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

#Declara variaveis globais
nome_foto = ""
foto_feed = False
espera = False

# Inicializa a contagem de tempo
start_time = None
foto_capturada = False

# Define as coordenadas do canto superior esquerdo e a largura e altura do retângulo de corte
x = 420
y = 0
w = 1080
h = 1080

# Função para exibir a foto e a webcam
def exibir_webcam(video):
    global foto_feed, start_time, espera
    detectado = False
    
    while True:
        # Lê o quadro da webcam
        ret, image = video.read()
        
        if foto_feed:
            # Corta a imagem
            image = image[y:y+h, x:x+w]
            
        # Rotaciona o frame
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Espelha o frame
        image = cv2.flip(image, 1)
        print (f'detectado: {detectado}')
        print (f'espera: {espera}')
        if not detectado:
            if not espera:
                if detect_palma(image):
                    # Se a mão estiver levantada, inicie o contador de 5 segundos
                    start_time = time.time()
                    detectado = True                        
        else:
            for i in range(5):
                for j in range(5):
                    cv2.putText(image, f"{time.time() - start_time:.0f}", (image.shape[1] // 2 + i//2, image.shape[0] // 2 + j//2), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 2)

            # Se o contador chegou a 5 segundos, tire uma foto
            if time.time() - start_time > 5:
                capturarFoto()
                detectado = False
        
        ret, png = cv2.imencode('.png', image)
        
        frame = png.tobytes()

        # Atualiza o elemento canvas com a nova imagem
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n\r\n')
        
# Função para detectar a palma da mão
def detect_palma(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # for 
        hand_landmarks = results.multi_hand_landmarks[0]
            # Lógica para detecção da palma da mão
            # Desenhar as anotações nas mãos detectadas
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, landmark_drawing_spec=landmark_drawing_spec, connection_drawing_spec=connection_drawing_spec)
        
        
        # Obter as coordenadas dos landmarks do dedo indicador e do polegar
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

        if (
            thumb_tip.y > index_tip.y and
            index_tip.y > middle_tip.y and
            thumb_tip.y > ring_tip.y and
            thumb_tip.y > pinky_tip.y
        ):
            return True

    return False

@app.route('/')
@app.route('/index')
def index():
    
    return render_template('index.html')


@app.route('/configuracao', methods=['GET', 'POST'])
def configuracao():
    global foto_feed
    
    if request.method == 'POST':
        
        # Lê a opção de resolução escolhida pelo usuário
        resolution = request.form['resolution']

        # Define a largura e altura da resolução da webcam com base na opção escolhida pelo usuário
        if resolution == 'feed':
            foto_feed = True
        else:
            foto_feed = False

        return render_template('index.html')
    else:
        return render_template('configuracao.html')

@app.route('/download/<filename>')
def download_image(filename):
    photo = f'static/photos/{filename}'
    return send_file(photo, as_attachment=True)

@app.route('/video_feed')        
def video_feed():
    global video
    return Response(exibir_webcam(video),mimetype='multipart/x-mixed-replace; boundary=frame')

#captura foto
@app.route('/captura_foto')
def capturarFoto():
    global nome_foto, foto_feed, foto_capturada, espera
    success, frame = video.read()
    if success:
        
        if foto_feed:
            # Corta a imagem
            frame = frame[y:y+h, x:x+w]
                
        # Rotaciona o frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Capturar uma foto
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        
        # frame = frame[y:y+h, x:x+w]
        
        nome_foto = f'foto_{str(uuid.uuid4())}.png'
        
        # Ler a imagem da moldura com o parâmetro cv2.IMREAD_UNCHANGED para preservar a transparência
        moldura = cv2.imread('static/images/moldura.png', cv2.IMREAD_UNCHANGED)
        
        # Separar os canais da imagem da moldura em BGR e alfa
        b, g, r, a = cv2.split(moldura)

        # Criar uma máscara binária usando o canal alfa
        mask = a / 255
        
        # Verificar se a máscara tem o mesmo tamanho que a imagem de origem
        if moldura.shape != frame.shape[:2]:
            # Redimensionar a imagem da moldura e a máscara para o mesmo tamanho da imagem da foto
            moldura = cv2.resize(moldura, (frame.shape[1], frame.shape[0]))
        
        # Verificar se a máscara tem o mesmo tamanho que a imagem de origem
        if mask.shape != frame.shape[:2]:
            # Redimensionar a máscara para o mesmo tamanho da imagem de origem
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
        # Verificar se o tipo da máscara é CV_8U ou CV_8S
        if mask.dtype != np.uint8:
            # Converter a máscara para o tipo CV_8U
            mask = mask.astype(np.uint8)
            
        # Aplicar a máscara na imagem da foto usando cv2.bitwise_and()
        frame_masked = cv2.bitwise_and(frame, frame, mask=1-mask)

        # Aplicar a máscara invertida na imagem da moldura usando cv2.bitwise_and()
        moldura_masked = cv2.bitwise_and(moldura, moldura, mask=mask)

        # Combinar as duas imagens usando cv2.add()
        frame_with_frame = cv2.add(frame_masked, moldura_masked)

        # Salvar a imagem com moldura
        cv2.imwrite(f'static/photos/{nome_foto}', frame_with_frame)
        
        foto_capturada = True
        espera = True
        
        
    return nome_foto
    # Serve a foto capturada para o frontend
    # return send_file(f'static/photos/{nome_foto}', mimetype='image/png')
    
@app.route('/captured_image')
def captured_image():
    global nome_foto
    # Serve a foto capturada para o frontend
    return send_file(f'static/photos/{nome_foto}', mimetype='image/png')

@app.route('/captured')
def captured():
    global foto_capturada
    fotocap=foto_capturada
    foto_capturada = False
    
    # Retorna o estado atual do reconhecimento de gestos
    return jsonify(foto_capturada=fotocap)

@app.route('/qr')
def qr():
    global nome_foto
    if nome_foto: 
        # dados para o qr code
        data = f'{request.host_url}download/{nome_foto}'
        # gerar a imagem do qr code
        img = qrcode.make(data)
        # converter a imagem em bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        # retornar a imagem como resposta HTTP
    return Response(img_bytes, mimetype='image/png')

@app.route('/esperadetect')
def esperadetect():
    global espera
    espera = False
    return 'ok'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)

# Liberar a câmera e fechar as janelas
video.release()