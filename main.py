from flask import Flask, render_template, send_file, Response, request, jsonify
import cv2
import uuid
import qrcode
from io import BytesIO
import numpy as np
import mediapipe as mp
import time
import subprocess


app = Flask(__name__)

# # O comando DOS que você deseja executar
# comando = "ngrok http 2204"

# # Use subprocess.run() para executar o comando
# resultado = subprocess.run(comando, shell=True, text=True, capture_output=True)

# # Imprima a saída do comando
# print(resultado.stdout)

# Inicializa a captura da webcam
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Definir a largura e a altura da imagem em pixels
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
foto_seq = 0
detectado = False
disparo = False
url_qr = "https://b39c-2804-431-cffe-4fab-5474-f8b3-44e6-a09e.ngrok-free.app/"
img_fundo = "static/images/mundosenai.png"

# Inicializa a contagem de tempo
start_time = None
foto_capturada = False

def msg_tela(image, msg):
    for i in range(5):
                for j in range(5):
                    cv2.putText(image, msg, (image.shape[1] // 2 + i, image.shape[0] // 2 + j), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 2)

# Função para exibir a foto e a webcam
def exibir_webcam(video):
    global foto_feed, start_time, espera, detectado, disparo
    
    while True:
        # Lê o quadro da webcam
        ret, image = video.read()
        
        # Espelha o frame
        image = cv2.flip(image, 1)
        if not detectado:
            if not espera:
                if detect_palma(image) or disparo:
                        
                    # Se a mão estiver levantada, inicie o contador de 5 segundos
                            
                    start_time = time.time()
                    detectado = True                        
        else:
            count = time.time() - start_time
            
            # Se o contador chegou a 5 segundos, tire uma foto
            if count > 5:
                capturarFoto()
                time.sleep(1)
                detectado = False
            else:    
                msg_tela(image,f"{5 - count:.0f}")
        
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
        # mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, landmark_drawing_spec=landmark_drawing_spec, connection_drawing_spec=connection_drawing_spec)
        
        
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
    global foto_feed, url_qr
    
    if request.method == 'POST':
        
        # Lê a opção de resolução escolhida pelo usuário
        resolution = request.form['resolution']
        url_qr = request.form['site_url']

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
    global nome_foto, foto_feed, foto_capturada, espera, foto_seq, detectado,disparo
    
    # Define as coordenadas do canto superior esquerdo e a largura e altura do retângulo de corte
    x = 74
    y = 220
    w = 900
    h = 550
    borda = 10
    
    success, frame = video.read()
    if success:
        
        if foto_seq < 3:
            # Capturar uma foto
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            # Redimensiona as imagens para o mesmo tamanho
            frame = cv2.resize(frame, (w, h))
            
            nome_foto = f'foto_{str(foto_seq)}.png'
            
            # Salvar a imagem com moldura
            cv2.imwrite(f'static/photos/{nome_foto}', frame)
            
            disparo = True
            
            foto_seq = foto_seq + 1
                    
            if foto_seq == 3:                
                # Cria uma imagem em branco para o painel de fotos
                bg_foto = np.zeros((h+borda, w+borda, 3), dtype=np.uint8)
                
                moldura = cv2.imread(img_fundo)
                nome_foto = f'foto_{str(uuid.uuid4())}.png'
                            
                
                
                for seq in range(3):
                    foto = cv2.imread(f'static/photos/foto_{str(seq)}.png')
                    moldura[y-borda//2:y+h+borda//2,x-borda//2:x+w+borda//2] = bg_foto
                    moldura[y:y+h,x:x+w] = foto
                    y = y + h + 15

                # Salvar a imagem com moldura
                cv2.imwrite(f'static/photos/{nome_foto}', moldura)
                
                foto_capturada = True
                disparo = False
                foto_seq = 0
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
    global nome_foto, url_qr
    if nome_foto: 
        # dados para o qr code
        #url_qr = "https://b39c-2804-431-cffe-4fab-5474-f8b3-44e6-a09e.ngrok-free.app/"
        #url_qr = request.host_url
        #data = f'{request.host_url}download/{nome_foto}'
        data = f'{url_qr}download/{nome_foto}'
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