import cv2
import mediapipe as mp
import numpy as np
import math
import os
from datetime import datetime

class DetectorAngulosComAcoes:
    def __init__(self, base_path=None):
        # Caminho base das imagens (ajuste aqui se necessário)
        # Ex.: r"C:\Users\25254365\Desktop\aaa"
        self.base_path = base_path or r"C:\Users\25254365\Desktop\aaa"

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Histórico para detecção de gestos sustentados
        self.gesto_ativo = None
        self.contador_gesto = 0
        self.threshold_gesto = 30  # ~1s em 30fps
        
        # Nomes padronizados dos arquivos (evita acentos e espaços)
        # Certifique-se de que esses arquivos existem na pasta base_path
        self.arquivos_imagens = {
            "PINCA": "pinca.png",
            "AGUDO": "agudo.png",
            "RETO": "reto.png",
            "OBTUSO": "obtuso.png",
            "ABERTO": "aberto.png"
        }
        
        # Cache de imagens carregadas
        self.imagens_cache = {nome: None for nome in self.arquivos_imagens.values()}
        
        # Mapa de gestos
        self.gestos = {
            "PINCA": {
                "angulo_min": 0,
                "angulo_max": 30,
                "descricao": "Dedos muito próximos (pinça)",
                "cor": (0, 0, 255),  # Vermelho
                "imagem": self.arquivos_imagens["PINCA"]
            },
            "AGUDO": {
                "angulo_min": 30,
                "angulo_max": 60,
                "descricao": "Ângulo agudo",
                "cor": (0, 165, 255),  # Laranja
                "imagem": self.arquivos_imagens["AGUDO"]
            },
            "RETO": {
                "angulo_min": 60,
                "angulo_max": 90,
                "descricao": "Ângulo reto (90°)",
                "cor": (0, 255, 0),  # Verde
                "imagem": self.arquivos_imagens["RETO"]
            },
            "OBTUSO": {
                "angulo_min": 90,
                "angulo_max": 120,
                "descricao": "Ângulo obtuso",
                "cor": (255, 0, 0),  # Azul (BGR)
                "imagem": self.arquivos_imagens["OBTUSO"]
            },
            "ABERTO": {
                "angulo_min": 120,
                "angulo_max": 181,  # incluir 180
                "descricao": "Dedos bem abertos",
                "cor": (255, 255, 0),  # Ciano (BGR)
                "imagem": self.arquivos_imagens["ABERTO"]
            }
        }
        
        # Carregar imagens
        self.carregar_imagens()
    
    def _tentativas_caminho(self, nome_arquivo):
        # 1) Caminho absoluto combinado com base_path
        if self.base_path:
            yield os.path.join(self.base_path, nome_arquivo)
        # 2) Caminho relativo à pasta do script
        pasta_script = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        yield os.path.join(pasta_script, nome_arquivo)
        # 3) Caminho relativo ao diretório corrente
        yield os.path.join(os.getcwd(), nome_arquivo)

    def carregar_imagens(self):
        """Carrega as imagens em cache para exibição em tempo real"""
        print("Carregando imagens...")
        for nome_arquivo in list(self.imagens_cache.keys()):
            caminho_encontrado = None
            for caminho in self._tentativas_caminho(nome_arquivo):
                if os.path.exists(caminho):
                    caminho_encontrado = caminho
                    break
            if not caminho_encontrado:
                print(f"[AVISO] Arquivo não encontrado: {nome_arquivo}. Verifique o caminho/pasta.")
                self.imagens_cache[nome_arquivo] = None
                continue
            img = cv2.imread(caminho_encontrado)
            if img is None:
                print(f"[ERRO] Falha ao abrir: {caminho_encontrado}")
                self.imagens_cache[nome_arquivo] = None
                continue
            # Redimensionar para caber na janela (400x300) mantendo proporção
            alvo_w, alvo_h = 400, 300
            h, w = img.shape[:2]
            escala = min(alvo_w / w, alvo_h / h)
            novo_w, novo_h = int(w * escala), int(h * escala)
            img_red = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_AREA)
            # Colocar em canvas 400x300 com bordas
            canvas = np.zeros((alvo_h, alvo_w, 3), dtype=np.uint8)
            y_off = (alvo_h - novo_h) // 2
            x_off = (alvo_w - novo_w) // 2
            canvas[y_off:y_off+novo_h, x_off:x_off+novo_w] = img_red
            self.imagens_cache[nome_arquivo] = canvas
            print(f"[OK] Imagem carregada: {nome_arquivo} -> {caminho_encontrado}")
        print("Carregamento de imagens concluído.\n")
    
    def calcular_angulo(self, p1, p2, p3):
        """Calcula ângulo entre três pontos (p2 é o vértice)"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        v1 = np.array([x1 - x2, y1 - y2], dtype=float)
        v2 = np.array([x3 - x2, y3 - y2], dtype=float)
        
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
        cos_ang = np.dot(v1, v2) / denom
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        
        ang = math.degrees(math.acos(cos_ang))
        return ang
    
    def identificar_gesto(self, angulo):
        """Identifica qual gesto corresponde ao ângulo"""
        for nome_gesto, config in self.gestos.items():
            if config["angulo_min"] <= angulo < config["angulo_max"]:
                return nome_gesto
        return None
    
    def processar_frame(self, frame):
        """Processa frame e detecta ângulos"""
        h, w, _ = frame.shape
        
        # Processar com MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        gesto_atual = None
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Desenhar esqueleto da mão com linhas roxas
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(200, 0, 200), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(200, 0, 200), thickness=2)
                )
                
                # Converter landmarks para pixel
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                
                if len(pts) >= 9:
                    # Ângulo entre POLEGAR (4) e INDICADOR (8) usando punho (0) como vértice
                    ang_polegar_indicador = self.calcular_angulo(pts[4], pts[0], pts[8])
                    
                    # Desenhar linha entre polegar e indicador
                    cv2.line(frame, pts[4], pts[8], (0, 0, 255), 3)
                    
                    # Identificar gesto
                    gesto = self.identificar_gesto(ang_polegar_indicador)
                    gesto_atual = gesto
                    
                    # Offset para texto
                    y_offset = idx * 80 + 30
                    
                    # Exibir informações
                    cv2.putText(frame, f"MAO {idx + 1} - Polegar vs Indicador", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Ângulo entre polegar e indicador
                    if gesto:
                        config = self.gestos[gesto]
                        cv2.putText(frame, f"Angulo: {ang_polegar_indicador:.1f}  - {gesto}", 
                                   (10, y_offset + 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, config["cor"], 3)
                        # Processa gesto sustentado
                        self.processador_gesto(gesto)
                    else:
                        cv2.putText(frame, f"Angulo: {ang_polegar_indicador:.1f}", 
                                   (10, y_offset + 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                        self.contador_gesto = 0
                        self.gesto_ativo = None
        
        # Mostrar legenda
        cv2.putText(frame, "Gestos: PINCA | AGUDO | RETO | OBTUSO | ABERTO",
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Exibir imagem ao lado baseado no gesto detectado
        frame_com_imagem = self.adicionar_imagem_lateral(frame, gesto_atual)
        
        return frame_com_imagem
    
    def adicionar_imagem_lateral(self, frame, gesto):
        """Adiciona a imagem correspondente ao gesto na lateral da tela"""
        h, w = frame.shape[:2]
        
        # Criar uma nova imagem com espaço para a imagem lateral (400x300)
        lateral_w = 420  # 400 imagem + 20 margem
        nova_largura = w + lateral_w
        frame_expandido = np.zeros((h, nova_largura, 3), dtype=np.uint8)
        frame_expandido[:, :w] = frame
        
        x_inicio = w + 10
        
        if gesto and gesto in self.gestos:
            nome_imagem = self.gestos[gesto].get("imagem")
            cor = self.gestos[gesto]["cor"]
            
            img = self.imagens_cache.get(nome_imagem)
            if img is not None:
                img_h, img_w = img.shape[:2]
                y_inicio = max(10, (h - img_h) // 2)
                y_fim = min(y_inicio + img_h, h - 10)
                
                # Ajuste caso a imagem seja maior que o espaço vertical
                recorte_h = y_fim - y_inicio
                if recorte_h > 0:
                    frame_expandido[y_inicio:y_inicio+recorte_h, x_inicio:x_inicio+img_w] = img[:recorte_h, :img_w]
                
                # Borda e título
                cv2.rectangle(frame_expandido, (x_inicio-5, y_inicio-5), 
                              (x_inicio+img_w+5, y_inicio+recorte_h+5), 
                              cor, 2)
                cv2.putText(frame_expandido, gesto, (x_inicio, max(30, y_inicio - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
            else:
                # Mensagem quando o arquivo não foi carregado
                cv2.putText(frame_expandido, f"Imagem '{nome_imagem}' nao carregada",
                            (x_inicio, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        else:
            # Mensagem se nenhum gesto for detectado
            cv2.putText(frame_expandido, "Nenhum gesto",
                        (x_inicio + 20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        
        return frame_expandido
    
    def processador_gesto(self, gesto):
        """Processa gesto quando é identificado de forma sustentada"""
        if gesto == self.gesto_ativo:
            self.contador_gesto += 1
        else:
            self.gesto_ativo = gesto
            self.contador_gesto = 1
        
        if self.contador_gesto == self.threshold_gesto:
            config = self.gestos[gesto]
            print("\n" + "="*50)
            print(f"GESTO SUSTENTADO: {config['descricao']}")
            print("="*50)
            self.contador_gesto = self.threshold_gesto + 1  # Evitar repetição
    
    def executar(self):
        """Loop principal da câmera"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERRO] Nao foi possivel abrir a camera.")
            return
        
        print("\n" + "="*60)
        print("DETECTOR DE ANGULOS COM ACOES")
        print("="*60)
        print("\nGestos disponíveis:")
        for nome, config in self.gestos.items():
            print(f"  {nome}: {config['descricao']} ({config['angulo_min']}-{config['angulo_max']})")
        print("\n  Acao: Mantenha o gesto por ~1 segundo para ativar a acao")
        print("\nPressione ESC para sair")
        print("="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERRO] Falha ao ler frame da camera.")
                break
            
            # Espelhar frame
            frame = cv2.flip(frame, 1)
            
            # Processar
            frame = self.processar_frame(frame)
            
            # Mostrar
            cv2.imshow("Detector de Angulos com Acoes + Preview", frame)
            
            # ESC para sair
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Você pode passar o caminho da pasta aqui, se quiser:
    # detector = DetectorAngulosComAcoes(base_path=r"C:\caminho\para\imagens")
    detector = DetectorAngulosComAcoes()
    detector.executar()
