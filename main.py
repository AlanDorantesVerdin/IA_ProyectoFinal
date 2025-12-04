import cv2
import numpy as np
import threading
import tkinter as tk 
import re 
from PIL import Image, ImageTk, ImageDraw 
import customtkinter as ctk
import pygame
from ultralytics import YOLO
from paddleocr import PaddleOCR
from cvzone.Utils import cornerRect, putTextRect

# -------------------------------------------------------------------------
# MÓDULO DE AUDIO
# -------------------------------------------------------------------------
# Inicializamos el mixer de Pygame para efectos de sonido.
pygame.mixer.init()

def play_alert_sound():
    """
    Genera y reproduce un sonido de alerta sintético (onda senoidal).
    Esto evita depender de archivos de audio externos (.mp3/.wav).
    """
    try:
        # Creamos una onda senoidal pura matemáticamente usando NumPy
        # Frecuencia: 1000Hz | Tasa de muestreo: 44100Hz
        pygame.mixer.Sound(buffer=np.sin(2 * np.pi * np.arange(22050) * 1000 / 44100).astype(np.float32)).play()
    except:
        pass 

# -------------------------------------------------------------------------
# CLASE PRINCIPAL DE LA APLICACIÓN
# -------------------------------------------------------------------------
class LicensePlateApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Paso 1: Configuración inicial de la Ventana Gráfica ---
        self.title("Sistema Profesional de Detección de Placas")
        self.geometry("1280x720")
        
        # Intentamos maximizar la ventana dependiendo del Sistema Operativo
        try:
            self.state("zoomed") # Funciona en Windows
        except:
            self.attributes("-zoomed", True) # Alternativa para Linux/Mac

        # Tema visual oscuro para apariencia moderna
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        # --- Paso 2: Inicialización de Variables de Estado ---
        self.roi_coords = None                      # Guardará las coordenadas (x, y, w, h) del área de interés
        self.scale_factor = 1.0                     # Factor de escala para convertir coordenadas de Pantalla <-> Video
        self.target_plate = ""                      # Placa que el usuario quiere buscar
        self.alert_triggered_plates = set()         # Registro de placas ya alertadas para no repetir sonido
        self.vehicle_data = {}                      # 'Memoria' de la interfaz gráfica {id_carro: widgets}
        self.running = False                        # Bandera de control para el hilo de ejecución
        self.video_path = './videos/carros2.mp4'

        # Variables auxiliares para el dibujo del ROI con transparencia
        self.roi_bg_pil = None 
        self.final_roi_coords = None 

        # --- Paso 3: Carga del primer fotograma para configuración ---
        # Necesitamos leer un cuadro del video ANTES de empezar para que el usuario seleccione el ROI
        self.cap_temp = cv2.VideoCapture(self.video_path)
        ret, self.first_frame = self.cap_temp.read()
        self.cap_temp.release()
        
        if not ret:
            print("Error Crítico: No se pudo leer el archivo de video.")
            return

        # --- Paso 4: Diseño del Layout (Grid) ---
        # Dividimos la pantalla en 2 columnas: Video (4 partes) y Panel (1 parte)
        self.grid_columnconfigure(0, weight=4) 
        self.grid_columnconfigure(1, weight=1) 
        self.grid_rowconfigure(0, weight=1)

        # Contenedor principal para el Video
        self.view_area = ctk.CTkFrame(self, fg_color="#000000")
        self.view_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.view_area.pack_propagate(False) # Evita que el contenedor cambie de tamaño automáticamente

        # Iniciamos en el modo de selección de área
        self.setup_roi_ui()

    # =========================================================
    #            MÓDULO 1: SELECCIÓN DE ROI (INTERFAZ)
    # =========================================================
    def setup_roi_ui(self):
        """ Configura el panel lateral con instrucciones para seleccionar el área. """
        self.roi_panel = ctk.CTkFrame(self, fg_color="#2b2b2b")
        self.roi_panel.grid(row=0, column=1, sticky="nsew", padx=(0,10), pady=10)
        
        ctk.CTkLabel(self.roi_panel, text="CONFIGURACIÓN", font=("Roboto", 20, "bold")).pack(pady=(30, 10))
        ctk.CTkLabel(self.roi_panel, text="1. Dibuja el recuadro rojo\nsobre la carretera.", font=("Roboto", 14), text_color="gray").pack(pady=10)

        # Botón desactivado inicialmente hasta que el usuario dibuje
        self.btn_confirm = ctk.CTkButton(self.roi_panel, text="CONFIRMAR ÁREA", fg_color="#0055aa", state="disabled", command=self.confirm_roi_and_start)
        self.btn_confirm.pack(pady=20)

        # Usamos un Canvas de Tkinter tradicional porque permite dibujar formas libremente
        self.canvas = tk.Canvas(self.view_area, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Vinculamos los eventos del mouse para la lógica de "Arrastrar y Soltar"
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", self.refresh_roi_image) # Se ejecuta si cambia el tamaño de ventana

        self.start_x = None; self.start_y = None
        # Pequeño retraso para asegurar que la UI cargó antes de pintar la imagen
        self.after(100, self.refresh_roi_image)

    def refresh_roi_image(self, event=None):
        """ Redimensiona la imagen estática del video para que quepa en el Canvas (Ajuste 'Fit'). """
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 1 and canvas_h > 1:
            # Conversión de color BGR (OpenCV) a RGB (Pantalla)
            frame_rgb = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2RGB)
            h_vid, w_vid, _ = frame_rgb.shape
            
            # Algoritmo de escalado: Calculamos qué dimensión limita el tamaño (ancho o alto)
            self.scale_factor = min(canvas_w / w_vid, canvas_h / h_vid)
            new_w = int(w_vid * self.scale_factor)
            new_h = int(h_vid * self.scale_factor)
            
            # Procesamiento de imagen con PIL
            img = Image.fromarray(frame_rgb)
            img = img.resize((new_w, new_h), Image.LANCZOS) # Calidad alta para imagen estática
            
            # Guardamos la imagen base en formato RGBA para poder sobreponer capas transparentes
            self.roi_bg_pil = img.convert("RGBA") 
            self.tk_image_roi = ImageTk.PhotoImage(self.roi_bg_pil)
            
            # Centramos la imagen en el canvas
            self.canvas.delete("all")
            self.img_offset_x = (canvas_w - new_w) // 2
            self.img_offset_y = (canvas_h - new_h) // 2
            
            self.canvas.create_image(self.img_offset_x, self.img_offset_y, anchor="nw", image=self.tk_image_roi, tags="bg_img")

    def on_mouse_down(self, event):
        """ Guarda el punto inicial del clic. """
        self.start_x = event.x
        self.start_y = event.y

    def on_mouse_drag(self, event):
        """ Dibuja el rectángulo con transparencia mientras se arrastra el mouse. """
        if self.roi_bg_pil:
            # 1. Creamos una capa vacía transparente
            overlay = Image.new("RGBA", self.roi_bg_pil.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            
            # 2. Ajustamos coordenadas (Mouse -> Coordenadas relativas a la imagen)
            x1 = self.start_x - self.img_offset_x
            y1 = self.start_y - self.img_offset_y
            x2 = event.x - self.img_offset_x
            y2 = event.y - self.img_offset_y
            
            # 3. Dibujamos el rectángulo
            # fill=(R, G, B, Alpha) -> Alpha 70 da el efecto transparente
            draw.rectangle((x1, y1, x2, y2), fill=(255, 0, 0, 70), outline=(255, 0, 0, 255), width=3)
            
            # 4. Fusionamos la capa roja sobre la imagen original
            combined = Image.alpha_composite(self.roi_bg_pil, overlay)
            
            # 5. Actualizamos lo que ve el usuario
            self.tk_image_roi = ImageTk.PhotoImage(combined)
            self.canvas.itemconfig("bg_img", image=self.tk_image_roi)
            
            self.final_roi_coords = (x1, y1, x2, y2)

    def on_mouse_up(self, event):
        """ Habilita el botón de confirmar una vez que se suelta el mouse. """
        if self.final_roi_coords:
            self.btn_confirm.configure(state="normal", fg_color="#1f6aa5")

    def confirm_roi_and_start(self):
        """ Traduce las coordenadas dibujadas a coordenadas reales del video e inicia la IA. """
        if self.final_roi_coords:
            x1, y1, x2, y2 = self.final_roi_coords
            
            # Matemáticas: Deshacemos el escalado y el offset para obtener posición real en el video original
            real_x1 = int(min(x1, x2) / self.scale_factor)
            real_y1 = int(min(y1, y2) / self.scale_factor)
            real_x2 = int(max(x1, x2) / self.scale_factor)
            real_y2 = int(max(y1, y2) / self.scale_factor)

            # Guardamos ROI como (x, y, ancho, alto)
            self.roi_coords = (max(0, real_x1), max(0, real_y1), abs(real_x2 - real_x1), abs(real_y2 - real_y1))
            
            # Limpieza y cambio de interfaz
            self.roi_panel.destroy()
            self.canvas.destroy()
            self.setup_detection_ui()
            self.start_detection_thread()

    # =========================================================
    #            MÓDULO 2: INTERFAZ DE DETECCIÓN (UI)
    # =========================================================
    def setup_detection_ui(self):
        """ Construye el panel de control lateral derecho. """
        self.side_panel = ctk.CTkFrame(self, fg_color="#2b2b2b")
        self.side_panel.grid(row=0, column=1, sticky="nsew", padx=(0,10), pady=10)

        ctk.CTkLabel(self.side_panel, text="PANEL DE CONTROL", font=("Roboto", 20, "bold")).pack(pady=15)

        # Sección de Búsqueda
        s_frame = ctk.CTkFrame(self.side_panel, fg_color="#333333")
        s_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(s_frame, text="Buscar Placa (Alerta)", font=("Roboto", 14)).pack(anchor="w", padx=10, pady=(5,0))
        
        self.entry_search = ctk.CTkEntry(s_frame, placeholder_text="Ej: ABC1234")
        self.entry_search.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(s_frame, text="Fijar Objetivo", command=self.set_target_plate, fg_color="#444444").pack(fill="x", padx=10, pady=(0,10))
        
        self.lbl_status = ctk.CTkLabel(self.side_panel, text="ESTADO: Monitoreando...", text_color="gray")
        self.lbl_status.pack(pady=5)

        # Lista de Vehículos (Scrollable)
        ctk.CTkLabel(self.side_panel, text="Vehículos Detectados", font=("Roboto", 16, "bold")).pack(pady=(15,5))
        self.scroll_frame = ctk.CTkScrollableFrame(self.side_panel, fg_color="#1e1e1e")
        self.scroll_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Label donde se proyectará el video
        self.lbl_video = ctk.CTkLabel(self.view_area, text="", font=("Roboto", 20))
        self.lbl_video.pack(expand=True, fill="both")

    def set_target_plate(self):
        """ Establece la placa objetivo limpiando caracteres no deseados. """
        txt = self.entry_search.get().strip().upper()
        # Regex: Eliminar todo lo que NO sea A-Z o 0-9
        txt = re.sub(r'[^A-Z0-9]', '', txt)
        
        if txt:
            self.target_plate = txt
            self.lbl_status.configure(text=f"BUSCANDO: {self.target_plate}", text_color="#00FFFF")
        else:
            self.target_plate = ""
            self.lbl_status.configure(text="ESTADO: Monitoreando...", text_color="gray")

    def update_vehicle_list(self, track_id, plate_text, is_target=False):
        """ Actualiza o crea la tarjeta del vehículo en la lista lateral. """
        if track_id in self.vehicle_data:
            # Si ya existe, actualizamos solo el texto
            lbl_id, lbl_plate, frame_ref = self.vehicle_data[track_id]
            lbl_plate.configure(text=plate_text)
            
            # Si es el objetivo, cambiamos el estilo a Turquesa/Negro para resaltar
            if is_target:
                frame_ref.configure(fg_color="#00E5FF", border_width=2, border_color="white") 
                lbl_id.configure(text_color="black")
                lbl_plate.configure(text_color="black", font=("Roboto", 16, "bold"))
        else:
            # Si es nuevo, creamos los widgets
            item_frame = ctk.CTkFrame(self.scroll_frame, fg_color="#333333")
            item_frame.pack(fill="x", pady=2)
            
            lbl_id = ctk.CTkLabel(item_frame, text=f"ID: {track_id}", width=50, anchor="w", font=("Roboto", 12, "bold"))
            lbl_id.pack(side="left", padx=5)
            
            lbl_plate = ctk.CTkLabel(item_frame, text=plate_text, font=("Roboto", 14))
            lbl_plate.pack(side="left", padx=10)
            
            self.vehicle_data[track_id] = (lbl_id, lbl_plate, item_frame)
            # Llamada recursiva para aplicar estilo si nació siendo objetivo
            if is_target: self.update_vehicle_list(track_id, plate_text, True)

    # =========================================================
    #            MÓDULO 3: LÓGICA DE IA Y PROCESAMIENTO
    # =========================================================
    def start_detection_thread(self):
        """ Inicia el bucle de detección en un hilo separado para no congelar la GUI. """
        self.running = True
        thread = threading.Thread(target=self.run_detection_loop)
        thread.daemon = True
        thread.start()

    def run_detection_loop(self):
        """ Bucle principal de procesamiento de video e Inteligencia Artificial. """
        print("Iniciando modelos de IA...")
        
        # 1. Carga de Modelos
        coco_model = YOLO('yolo11n.pt')            # Modelo Nano para tracking de vehículos (Rápido)
        license_plate_detector = YOLO('plate_detector.pt') # Modelo especializado en detectar placas
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False) # Lector de texto

        cap = cv2.VideoCapture(self.video_path)
        
        # Clases COCO relevantes: 2=carro, 3=moto, 5=bus, 7=camión
        vehicles = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        bytetrack_tracker = "bytetrack.yaml"
        
        # 'Memoria' lógica para almacenar la mejor lectura de placa de cada ID
        vehicle_plates_memory = {} 
        
        # Diccionario para Persistencia Visual (evita parpadeos al saltar frames)
        annotations = {'vehicles': [], 'plates': []} 

        x_roi, y_roi, w_roi, h_roi = self.roi_coords
        frame_count = 0
        process_every_n = 3 # Optimización: Procesar IA solo 1 de cada 3 frames

        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            
            # --- FASE A: PROCESAMIENTO PESADO (IA) ---
            # Solo ejecutamos esto si estamos en el frame correspondiente (1 de cada 3)
            if frame_count % process_every_n == 0:
                # Limpiamos anotaciones anteriores
                annotations['vehicles'] = []
                annotations['plates'] = []

                # Recortamos el área de interés (ROI) para procesar menos píxeles
                roi_frame = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi].copy()
                
                # Paso 1: Tracking de Vehículos (YOLO Nano)
                results = coco_model.track(roi_frame, persist=True, tracker=bytetrack_tracker, classes=list(vehicles.keys()), iou=0.5, agnostic_nms=True, verbose=False)
                
                vehicle_tracks = {}
                # Procesamos resultados del tracking
                if results[0].boxes.id is not None:
                    for box, track_id, class_id in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        track_id = int(track_id)
                        # Ajustamos coordenadas relativas (ROI) a absolutas (Video completo)
                        x1 += x_roi; x2 += x_roi; y1 += y_roi; y2 += y_roi
                        vehicle_tracks[track_id] = (x1, y1, x2, y2)
                        
                        # Guardamos datos visuales para dibujar después
                        annotations['vehicles'].append({
                            'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                            'id': track_id,
                            'pos_text': (int(x1), int(y1) - 10)
                        })

                # Paso 2: Detección de Placas (YOLO Custom)
                license_plates = license_plate_detector(roi_frame, verbose=False)[0]
                
                for license_plate in license_plates.boxes.data.tolist():
                    xp1, yp1, xp2, yp2, score, class_id = license_plate
                    # Coordenadas absolutas
                    xp1 += x_roi; xp2 += x_roi; yp1 += y_roi; yp2 += y_roi
                    
                    # Validación de límites de imagen
                    h_img, w_img, _ = frame.shape
                    xp1, yp1, xp2, yp2 = int(max(0, xp1)), int(max(0, yp1)), int(min(w_img, xp2)), int(min(h_img, yp2))

                    # Paso 3: Asociación Placa -> Vehículo
                    for track_id, (xc1, yc1, xc2, yc2) in vehicle_tracks.items():
                        # ¿Está la placa DENTRO del vehículo?
                        if xp1 > xc1 and yp1 > yc1 and xp2 < xc2 and yp2 < yc2:
                            if xp2 > xp1 and yp2 > yp1:
                                plate_crop = frame[yp1:yp2, xp1:xp2]
                                if plate_crop.size > 0:
                                    try:
                                        # Zoom digital para ayudar al OCR
                                        plate_crop_zoom = cv2.resize(plate_crop, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
                                        # Paso 4: Lectura de Texto (OCR)
                                        ocr_result = ocr.ocr(plate_crop_zoom, cls=True)
                                    except:
                                        ocr_result = None
                                    
                                    detected_text = ""
                                    conf = 0.0
                                    
                                    # Procesar resultado crudo del OCR
                                    if ocr_result:
                                        for line in ocr_result:
                                            if line:
                                                for word_info in line:
                                                    if len(word_info) > 1:
                                                        text, c = word_info[1]
                                                        if c > 0.8: # Umbral de confianza
                                                            # LIMPIEZA: Solo letras y números
                                                            detected_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                                                            conf = c

                                    # Paso 5: Heurística de Actualización
                                    if detected_text:
                                        should_update = False
                                        # Si es un carro nuevo -> Guardar
                                        if track_id not in vehicle_plates_memory:
                                            should_update = True
                                        else:
                                            current_text, current_conf = vehicle_plates_memory[track_id]
                                            # Prioridad a lecturas más largas (evita perder caracteres)
                                            if len(detected_text) > len(current_text):
                                                should_update = True
                                            # Corrección si misma longitud pero mejor confianza
                                            elif len(detected_text) == len(current_text) and conf > current_conf:
                                                should_update = True

                                        if should_update:
                                            vehicle_plates_memory[track_id] = (detected_text, conf)
                                            
                                            # Paso 6: Verificación de Alerta
                                            is_target_found = False
                                            if self.target_plate and self.target_plate in detected_text:
                                                is_target_found = True
                                                if detected_text not in self.alert_triggered_plates:
                                                    play_alert_sound()
                                                    self.alert_triggered_plates.add(detected_text)
                                            
                                            # Actualizar GUI en hilo principal
                                            self.after(0, lambda t=track_id, p=detected_text, obj=is_target_found: self.update_vehicle_list(t, p, obj))

                            # Preparar datos visuales para dibujar la placa
                            if track_id in vehicle_plates_memory:
                                curr_text = vehicle_plates_memory[track_id][0]
                                # Fondo rojo si es objetivo, negro si es normal
                                color_bg = (0, 0, 255) if (self.target_plate and self.target_plate in curr_text) else (0, 0, 0)
                                annotations['plates'].append({
                                    'text': curr_text,
                                    'pos': (int(xp1), int(yp1) - 10),
                                    'color': color_bg
                                })

            # --- FASE B: DIBUJADO PERSISTENTE (TODOS LOS FRAMES) ---
            # Dibujamos lo que tenemos en memoria, incluso si en este frame la IA descansó.
            for v in annotations['vehicles']:
                cornerRect(frame, v['bbox'], l=10, rt=2, colorR=(255, 0, 0))
                putTextRect(frame, f'Car {v["id"]}', v['pos_text'], scale=1.2, thickness=2, colorR=(255, 0, 0), colorB=(255, 255, 255))
            
            for p in annotations['plates']:
                putTextRect(frame, p['text'], p['pos'], scale=1.5, thickness=2, colorR=p['color'], colorB=(255, 255, 255), border=2)

            self.update_video_ui(frame)

        cap.release()

    def update_video_ui(self, frame):
        """ Actualiza la etiqueta de video en la interfaz gráfica. """
        # Conversión de color necesaria para PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        
        # Dimensiones actuales del contenedor (puede haber cambiado si se redimensionó la ventana)
        canvas_w = self.view_area.winfo_width()
        canvas_h = self.view_area.winfo_height()
        
        if canvas_w > 10 and canvas_h > 10:
            # Escalado proporcional 'Fit' (que quepa todo)
            scale = min(canvas_w / w, canvas_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            img = Image.fromarray(frame_rgb)
            # Usamos NEAREST para velocidad en tiempo real
            img = img.resize((new_w, new_h), Image.NEAREST)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
            
            # Programar actualización en el hilo principal de Tkinter
            self.after(0, lambda: self.lbl_video.configure(image=imgtk, text=""))

if __name__ == "__main__":
    app = LicensePlateApp()
    app.mainloop()