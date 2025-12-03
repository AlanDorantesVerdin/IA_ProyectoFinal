import cv2
import numpy as np
import threading
import tkinter as tk 
import re 
from PIL import Image, ImageTk, ImageDraw # IMPORTANTE: ImageDraw para dibujar con transparencia
import customtkinter as ctk
import pygame
from ultralytics import YOLO
from paddleocr import PaddleOCR
from cvzone.Utils import cornerRect, putTextRect

# --- CONFIGURACIÓN DE SONIDO ---
pygame.mixer.init()
def play_alert_sound():
    try:
        pygame.mixer.Sound(buffer=np.sin(2 * np.pi * np.arange(22050) * 1000 / 44100).astype(np.float32)).play()
    except:
        pass 

class LicensePlateApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- CONFIGURACIÓN VENTANA ---
        self.title("Sistema Profesional de Detección de Placas")
        self.geometry("1280x720")
        try:
            self.state("zoomed") 
        except:
            self.attributes("-zoomed", True)

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        # Variables
        self.roi_coords = None 
        self.scale_factor = 1.0 
        self.target_plate = ""
        self.alert_triggered_plates = set()
        self.vehicle_data = {}
        self.running = False
        self.video_path = './videos/carros2.mp4'

        # Variables para ROI con transparencia
        self.roi_bg_pil = None # Imagen base limpia
        self.final_roi_coords = None # Coordenadas finales del dibujo

        # Cargar primera imagen para ROI
        self.cap_temp = cv2.VideoCapture(self.video_path)
        ret, self.first_frame = self.cap_temp.read()
        self.cap_temp.release()
        
        if not ret:
            print("Error: No se pudo leer el video.")
            return

        # Grid
        self.grid_columnconfigure(0, weight=4) 
        self.grid_columnconfigure(1, weight=1) 
        self.grid_rowconfigure(0, weight=1)

        # Vista Video
        self.view_area = ctk.CTkFrame(self, fg_color="#000000")
        self.view_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.view_area.pack_propagate(False)

        self.setup_roi_ui()

    # =========================================================
    #            MÓDULO 1: SELECCIÓN DE ROI (TRANSPARENTE)
    # =========================================================
    def setup_roi_ui(self):
        self.roi_panel = ctk.CTkFrame(self, fg_color="#2b2b2b")
        self.roi_panel.grid(row=0, column=1, sticky="nsew", padx=(0,10), pady=10)
        
        ctk.CTkLabel(self.roi_panel, text="CONFIGURACIÓN", font=("Roboto", 20, "bold")).pack(pady=(30, 10))
        ctk.CTkLabel(self.roi_panel, text="1. Dibuja el recuadro rojo\nsobre la carretera.", font=("Roboto", 14), text_color="gray").pack(pady=10)

        self.btn_confirm = ctk.CTkButton(self.roi_panel, text="CONFIRMAR ÁREA", fg_color="#0055aa", state="disabled", command=self.confirm_roi_and_start)
        self.btn_confirm.pack(pady=20)

        self.canvas = tk.Canvas(self.view_area, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", self.refresh_roi_image)

        self.start_x = None; self.start_y = None
        self.after(100, self.refresh_roi_image)

    def refresh_roi_image(self, event=None):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 1 and canvas_h > 1:
            frame_rgb = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2RGB)
            h_vid, w_vid, _ = frame_rgb.shape
            
            self.scale_factor = min(canvas_w / w_vid, canvas_h / h_vid)
            new_w = int(w_vid * self.scale_factor)
            new_h = int(h_vid * self.scale_factor)
            
            # --- CREAR BASE PIL PARA DIBUJAR ---
            img = Image.fromarray(frame_rgb)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            
            # Guardamos la versión RGBA limpia para usarla de base en cada movimiento
            self.roi_bg_pil = img.convert("RGBA") 
            self.tk_image_roi = ImageTk.PhotoImage(self.roi_bg_pil)
            
            self.canvas.delete("all")
            self.img_offset_x = (canvas_w - new_w) // 2
            self.img_offset_y = (canvas_h - new_h) // 2
            
            # Tag "bg" para identificarla y actualizarla rápido
            self.canvas.create_image(self.img_offset_x, self.img_offset_y, anchor="nw", image=self.tk_image_roi, tags="bg_img")

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def on_mouse_drag(self, event):
        if self.roi_bg_pil:
            # 1. Crear capa transparente vacía
            overlay = Image.new("RGBA", self.roi_bg_pil.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            
            # 2. Calcular coordenadas relativas a la imagen (quitando los bordes negros)
            x1 = self.start_x - self.img_offset_x
            y1 = self.start_y - self.img_offset_y
            x2 = event.x - self.img_offset_x
            y2 = event.y - self.img_offset_y
            
            # 3. Dibujar rectángulo con relleno semitransparente (R, G, B, Alpha)
            # Alpha 70 = Transparencia sutil, Alpha 255 = Sólido
            draw.rectangle((x1, y1, x2, y2), fill=(255, 0, 0, 70), outline=(255, 0, 0, 255), width=3)
            
            # 4. Fusionar capa roja sobre la imagen original
            combined = Image.alpha_composite(self.roi_bg_pil, overlay)
            
            # 5. Actualizar Canvas
            self.tk_image_roi = ImageTk.PhotoImage(combined)
            self.canvas.itemconfig("bg_img", image=self.tk_image_roi)
            
            # Guardamos coords temporales para cuando suelte el mouse
            self.final_roi_coords = (x1, y1, x2, y2)

    def on_mouse_up(self, event):
        if self.final_roi_coords:
            self.btn_confirm.configure(state="normal", fg_color="#1f6aa5")

    def confirm_roi_and_start(self):
        if self.final_roi_coords:
            x1, y1, x2, y2 = self.final_roi_coords
            
            # Ordenar coordenadas (por si dibujó de abajo hacia arriba)
            real_x1 = int(min(x1, x2) / self.scale_factor)
            real_y1 = int(min(y1, y2) / self.scale_factor)
            real_x2 = int(max(x1, x2) / self.scale_factor)
            real_y2 = int(max(y1, y2) / self.scale_factor)

            self.roi_coords = (max(0, real_x1), max(0, real_y1), abs(real_x2 - real_x1), abs(real_y2 - real_y1))
            
            self.roi_panel.destroy()
            self.canvas.destroy()
            self.setup_detection_ui()
            self.start_detection_thread()

    # =========================================================
    #            MÓDULO 2: INTERFAZ DE DETECCIÓN
    # =========================================================
    def setup_detection_ui(self):
        self.side_panel = ctk.CTkFrame(self, fg_color="#2b2b2b")
        self.side_panel.grid(row=0, column=1, sticky="nsew", padx=(0,10), pady=10)

        ctk.CTkLabel(self.side_panel, text="PANEL DE CONTROL", font=("Roboto", 20, "bold")).pack(pady=15)

        s_frame = ctk.CTkFrame(self.side_panel, fg_color="#333333")
        s_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(s_frame, text="Buscar Placa (Alerta)", font=("Roboto", 14)).pack(anchor="w", padx=10, pady=(5,0))
        self.entry_search = ctk.CTkEntry(s_frame, placeholder_text="Ej: ABC1234")
        self.entry_search.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(s_frame, text="Fijar Objetivo", command=self.set_target_plate, fg_color="#444444").pack(fill="x", padx=10, pady=(0,10))
        
        self.lbl_status = ctk.CTkLabel(self.side_panel, text="ESTADO: Monitoreando...", text_color="gray")
        self.lbl_status.pack(pady=5)

        ctk.CTkLabel(self.side_panel, text="Vehículos Detectados", font=("Roboto", 16, "bold")).pack(pady=(15,5))
        self.scroll_frame = ctk.CTkScrollableFrame(self.side_panel, fg_color="#1e1e1e")
        self.scroll_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.lbl_video = ctk.CTkLabel(self.view_area, text="", font=("Roboto", 20))
        self.lbl_video.pack(expand=True, fill="both")

    def set_target_plate(self):
        txt = self.entry_search.get().strip().upper()
        txt = re.sub(r'[^A-Z0-9]', '', txt)
        
        if txt:
            self.target_plate = txt
            self.lbl_status.configure(text=f"BUSCANDO: {self.target_plate}", text_color="#00FFFF")
        else:
            self.target_plate = ""
            self.lbl_status.configure(text="ESTADO: Monitoreando...", text_color="gray")

    def update_vehicle_list(self, track_id, plate_text, is_target=False):
        if track_id in self.vehicle_data:
            lbl_id, lbl_plate, frame_ref = self.vehicle_data[track_id]
            lbl_plate.configure(text=plate_text)
            
            if is_target:
                frame_ref.configure(fg_color="#00E5FF", border_width=2, border_color="white") 
                lbl_id.configure(text_color="black")
                lbl_plate.configure(text_color="black", font=("Roboto", 16, "bold"))
        else:
            item_frame = ctk.CTkFrame(self.scroll_frame, fg_color="#333333")
            item_frame.pack(fill="x", pady=2)
            
            lbl_id = ctk.CTkLabel(item_frame, text=f"ID: {track_id}", width=50, anchor="w", font=("Roboto", 12, "bold"))
            lbl_id.pack(side="left", padx=5)
            
            lbl_plate = ctk.CTkLabel(item_frame, text=plate_text, font=("Roboto", 14))
            lbl_plate.pack(side="left", padx=10)
            
            self.vehicle_data[track_id] = (lbl_id, lbl_plate, item_frame)
            if is_target: self.update_vehicle_list(track_id, plate_text, True)

    # =========================================================
    #            MÓDULO 3: LÓGICA DE IA (OPTIMIZADA)
    # =========================================================
    def start_detection_thread(self):
        self.running = True
        thread = threading.Thread(target=self.run_detection_loop)
        thread.daemon = True
        thread.start()

    def run_detection_loop(self):
        print("Cargando modelo YOLO Nano (Ultrarrápido)...")
        # Modelos
        coco_model = YOLO('yolo11n.pt') 
        license_plate_detector = YOLO('plate_detector.pt')
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        cap = cv2.VideoCapture(self.video_path)
        vehicles = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        bytetrack_tracker = "bytetrack.yaml"
        vehicle_plates_memory = {} 
        
        # Persistencia visual
        annotations = {'vehicles': [], 'plates': []} 

        x_roi, y_roi, w_roi, h_roi = self.roi_coords
        frame_count = 0
        process_every_n = 3 

        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            
            # --- FASE 1: PROCESAMIENTO IA (1 de cada 3 frames) ---
            if frame_count % process_every_n == 0:
                annotations['vehicles'] = []
                annotations['plates'] = []

                roi_frame = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi].copy()
                
                # Tracking
                results = coco_model.track(roi_frame, persist=True, tracker=bytetrack_tracker, classes=list(vehicles.keys()), iou=0.5, agnostic_nms=True, verbose=False)
                
                vehicle_tracks = {}
                if results[0].boxes.id is not None:
                    for box, track_id, class_id in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        track_id = int(track_id)
                        x1 += x_roi; x2 += x_roi; y1 += y_roi; y2 += y_roi
                        vehicle_tracks[track_id] = (x1, y1, x2, y2)
                        
                        annotations['vehicles'].append({
                            'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                            'id': track_id,
                            'pos_text': (int(x1), int(y1) - 10)
                        })

                # Placas
                license_plates = license_plate_detector(roi_frame, verbose=False)[0]
                
                for license_plate in license_plates.boxes.data.tolist():
                    xp1, yp1, xp2, yp2, score, class_id = license_plate
                    xp1 += x_roi; xp2 += x_roi; yp1 += y_roi; yp2 += y_roi
                    h_img, w_img, _ = frame.shape
                    xp1, yp1, xp2, yp2 = int(max(0, xp1)), int(max(0, yp1)), int(min(w_img, xp2)), int(min(h_img, yp2))

                    for track_id, (xc1, yc1, xc2, yc2) in vehicle_tracks.items():
                        if xp1 > xc1 and yp1 > yc1 and xp2 < xc2 and yp2 < yc2:
                            if xp2 > xp1 and yp2 > yp1:
                                plate_crop = frame[yp1:yp2, xp1:xp2]
                                if plate_crop.size > 0:
                                    try:
                                        plate_crop_zoom = cv2.resize(plate_crop, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
                                        ocr_result = ocr.ocr(plate_crop_zoom, cls=True)
                                    except:
                                        ocr_result = None
                                    
                                    detected_text = ""
                                    conf = 0.0
                                    if ocr_result:
                                        for line in ocr_result:
                                            if line:
                                                for word_info in line:
                                                    if len(word_info) > 1:
                                                        text, c = word_info[1]
                                                        if c > 0.8:
                                                            detected_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                                                            conf = c

                                    if detected_text:
                                        should_update = False
                                        if track_id not in vehicle_plates_memory:
                                            should_update = True
                                        else:
                                            current_text, current_conf = vehicle_plates_memory[track_id]
                                            if len(detected_text) > len(current_text):
                                                should_update = True
                                            elif len(detected_text) == len(current_text) and conf > current_conf:
                                                should_update = True

                                        if should_update:
                                            vehicle_plates_memory[track_id] = (detected_text, conf)
                                            
                                            is_target_found = False
                                            if self.target_plate and self.target_plate in detected_text:
                                                is_target_found = True
                                                if detected_text not in self.alert_triggered_plates:
                                                    play_alert_sound()
                                                    self.alert_triggered_plates.add(detected_text)
                                            
                                            self.after(0, lambda t=track_id, p=detected_text, obj=is_target_found: self.update_vehicle_list(t, p, obj))

                            if track_id in vehicle_plates_memory:
                                curr_text = vehicle_plates_memory[track_id][0]
                                color_bg = (0, 0, 255) if (self.target_plate and self.target_plate in curr_text) else (0, 0, 0)
                                annotations['plates'].append({
                                    'text': curr_text,
                                    'pos': (int(xp1), int(yp1) - 10),
                                    'color': color_bg
                                })

            # --- FASE 2: DIBUJADO PERSISTENTE ---
            for v in annotations['vehicles']:
                cornerRect(frame, v['bbox'], l=10, rt=2, colorR=(255, 0, 0))
                putTextRect(frame, f'Car {v["id"]}', v['pos_text'], scale=1.2, thickness=2, colorR=(255, 0, 0), colorB=(255, 255, 255))
            
            for p in annotations['plates']:
                putTextRect(frame, p['text'], p['pos'], scale=1.5, thickness=2, colorR=p['color'], colorB=(255, 255, 255), border=2)

            self.update_video_ui(frame)

        cap.release()

    def update_video_ui(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        canvas_w = self.view_area.winfo_width()
        canvas_h = self.view_area.winfo_height()
        
        if canvas_w > 10 and canvas_h > 10:
            scale = min(canvas_w / w, canvas_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            img = Image.fromarray(frame_rgb)
            img = img.resize((new_w, new_h), Image.NEAREST)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
            
            self.after(0, lambda: self.lbl_video.configure(image=imgtk, text=""))

if __name__ == "__main__":
    app = LicensePlateApp()
    app.mainloop()