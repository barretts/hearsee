import pygame
import numpy as np
import sys
import cv2
import mediapipe as mp
import math
import colorsys # Needed for RGB to HSL conversion
from tkinter import Tk, filedialog # Needed for the file open dialog

# ==============================================================================
# 1. INITIALIZATION AND CORE SETTINGS
# ==============================================================================

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# --- Visual & Grid Settings ---
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 640

# --- Face Tracking & Interaction Settings ---
SMOOTHING_FACTOR = 0.3
MOVEMENT_SCALE = 5.0
BLENDING_ENABLED = True
BLENDING_INTENSITY = 1.0
BLENDING_RADIUS = 2.0 

# --- Global variables ---
g_smooth_cursor_x, g_smooth_cursor_y = 0.5, 0.5
g_center_offset_x, g_center_offset_y = 0.0, 0.0
g_global_mix_enabled = False

# ==============================================================================
# 2. MODULAR SOUND MODEL ARCHITECTURE (Phase 1.1)
# ==============================================================================

class SoundModel:
    """
    A modular class to handle the logic of converting visual data into sound.
    This default implementation is based on the "Naturalistic" model from the research.
    """
    def __init__(self, sample_rate=44100, master_volume=0.05):
        self.sample_rate = sample_rate
        self.duration = 0.2
        self.master_volume = master_volume
        self.min_frequency = 120
        self.max_frequency = 1200
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)

    def color_to_properties(self, r, g, b):
        """Converts an RGB color into sound properties (freq, amp, timbre)."""
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
        h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
        
        # Custom mapping for a more intuitive feel (calmer blues)
        if 0.5 < h < 0.75:
            blue_hue_normalized = (h - 0.5) / 0.25
            frequency = self.min_frequency + (blue_hue_normalized * (self.min_frequency * 2))
            timbre_complexity = s * 0.5 
        else:
            frequency = self.min_frequency + (h * (self.max_frequency - self.min_frequency))
            timbre_complexity = s
            
        amplitude = l
        return (frequency, amplitude, timbre_complexity)

    def generate_mono_wave(self, freq, amp, timbre):
        """Generates a single mono audio wave from sound properties."""
        pure_wave = np.sin(freq * self.t * 2 * np.pi)
        complex_wave = (np.sin(freq * self.t * 2 * np.pi) + 
                        0.5 * np.sin(freq * 2 * self.t * 2 * np.pi) + 
                        0.3 * np.sin(freq * 3 * self.t * 2 * np.pi))
        mono_wave = (1 - timbre) * pure_wave + timbre * complex_wave

        # "Glare" effect for overwhelming sounds
        if amp > 0.9 and timbre > 0.9:
            glare_freq = freq * 1.05 
            glare_wave = np.sin(glare_freq * self.t * 2 * np.pi)
            mono_wave += glare_wave * 0.4
        
        return mono_wave * amp # Return wave with amplitude applied

    def create_stereo_wave(self, landscape_img, cursor_x, cursor_y, pan_x):
        """Creates the final stereo sound wave based on image data and cursor position."""
        center_color_rgb = get_pixel_color_from_image(landscape_img, cursor_x, cursor_y)
        if center_color_rgb is None: center_color_rgb = (0,0,0)
        
        center_props = self.color_to_properties(*center_color_rgb)
        
        if BLENDING_ENABLED and BLENDING_INTENSITY > 0 and BLENDING_RADIUS > 0:
            weighted_props = [(*center_props, 1.0)]
            pixel_size_norm = 1.0 / landscape_img.shape[0]
            max_dist_pixels = int(BLENDING_RADIUS * 4)
            for i in range(1, max_dist_pixels + 1):
                for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    dx, dy = int(i * np.cos(angle)), int(i * np.sin(angle))
                    sample_x, sample_y = cursor_x + (dx * pixel_size_norm), cursor_y + (dy * pixel_size_norm)
                    color = get_pixel_color_from_image(landscape_img, sample_x, sample_y)
                    if color is not None:
                        dist_sq = dx**2 + dy**2
                        if dist_sq == 0: continue
                        weight = 1.0 / dist_sq
                        props = self.color_to_properties(*color)
                        weighted_props.append((*props, weight))
            
            total_weight = sum(p[3] for p in weighted_props)
            if total_weight > 0:
                w_avg_freq = sum(p[0] * p[3] for p in weighted_props) / total_weight
                w_avg_amp = sum(p[1] * p[3] for p in weighted_props) / total_weight
                w_avg_timbre = sum(p[2] * p[3] for p in weighted_props) / total_weight
                
                focused_freq = (center_props[0] + w_avg_freq * BLENDING_INTENSITY) / (1 + BLENDING_INTENSITY)
                focused_amp = (center_props[1] + w_avg_amp * BLENDING_INTENSITY) / (1 + BLENDING_INTENSITY)
                focused_timbre = (center_props[2] + w_avg_timbre * BLENDING_INTENSITY) / (1 + BLENDING_INTENSITY)
        else:
            focused_freq, focused_amp, focused_timbre = center_props

        focused_wave = self.generate_mono_wave(focused_freq, focused_amp, focused_timbre)

        # Global mix logic
        if g_global_mix_enabled:
            # For simplicity in this refactor, we can approximate the global sound
            # A full implementation would average properties as before
            global_color = cv2.mean(landscape_img)[:3]
            global_props = self.color_to_properties(*global_color)
            global_wave = self.generate_mono_wave(*global_props)
            mixed_mono_wave = (focused_wave * 0.7) + (global_wave * 0.3)
        else:
            mixed_mono_wave = focused_wave

        if np.max(np.abs(mixed_mono_wave)) == 0:
             return np.zeros((len(self.t), 2), dtype=np.int16)

        normalized_wave = mixed_mono_wave / np.max(np.abs(mixed_mono_wave))
        final_wave = normalized_wave * self.master_volume * 32767

        left_vol, right_vol = (1.0 + pan_x) / 2.0, (1.0 - pan_x) / 2.0
        stereo_wave = np.zeros((len(self.t), 2), dtype=np.int16)
        stereo_wave[:, 0] = (final_wave * left_vol).astype(np.int16)
        stereo_wave[:, 1] = (final_wave * right_vol).astype(np.int16)
        return stereo_wave

# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================

def generate_child_landscape():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[0:154, :] = (135, 206, 235) # Sky
    cv2.circle(img, (200, 60), 25, (255, 255, 0), -1) # Sun
    img[154:230, :] = (34, 139, 34) # Grass
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def load_image_from_file():
    root = Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename()
    if not filepath: return None
    try:
        return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    except: return None

def get_pixel_color_from_image(img, x, y):
    if 0 <= x <= 1 and 0 <= y <= 1:
        h, w, _ = img.shape
        return img[int(y * (h - 1)), int(x * (w - 1))]
    return None

def smooth_value(current, target, factor):
    return current + (target - current) * factor

def draw_info_panel(screen, font, data):
    y_offset = 10
    for key, value in data.items():
        text_surface = font.render(f"{key}: {value}", True, (255, 255, 255))
        screen.blit(text_surface, (10, y_offset))
        y_offset += 25

# ==============================================================================
# 4. MAIN APPLICATION LOOP
# ==============================================================================

def main():
    global g_smooth_cursor_x, g_smooth_cursor_y, g_center_offset_x, g_center_offset_y
    global BLENDING_INTENSITY, BLENDING_RADIUS, BLENDING_ENABLED, g_global_mix_enabled
    
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)
        face_tracking_active = cap.isOpened()
    except Exception as e:
        face_mesh, cap, face_tracking_active = None, None, False

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("HearSee - Phase 1.1")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # --- Initialize the sound model ---
    sound_model = SoundModel()
    
    current_image = cv2.cvtColor(generate_child_landscape(), cv2.COLOR_BGR2RGB)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    new_image = load_image_from_file()
                    if new_image is not None: current_image = new_image
                elif event.key == pygame.K_g: g_global_mix_enabled = not g_global_mix_enabled
                elif event.key == pygame.K_SPACE and face_tracking_active:
                    ret, frame = cap.read()
                    if ret:
                        results = face_mesh.process(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
                        if results.multi_face_landmarks:
                            face_x, face_y = results.multi_face_landmarks[0].landmark[1].x, results.multi_face_landmarks[0].landmark[1].y
                            g_center_offset_x, g_center_offset_y = face_x - 0.5, face_y - 0.5
                    g_smooth_cursor_x, g_smooth_cursor_y = 0.5, 0.5
                # Other key events...
                elif event.key == pygame.K_TAB: BLENDING_ENABLED = not BLENDING_ENABLED
                elif event.key == pygame.K_UP: BLENDING_INTENSITY = min(BLENDING_INTENSITY + 0.2, 5.0)
                elif event.key == pygame.K_DOWN: BLENDING_INTENSITY = max(BLENDING_INTENSITY - 0.2, 0.0)
                elif event.key == pygame.K_RIGHT: BLENDING_RADIUS = min(BLENDING_RADIUS + 0.5, 40.0) 
                elif event.key == pygame.K_LEFT: BLENDING_RADIUS = max(BLENDING_RADIUS - 0.5, 0.0)

        # --- Input handling (face or mouse) ---
        cursor_target_x, cursor_target_y = g_smooth_cursor_x, g_smooth_cursor_y
        if face_tracking_active:
            ret, frame = cap.read()
            if ret:
                results = face_mesh.process(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    face_x, face_y = results.multi_face_landmarks[0].landmark[1].x, results.multi_face_landmarks[0].landmark[1].y
                    cursor_target_x = (face_x - g_center_offset_x - 0.5) * MOVEMENT_SCALE + 0.5
                    cursor_target_y = (face_y - g_center_offset_y - 0.5) * MOVEMENT_SCALE + 0.5
        else:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            cursor_target_x, cursor_target_y = mouse_x / WINDOW_WIDTH, mouse_y / WINDOW_HEIGHT

        g_smooth_cursor_x = smooth_value(g_smooth_cursor_x, cursor_target_x, SMOOTHING_FACTOR)
        g_smooth_cursor_y = smooth_value(g_smooth_cursor_y, cursor_target_y, SMOOTHING_FACTOR)
        final_cursor_x = np.clip(g_smooth_cursor_x, 0.0, 1.0)
        final_cursor_y = np.clip(g_smooth_cursor_y, 0.0, 1.0)
        
        # --- Generate and play sound using the sound model ---
        pan_value = (final_cursor_x - 0.5) * 2.0
        audio_wave = sound_model.create_stereo_wave(current_image, final_cursor_x, final_cursor_y, pan_value)
        sound = pygame.sndarray.make_sound(audio_wave)
        sound.play()

        # --- Drawing and UI ---
        screen.fill((0,0,0))
        img_surface = pygame.surfarray.make_surface(np.transpose(current_image, (1, 0, 2)))
        screen.blit(pygame.transform.scale(img_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)), (0, 0))
        
        cursor_px, cursor_py = int(final_cursor_x * WINDOW_WIDTH), int(final_cursor_y * WINDOW_HEIGHT)
        pygame.draw.circle(screen, (255,255,255), (cursor_px, cursor_py), 12, 2)

        draw_info_panel(screen, font, {
            "Active Model": "1: Naturalistic",
            "Global Mix": f"{'ON' if g_global_mix_enabled else 'OFF'} (G)",
            "Blending": f"{'ON' if BLENDING_ENABLED else 'OFF'} (TAB)",
        })
        
        pygame.display.flip()
        clock.tick(30)

    if cap: cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
