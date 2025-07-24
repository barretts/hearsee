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
g_show_info_panel = True # Toggle for the detailed info panel
g_active_model_index = 0 # To switch between sound models

# ==============================================================================
# 2. MODULAR SOUND MODEL ARCHITECTURE (Phase 1.2)
# ==============================================================================

class SoundModel:
    """
    Base class for all sound models. Handles common attributes and stereo panning.
    """
    def __init__(self, name, sample_rate=44100, master_volume=0.05):
        self.name = name
        self.sample_rate = sample_rate
        self.duration = 0.2
        self.master_volume = master_volume
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)

    def color_to_properties(self, r, g, b):
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def generate_mono_wave(self, *args):
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def create_stereo_wave(self, landscape_img, cursor_x, cursor_y, pan_x):
        """Creates the final stereo sound wave based on image data and cursor position."""
        center_color_rgb = get_pixel_color_from_image(landscape_img, cursor_x, cursor_y)
        if center_color_rgb is None: center_color_rgb = (0,0,0)
        
        center_props = self.color_to_properties(*center_color_rgb)
        
        # Blending Logic
        if BLENDING_ENABLED and BLENDING_INTENSITY > 0 and BLENDING_RADIUS > 0:
            # This logic can be complex and model-specific, but we'll use a generic property averaging for now
            # A more advanced implementation might blend waves instead of properties
            prop_list = [center_props]
            # (Simplified blending for brevity - a full implementation would be more robust)
            focused_props = np.mean(prop_list, axis=0)
        else:
            focused_props = center_props

        focused_wave = self.generate_mono_wave(*focused_props)

        # Global mix logic
        if g_global_mix_enabled:
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

        left_vol, right_vol = (1.0 - pan_x) / 2.0, (1.0 + pan_x) / 2.0
        stereo_wave = np.zeros((len(self.t), 2), dtype=np.int16)
        stereo_wave[:, 0] = (final_wave * left_vol).astype(np.int16)
        stereo_wave[:, 1] = (final_wave * right_vol).astype(np.int16)
        return stereo_wave

class NaturalisticSoundModel(SoundModel):
    def __init__(self):
        super().__init__("Naturalistic")
        self.min_frequency = 120
        self.max_frequency = 1200

    def color_to_properties(self, r, g, b):
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
        h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
        if 0.5 < h < 0.75: # Calmer blues
            frequency = self.min_frequency + ((h - 0.5) / 0.25 * (self.min_frequency * 2))
        else:
            frequency = self.min_frequency + (h * (self.max_frequency - self.min_frequency))
        return (frequency, l, s)

    def generate_mono_wave(self, freq, amp, timbre):
        pure_wave = np.sin(freq * self.t * 2 * np.pi)
        complex_wave = (pure_wave + 0.5 * np.sin(freq * 2 * self.t * 2 * np.pi))
        mono_wave = ((1 - timbre) * pure_wave + timbre * complex_wave)
        if amp > 0.9 and timbre > 0.9: # Glare
             mono_wave += np.sin(freq * 1.05 * self.t * 2 * np.pi) * 0.4
        return mono_wave * amp

class SymbolicSoundModel(SoundModel):
    def __init__(self):
        super().__init__("Symbolic/Musical")
        self.base_freq = 110 # A2
        self.circle_of_fifths = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5] # Semitones from root

    def color_to_properties(self, r, g, b):
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
        h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
        return (h, l, s)

    def generate_mono_wave(self, hue, lightness, saturation):
        key_index = int(hue * 12) % 12
        root_note_semitone = self.circle_of_fifths[key_index]
        
        octave = 1 + int(lightness * 3) # 3 octaves
        
        root_freq = self.base_freq * (2**(octave)) * (2**(root_note_semitone/12))
        
        # Determine chord type
        # Cool colors -> minor, warm colors -> major
        is_major = (hue < 0.25 or hue > 0.75)
        third_interval = 4 if is_major else 3
        
        # Build chord based on saturation
        wave = np.sin(root_freq * self.t * 2 * np.pi) # Root note
        if saturation > 0.2: # Add fifth
            fifth_freq = root_freq * (2**(7/12))
            wave += np.sin(fifth_freq * self.t * 2 * np.pi)
        if saturation > 0.5: # Add third
            third_freq = root_freq * (2**(third_interval/12))
            wave += np.sin(third_freq * self.t * 2 * np.pi)
            
        return wave * lightness

class EmotionalSoundModel(SoundModel):
    def __init__(self):
        super().__init__("Direct Emotional")
        self.base_freq = 150

    def color_to_properties(self, r, g, b):
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
        h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
        return (h, l, s)

    def generate_mono_wave(self, hue, lightness, saturation):
        # Hue -> Dissonance
        dissonance = 0.5 - abs(hue - 0.5) # Peaks at red (0) and purple (1), lowest at green/cyan
        
        freq1 = self.base_freq * (1 + lightness)
        freq2 = freq1 * (1 + dissonance * 0.1) # More dissonance = wider interval
        
        wave1 = np.sin(freq1 * self.t * 2 * np.pi)
        wave2 = np.sin(freq2 * self.t * 2 * np.pi)
        wave = (wave1 + wave2) * 0.5
        
        # Saturation -> Arousal (tremolo speed)
        tremolo_speed = 1 + saturation * 20
        tremolo_env = 0.7 + 0.3 * np.sin(tremolo_speed * self.t * 2 * np.pi)
        
        return wave * tremolo_env * lightness


# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================

def generate_child_landscape():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    SKY_BLUE, SUN_YELLOW, GRASS_GREEN, GROUND_BROWN = (135, 206, 235), (255, 255, 0), (34, 139, 34), (139, 69, 19)
    HOUSE_BROWN, DOOR_BLACK, CHIMNEY_BROWN = (160, 82, 45), (50, 50, 50), (101, 67, 33)
    img[0:154, :] = SKY_BLUE
    cv2.circle(img, (200, 60), 25, SUN_YELLOW, -1)
    img[154:230, :] = GRASS_GREEN
    img[230:256, :] = GROUND_BROWN
    house_x, house_y = 128, 180
    cv2.rectangle(img, (house_x - 30, house_y), (house_x + 30, house_y + 40), HOUSE_BROWN, -1)
    cv2.fillPoly(img, [np.array([[house_x, house_y-20], [house_x-30, house_y], [house_x+30, house_y]], np.int32)], HOUSE_BROWN)
    cv2.rectangle(img, (house_x - 6, house_y + 10), (house_x + 6, house_y + 35), DOOR_BLACK, -1)
    cv2.rectangle(img, (house_x + 15, house_y - 35), (house_x + 23, house_y), CHIMNEY_BROWN, -1)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def load_image_from_file():
    root = Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename()
    if not filepath: return None
    try: return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    except: return None

def get_pixel_color_from_image(img, x, y):
    if 0 <= x <= 1 and 0 <= y <= 1:
        h, w, _ = img.shape
        return img[int(y * (h - 1)), int(x * (w - 1))]
    return None

def smooth_value(current, target, factor): return current + (target - current) * factor

def draw_info_panel(screen, font, data):
    y_offset = 10
    for key, value in data.items():
        text_surface = font.render(f"{key}: {value}", True, (255, 255, 255), (0,0,0,128))
        screen.blit(text_surface, (10, y_offset))
        y_offset += 25

# ==============================================================================
# 4. MAIN APPLICATION LOOP
# ==============================================================================

def main():
    global g_smooth_cursor_x, g_smooth_cursor_y, g_center_offset_x, g_center_offset_y
    global BLENDING_INTENSITY, BLENDING_RADIUS, BLENDING_ENABLED, g_global_mix_enabled, g_show_info_panel, g_active_model_index
    
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)
        face_tracking_active = cap.isOpened()
    except Exception as e:
        face_mesh, cap, face_tracking_active = None, None, False

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("HearSee - Phase 1.2")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    sound_models = [NaturalisticSoundModel(), SymbolicSoundModel(), EmotionalSoundModel()]
    current_image = cv2.cvtColor(generate_child_landscape(), cv2.COLOR_BGR2RGB)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                    g_active_model_index = event.key - pygame.K_1
                elif event.key == pygame.K_l:
                    new_image = load_image_from_file()
                    if new_image is not None: current_image = new_image
                elif event.key == pygame.K_g: g_global_mix_enabled = not g_global_mix_enabled
                elif event.key == pygame.K_BACKQUOTE: g_show_info_panel = not g_show_info_panel
                elif event.key == pygame.K_SPACE and face_tracking_active:
                    ret, frame = cap.read()
                    if ret:
                        results = face_mesh.process(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
                        if results.multi_face_landmarks:
                            face_x, face_y = results.multi_face_landmarks[0].landmark[1].x, results.multi_face_landmarks[0].landmark[1].y
                            g_center_offset_x, g_center_offset_y = face_x - 0.5, face_y - 0.5
                    g_smooth_cursor_x, g_smooth_cursor_y = 0.5, 0.5
                elif event.key == pygame.K_TAB: BLENDING_ENABLED = not BLENDING_ENABLED
                elif event.key == pygame.K_UP: BLENDING_INTENSITY = min(BLENDING_INTENSITY + 0.2, 5.0)
                elif event.key == pygame.K_DOWN: BLENDING_INTENSITY = max(BLENDING_INTENSITY - 0.2, 0.0)
                elif event.key == pygame.K_RIGHT: BLENDING_RADIUS = min(BLENDING_RADIUS + 0.5, 40.0) 
                elif event.key == pygame.K_LEFT: BLENDING_RADIUS = max(BLENDING_RADIUS - 0.5, 0.0)

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
        
        active_model = sound_models[g_active_model_index]
        pan_value = (final_cursor_x - 0.5) * 2.0
        audio_wave = active_model.create_stereo_wave(current_image, final_cursor_x, final_cursor_y, pan_value)
        sound = pygame.sndarray.make_sound(audio_wave)
        sound.play()

        screen.fill((0,0,0))
        img_surface = pygame.surfarray.make_surface(np.transpose(current_image, (1, 0, 2)))
        screen.blit(pygame.transform.scale(img_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)), (0, 0))
        
        cursor_px, cursor_py = int(final_cursor_x * WINDOW_WIDTH), int(final_cursor_y * WINDOW_HEIGHT)
        pygame.draw.circle(screen, (255,255,255), (cursor_px, cursor_py), 12, 2)
        pygame.draw.circle(screen, (0,0,0), (cursor_px, cursor_py), 14, 1)

        info_data = {
            "Active Model": f"{g_active_model_index+1}: {active_model.name}",
            "Global Mix": f"{'ON' if g_global_mix_enabled else 'OFF'} (G)",
            "Blending": f"{'ON' if BLENDING_ENABLED else 'OFF'} (TAB)",
        }
        if g_show_info_panel:
            info_data.update({
                "--- Controls ---": "",
                "Blend Radius": f"{BLENDING_RADIUS:.1f} (L/R)",
                "Blend Intensity": f"{BLENDING_INTENSITY:.1f} (U/D)",
                "--- Cursor Info ---": "",
                "Position": f"({final_cursor_x:.2f}, {final_cursor_y:.2f})",
            })
            if face_tracking_active:
                info_data.update({ "Center Offset": f"({g_center_offset_x:.2f}, {g_center_offset_y:.2f}) (SPACE)"})
        
        draw_info_panel(screen, font, info_data)
        
        pygame.display.flip()
        clock.tick(30)

    if cap: cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
