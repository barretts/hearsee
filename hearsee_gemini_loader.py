import pygame
import numpy as np
import sys
import cv2
import mediapipe as mp
import math
import colorsys # Needed for RGB to HSL conversion
from tkinter import Tk, filedialog # Needed for the file open dialog

# ==============================================================================
# 1. INITIALIZATION AND SETUP
# ==============================================================================

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# --- Audio Settings for HSL Sonification ---
# We now define a frequency range that will be mapped to the Hue color wheel.
MIN_FREQUENCY = 120  # Corresponds to Red (0 degrees on the color wheel)
MAX_FREQUENCY = 1200 # Corresponds to Magenta/Violet (approaching 360 degrees)
SAMPLE_RATE = 44100
DURATION = 0.2
MASTER_VOLUME = 0.05 # Slightly increased master volume as amplitude is now controlled by color lightness.

# --- Visual & Grid Settings ---
GRID_ROWS, GRID_COLS = 16, 16
GRID_CELL_SIZE = 40
WINDOW_WIDTH = GRID_COLS * GRID_CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * GRID_CELL_SIZE

# --- Face Tracking & Interaction Settings ---
SMOOTHING_FACTOR = 0.3
MOVEMENT_SCALE = 5.0
BLENDING_ENABLED = True # Blending is now the default, as it's more informative.
BLENDING_INTENSITY = 1.0
BLENDING_RADIUS = 2.0

# --- Global variables ---
g_smooth_cursor_x, g_smooth_cursor_y = 0.5, 0.5
g_center_offset_x, g_center_offset_y = 0.0, 0.0


# ==============================================================================
# 2. IMAGE AND COLOR-TO-SOUND CONVERSION (HSL-BASED)
# ==============================================================================

def generate_child_landscape():
    """Generates a simple 256x256 pixel image of a landscape to use as a default."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Sky
    img[0:154, :] = (135, 206, 235)
    # Sun
    cv2.circle(img, (200, 60), 25, (255, 255, 0), -1)
    # Grass
    img[154:230, :] = (34, 139, 34)
    # Ground
    img[230:256, :] = (139, 69, 19)
    # House
    house_x, house_y = 128, 180
    house_width, house_height = 60, 40
    cv2.rectangle(img, (house_x - house_width//2, house_y), (house_x + house_width//2, house_y + house_height), (160, 82, 45), -1)
    roof_points = np.array([[house_x, house_y - 20], [house_x - house_width//2, house_y], [house_x + house_width//2, house_y]])
    cv2.fillPoly(img, [roof_points], (160, 82, 45))
    return img

def load_image_from_file():
    """Opens a file dialog to let the user select an image. Returns a Pygame-compatible surface."""
    # We need a root Tkinter window to open a dialog, but we don't want to see it.
    root = Tk()
    root.withdraw()
    # Ask the user for a file path.
    filepath = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not filepath:
        return None # User cancelled
    
    try:
        # Load the image using OpenCV, then convert to the format needed for the app.
        img_bgr = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def get_pixel_color_from_image(img, normalized_x, normalized_y):
    """Safely gets the RGB color from the image at a given normalized (0.0 to 1.0) coordinate."""
    if 0 <= normalized_x <= 1 and 0 <= normalized_y <= 1:
        img_height, img_width, _ = img.shape
        pixel_x = int(normalized_x * (img_width - 1))
        pixel_y = int(normalized_y * (img_height - 1))
        return img[pixel_y, pixel_x]
    return (0, 0, 0)

def color_to_sound_properties(r, g, b):
    """
    Converts an RGB color into a set of sound properties using the HSL color model.
    - Hue -> Frequency (Pitch)
    - Lightness -> Amplitude (Volume)
    - Saturation -> Timbre (Complexity of the sound)
    """
    # Normalize RGB values to be between 0 and 1 for the conversion.
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    
    # Convert to HLS (Hue, Lightness, Saturation). colorsys returns values between 0 and 1.
    h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
    
    # Map Hue (0-1) to our defined frequency range.
    frequency = MIN_FREQUENCY + (h * (MAX_FREQUENCY - MIN_FREQUENCY))
    
    # Lightness (0-1) directly maps to amplitude.
    # We square it to make the volume changes more perceptible (our ears perceive loudness logarithmically).
    amplitude = l
    
    # Saturation (0-1) controls the timbre.
    # 0 = pure sine wave (grayscale colors), 1 = rich with harmonics.
    timbre_complexity = s
    
    return (frequency, amplitude, timbre_complexity)

def create_mixed_audio_wave(landscape_img, cursor_x, cursor_y, pan_x):
    """
    Creates the final stereo sound wave based on the HSL properties of the color under the cursor,
    optionally blending the properties of surrounding colors.
    """
    
    # --- Step 1: Get the sound properties of the color at the cursor's location. ---
    center_color_rgb = get_pixel_color_from_image(landscape_img, cursor_x, cursor_y)
    center_freq, center_amp, center_timbre = color_to_sound_properties(*center_color_rgb)

    final_freq, final_amp, final_timbre = center_freq, center_amp, center_timbre
    
    # --- Step 2: If blending is enabled, average the properties of surrounding colors. ---
    if BLENDING_ENABLED and BLENDING_INTENSITY > 0:
        properties_to_average = [(center_freq, center_amp, center_timbre)]
        
        pixel_size_norm = 1.0 / landscape_img.shape[0]
        num_samples = int(BLENDING_RADIUS * 2)

        for i in range(1, num_samples + 1):
            # Sample in a simple cross pattern for efficiency
            for dx_step, dy_step in [(i, 0), (-i, 0), (0, i), (0, -i)]:
                sample_x = cursor_x + (dx_step * pixel_size_norm)
                sample_y = cursor_y + (dy_step * pixel_size_norm)
                
                surround_color_rgb = get_pixel_color_from_image(landscape_img, sample_x, sample_y)
                if surround_color_rgb is not None:
                    props = color_to_sound_properties(*surround_color_rgb)
                    properties_to_average.append(props)

        # Calculate the average of all collected properties.
        # This creates a smooth blend of pitch, volume, and timbre.
        num_props = len(properties_to_average)
        avg_freq = sum(p[0] for p in properties_to_average) / num_props
        avg_amp = sum(p[1] for p in properties_to_average) / num_props
        avg_timbre = sum(p[2] for p in properties_to_average) / num_props
        
        # The final sound is a mix of the center and the average of the surroundings.
        final_freq = (center_freq + avg_freq * BLENDING_INTENSITY) / (1 + BLENDING_INTENSITY)
        final_amp = (center_amp + avg_amp * BLENDING_INTENSITY) / (1 + BLENDING_INTENSITY)
        final_timbre = (center_timbre + avg_timbre * BLENDING_INTENSITY) / (1 + BLENDING_INTENSITY)

    # --- Step 3: Generate the audio wave from the final properties. ---
    if final_amp <= 0.01: # If the color is essentially black, return silence.
        return np.zeros((int(SAMPLE_RATE * DURATION), 2), dtype=np.int16)

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Create the base pure tone (sine wave).
    pure_wave = np.sin(final_freq * t * 2 * np.pi)
    
    # Create a harmonically rich tone (sum of sine waves at harmonic frequencies).
    # This represents the "saturated" sound.
    complex_wave = (np.sin(final_freq * t * 2 * np.pi) + 
                    0.5 * np.sin(final_freq * 2 * t * 2 * np.pi) + 
                    0.3 * np.sin(final_freq * 3 * t * 2 * np.pi))
    
    # Mix the pure and complex waves based on the final timbre value.
    # If timbre is 0 (grayscale), we only get the pure wave.
    # If timbre is 1 (fully saturated), we get a mix biased towards the complex wave.
    mixed_mono_wave = (1 - final_timbre) * pure_wave + final_timbre * complex_wave

    # Normalize the resulting mono wave so its peak is 1.0.
    max_amplitude = np.max(np.abs(mixed_mono_wave))
    if max_amplitude > 0:
        mixed_mono_wave /= max_amplitude

    # --- Step 4: Apply final amplitude (from Lightness), master volume, and pan to stereo. ---
    final_mono_wave_with_volume = mixed_mono_wave * final_amp * MASTER_VOLUME
    
    left_vol = (1.0 - pan_x) / 2.0
    right_vol = (1.0 + pan_x) / 2.0
    
    stereo_wave = np.zeros((len(final_mono_wave_with_volume), 2), dtype=np.int16)
    stereo_wave[:, 0] = (final_mono_wave_with_volume * left_vol * 32767).astype(np.int16)
    stereo_wave[:, 1] = (final_mono_wave_with_volume * right_vol * 32767).astype(np.int16)
    
    return stereo_wave


# ==============================================================================
# 3. FACE TRACKING AND UTILITIES
# ==============================================================================

def get_face_position(landmarks):
    """Extracts the normalized (0-1) coordinates of the nose tip from MediaPipe landmarks."""
    return landmarks[1].x, landmarks[1].y

def smooth_value(current_value, target_value, factor):
    """Applies exponential smoothing to a value to reduce jitter."""
    return current_value + (target_value - current_value) * factor

def draw_info_panel(screen, font, data):
    """Draws a panel on the screen with debugging and status information."""
    y_offset = 10
    for key, value in data.items():
        text_surface = font.render(f"{key}: {value}", True, (255, 255, 255))
        screen.blit(text_surface, (10, y_offset))
        y_offset += 25

# ==============================================================================
# 4. MAIN APPLICATION LOOP
# ==============================================================================

def main():
    """The main function where the application runs."""
    global g_smooth_cursor_x, g_smooth_cursor_y, g_center_offset_x, g_center_offset_y
    global BLENDING_INTENSITY, BLENDING_RADIUS, BLENDING_ENABLED
    
    # --- Initialize MediaPipe and Webcam ---
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)
        face_tracking_active = cap.isOpened()
        if face_tracking_active: print("✅ Face tracking initialized.")
        else: print("⚠️ Could not open webcam. Falling back to mouse control.")
    except Exception as e:
        print(f"⚠️ Could not initialize MediaPipe: {e}. Falling back to mouse control.")
        face_mesh = None
        cap = None
        face_tracking_active = False

    # --- Setup Pygame Display and Initial Image ---
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("HearSee - HSL Soundscape")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # Generate or load the initial image.
    current_image = generate_child_landscape()
    # Convert from OpenCV's BGR to Pygame's required RGB format.
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

    print("\n🚀 Starting HearSee (HSL Mode)...")
    print("Press [L] to load a new image.")
    print("Press [ESC] to quit.")

    running = True
    while running:
        # --- Event Handling (Keyboard Input) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    print("Opening file dialog...")
                    new_image = load_image_from_file()
                    if new_image is not None:
                        current_image = new_image
                        print("✅ New image loaded.")
                elif event.key == pygame.K_SPACE and face_tracking_active:
                    ret, frame = cap.read()
                    if ret:
                        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if results.multi_face_landmarks:
                            face_x, face_y = get_face_position(results.multi_face_landmarks[0].landmark)
                            g_center_offset_x = face_x - 0.5
                            g_center_offset_y = face_y - 0.5
                            print("✨ Center re-calibrated.")
                    g_smooth_cursor_x, g_smooth_cursor_y = 0.5, 0.5
                elif event.key == pygame.K_TAB:
                    BLENDING_ENABLED = not BLENDING_ENABLED
                elif event.key == pygame.K_UP: BLENDING_INTENSITY = min(BLENDING_INTENSITY + 0.2, 5.0)
                elif event.key == pygame.K_DOWN: BLENDING_INTENSITY = max(BLENDING_INTENSITY - 0.2, 0.0)
                elif event.key == pygame.K_RIGHT: BLENDING_RADIUS = min(BLENDING_RADIUS + 0.5, 10.0)
                elif event.key == pygame.K_LEFT: BLENDING_RADIUS = max(BLENDING_RADIUS - 0.5, 0.0)

        # --- Update Cursor Position (Face or Mouse) ---
        cursor_target_x, cursor_target_y = g_smooth_cursor_x, g_smooth_cursor_y
        if face_tracking_active:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    face_x, face_y = get_face_position(results.multi_face_landmarks[0].landmark)
                    calibrated_x = (face_x - g_center_offset_x - 0.5) * MOVEMENT_SCALE + 0.5
                    calibrated_y = (face_y - g_center_offset_y - 0.5) * MOVEMENT_SCALE + 0.5
                    cursor_target_x, cursor_target_y = calibrated_x, calibrated_y
        else:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            cursor_target_x = mouse_x / WINDOW_WIDTH
            cursor_target_y = mouse_y / WINDOW_HEIGHT

        g_smooth_cursor_x = smooth_value(g_smooth_cursor_x, cursor_target_x, SMOOTHING_FACTOR)
        g_smooth_cursor_y = smooth_value(g_smooth_cursor_y, cursor_target_y, SMOOTHING_FACTOR)
        final_cursor_x = np.clip(g_smooth_cursor_x, 0.0, 1.0)
        final_cursor_y = np.clip(g_smooth_cursor_y, 0.0, 1.0)

        # --- Audio Generation ---
        pan_value = (final_cursor_x - 0.5) * 2.0
        audio_wave = create_mixed_audio_wave(current_image, final_cursor_x, final_cursor_y, pan_value)
        sound = pygame.sndarray.make_sound(audio_wave)
        sound.play()

        # --- Drawing and Display ---
        screen.fill((0,0,0))
        landscape_surface = pygame.surfarray.make_surface(np.transpose(current_image, (1, 0, 2)))
        screen.blit(pygame.transform.scale(landscape_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)), (0, 0))

        cursor_pixel_x = int(final_cursor_x * WINDOW_WIDTH)
        cursor_pixel_y = int(final_cursor_y * WINDOW_HEIGHT)
        pygame.draw.line(screen, (255,255,255), (cursor_pixel_x - 10, cursor_pixel_y), (cursor_pixel_x + 10, cursor_pixel_y), 3)
        pygame.draw.line(screen, (255,255,255), (cursor_pixel_x, cursor_pixel_y - 10), (cursor_pixel_x, cursor_pixel_y + 10), 3)
        pygame.draw.circle(screen, (255,255,255), (cursor_pixel_x, cursor_pixel_y), 12, 2)

        # Get color properties to display them on the info panel.
        rgb_color = get_pixel_color_from_image(current_image, final_cursor_x, final_cursor_y)
        h, l, s = colorsys.rgb_to_hls(rgb_color[0]/255.0, rgb_color[1]/255.0, rgb_color[2]/255.0)
        
        info_data = {
            "HUE (Pitch)": f"{h:.2f}",
            "LIGHTNESS (Volume)": f"{l:.2f}",
            "SATURATION (Timbre)": f"{s:.2f}",
            "Blending": f"{'ON' if BLENDING_ENABLED else 'OFF'} (TAB)",
            "Blend Intensity": f"{BLENDING_INTENSITY:.1f} (UP/DOWN)",
            "Status": "Face Tracking" if face_tracking_active else "Mouse Control (L to Load Image)"
        }
        draw_info_panel(screen, font, info_data)
        
        pygame.display.flip()
        clock.tick(30)

    # --- Cleanup ---
    if face_tracking_active and cap is not None:
        cap.release()
    pygame.quit()
    print("Program terminated. Goodbye! 👋")


if __name__ == "__main__":
    main()
