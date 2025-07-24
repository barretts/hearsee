import pygame
import numpy as np
import sys
import cv2
import mediapipe as mp
import math

# ==============================================================================
# 1. INITIALIZATION AND SETUP
# ==============================================================================

# Initialize Pygame and its mixer for audio playback.
# Using a standard 44.1kHz sample rate and 16-bit stereo audio.
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# --- Audio Settings ---
# These frequencies are chosen to be distinct for major colors in the landscape.
# This helps the user differentiate between objects by sound.
FREQ_SUN = 800      # A high, bright frequency for the sun.
FREQ_SKY = 150      # A low, deep bass frequency for the vast sky.
FREQ_GRASS = 500    # A clear, mid-range frequency for grass.
FREQ_HOUSE = 300    # A warm, lower-mid frequency for the house and ground.
FREQ_DARK = 250     # A low tone for dark objects like the door.
SAMPLE_RATE = 44100 # CD-quality audio sample rate.
DURATION = 0.2      # Duration of each sound chunk in seconds. A shorter duration makes the sound more responsive.
MASTER_VOLUME = 0.03 # A very low master volume to prevent loud surprises and protect hearing.

# --- Visual & Grid Settings ---
GRID_ROWS, GRID_COLS = 16, 16 # The visual grid displayed on screen.
GRID_CELL_SIZE = 40           # The size of each cell in pixels.
WINDOW_WIDTH = GRID_COLS * GRID_CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * GRID_CELL_SIZE

# --- Face Tracking & Interaction Settings ---
SMOOTHING_FACTOR = 0.3      # Reduces jitter from face tracking. Higher value = more smoothing but more lag.
MOVEMENT_SCALE = 5.0        # Amplifies head movement to make it easier to scan the whole image.
BLENDING_ENABLED = False    # Master switch to turn sound blending on or off (Toggled with TAB key).
BLENDING_INTENSITY = 1.0    # How much influence surrounding colors have. Higher value = more blending. (Adjust with UP/DOWN keys).
BLENDING_RADIUS = 2.0       # How far away (in pixels) the program looks for surrounding colors to blend. (Adjust with LEFT/RIGHT keys).

# --- Color Definitions (BGR for OpenCV, but used as RGB in Pygame) ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY_BLUE = (135, 206, 235)
GRASS_GREEN = (34, 139, 34)
GROUND_BROWN = (139, 69, 19)
SUN_YELLOW = (255, 255, 0)
HOUSE_BROWN = (160, 82, 45)
DOOR_BLACK = (50, 50, 50)
CHIMNEY_BROWN = (101, 67, 33)

# --- Global variables for smoothed cursor position and calibration ---
# These store the "memory" of the cursor's position to allow for smoothing.
g_smooth_cursor_x, g_smooth_cursor_y = 0.5, 0.5
# These store the offset needed to center the cursor when the user presses SPACE.
g_center_offset_x, g_center_offset_y = 0.0, 0.0


# ==============================================================================
# 2. IMAGE AND COLOR-TO-SOUND CONVERSION
# ==============================================================================

def generate_child_landscape():
    """Generates a simple 256x256 pixel image of a landscape.
    
    This image serves as the "world" that will be sonified. The colors are
    kept simple and distinct to make the soundscape easier to interpret.
    """
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Sky (top 60%)
    img[0:154, :] = SKY_BLUE
    
    # Sun (circle in the top right)
    cv2.circle(img, (200, 60), 25, SUN_YELLOW, -1)
    
    # Grass (middle 30%)
    img[154:230, :] = GRASS_GREEN
    
    # Ground (bottom 10%)
    img[230:256, :] = GROUND_BROWN
    
    # House (a simple brown rectangle with a roof)
    house_x, house_y = 128, 180
    house_width, house_height = 60, 40
    # House body
    img[house_y:house_y+house_height, house_x-house_width//2:house_x+house_width//2] = HOUSE_BROWN
    # Roof (triangle)
    roof_points = np.array([[house_x, house_y - 20], [house_x - house_width//2, house_y], [house_x + house_width//2, house_y]])
    cv2.fillPoly(img, [roof_points], HOUSE_BROWN)
    
    # Door and Chimney
    img[house_y + 10:house_y + 35, house_x - 6:house_x + 6] = DOOR_BLACK
    img[house_y - 35:house_y - 20, house_x + 15:house_x + 23] = CHIMNEY_BROWN
    
    return img

def get_pixel_color_from_image(img, normalized_x, normalized_y):
    """Safely gets the BGR color from the landscape image at a given normalized (0.0 to 1.0) coordinate."""
    if 0 <= normalized_x <= 1 and 0 <= normalized_y <= 1:
        img_height, img_width, _ = img.shape
        pixel_x = int(normalized_x * (img_width - 1))
        pixel_y = int(normalized_y * (img_height - 1))
        return img[pixel_y, pixel_x]
    return (0, 0, 0) # Return black if out of bounds

def color_to_audio_frequency(r, g, b):
    """
    Converts an RGB color into a single dominant audio frequency.
    
    This is the core of the sonification. By mapping specific, easily
    distinguishable colors to specific frequencies, we create a predictable
    and learnable soundscape.
    """
    # Normalize color values to be between 0 and 1
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    
    # Use thresholds to identify the primary colors of our landscape.
    # This is more reliable than trying to map every single possible color.
    if r_norm > 0.8 and g_norm > 0.8:  # Yellow (Sun)
        return FREQ_SUN
    elif b_norm > 0.5 and r_norm < 0.6:  # Blue (Sky)
        return FREQ_SKY
    elif g_norm > 0.5 and r_norm < 0.4:  # Green (Grass)
        return FREQ_GRASS
    elif r_norm > 0.4 and g_norm > 0.2 and b_norm < 0.3:  # Brown (House/Ground)
        return FREQ_HOUSE
    elif r_norm < 0.3 and g_norm < 0.3 and b_norm < 0.3:  # Dark colors
        return FREQ_DARK
    else: # Fallback for other colors (like clouds or blended edges)
        # We can create a simple frequency based on which color channel is dominant.
        if r_norm > g_norm and r_norm > b_norm: return 400 + (r_norm * 200) # Red dominant
        if g_norm > r_norm and g_norm > b_norm: return 600 + (g_norm * 200) # Green dominant
        return 200 + (b_norm * 100) # Blue dominant

def create_mixed_audio_wave(landscape_img, cursor_x, cursor_y, pan_x):
    """
    This is the main audio generation function. It creates the final stereo sound wave
    by mixing the sound of the color at the cursor's position with the sounds of
    surrounding colors.
    
    *** KEY CHANGE: This function now correctly normalizes the final audio signal
    to prevent the volume from increasing when blending is active. ***
    """
    
    # --- Step 1: Get the primary color and frequency at the cursor's location. ---
    center_color_bgr = get_pixel_color_from_image(landscape_img, cursor_x, cursor_y)
    if center_color_bgr is None:
        # If cursor is out of bounds, return silence.
        return np.zeros((int(SAMPLE_RATE * DURATION), 2), dtype=np.int16)
        
    center_b, center_g, center_r = center_color_bgr
    center_freq = color_to_audio_frequency(center_r, center_g, center_b)

    # --- Step 2: Define weights for the center vs. surrounding sounds. ---
    # This determines how much of the final sound comes from the center vs. the blend.
    # A higher BLENDING_INTENSITY makes the surrounding sounds more prominent.
    if BLENDING_ENABLED and BLENDING_INTENSITY > 0:
        # The sum of center_weight and surrounding_weight is always 1.0.
        # This ensures the total potential amplitude doesn't change, solving the volume jump issue.
        center_weight = 1.0 / (1.0 + BLENDING_INTENSITY)
        surrounding_weight = BLENDING_INTENSITY / (1.0 + BLENDING_INTENSITY)
    else:
        # If blending is off, the center sound gets 100% of the weight.
        center_weight = 1.0
        surrounding_weight = 0.0

    # --- Step 3: Generate the audio wave for the center frequency. ---
    # We create a time array `t` and then generate a sine wave for the center frequency.
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    # The final mixed wave starts with just the sound from the center position.
    mixed_mono_wave = np.sin(center_freq * t * 2 * np.pi) * center_weight

    # --- Step 4: If blending is enabled, find and add surrounding sounds. ---
    if surrounding_weight > 0:
        surrounding_sounds_to_mix = []
        pixel_size_norm = 1.0 / landscape_img.shape[0] # The size of one pixel in normalized (0-1) coordinates.
        
        # Sample in a spiral pattern around the cursor to find different colors.
        for i in range(1, int(BLENDING_RADIUS * 5)):
            for dx_step, dy_step in [(i,0), (-i,0), (0,i), (0,-i)]: # Check up, down, left, right
                sample_x = cursor_x + (dx_step * pixel_size_norm)
                sample_y = cursor_y + (dy_step * pixel_size_norm)
                
                # Get the color of this surrounding pixel.
                surround_color_bgr = get_pixel_color_from_image(landscape_img, sample_x, sample_y)
                if surround_color_bgr is not None and tuple(surround_color_bgr) != tuple(center_color_bgr):
                    surround_r, surround_g, surround_b = surround_color_bgr[2], surround_color_bgr[1], surround_color_bgr[0]
                    surround_freq = color_to_audio_frequency(surround_r, surround_g, surround_b)
                    
                    # Avoid adding the same "hint" frequency multiple times.
                    if surround_freq not in [f for f, w in surrounding_sounds_to_mix]:
                         # The weight of each surrounding sound is proportional to the total surrounding_weight.
                        surrounding_sounds_to_mix.append((surround_freq, surrounding_weight / (len(surrounding_sounds_to_mix)+1)))

        # Add the collected surrounding sounds to our main wave.
        for freq, weight in surrounding_sounds_to_mix:
            mixed_mono_wave += np.sin(freq * t * 2 * np.pi) * weight
            
    # --- Step 5: Normalize the final mixed mono wave. ---
    # This is the CRITICAL step to prevent volume changes.
    # We find the loudest point in our combined wave and scale the entire wave
    # so that this loudest point is at maximum amplitude (1.0).
    max_amplitude = np.max(np.abs(mixed_mono_wave))
    if max_amplitude > 0:
        mixed_mono_wave /= max_amplitude

    # --- Step 6: Apply master volume and pan the mono sound into stereo. ---
    final_mono_wave_with_volume = mixed_mono_wave * MASTER_VOLUME
    
    # Pan the sound left or right based on the cursor's horizontal position.
    # pan_x is from -1.0 (full left) to 1.0 (full right).
    left_vol = (1.0 - pan_x) / 2.0
    right_vol = (1.0 + pan_x) / 2.0
    
    # Create the final stereo sound array.
    stereo_wave = np.zeros((len(final_mono_wave_with_volume), 2), dtype=np.int16)
    stereo_wave[:, 0] = (final_mono_wave_with_volume * left_vol * 32767).astype(np.int16)
    stereo_wave[:, 1] = (final_mono_wave_with_volume * right_vol * 32767).astype(np.int16)
    
    return stereo_wave


# ==============================================================================
# 3. FACE TRACKING AND UTILITIES
# ==============================================================================

def get_face_position(landmarks):
    """Extracts the normalized (0-1) coordinates of the nose tip from MediaPipe landmarks."""
    # The nose tip (landmark #1) is a stable point for tracking the center of the face.
    nose_tip = landmarks[1]
    return nose_tip.x, nose_tip.y

def smooth_value(current_value, target_value, factor):
    """Applies exponential smoothing to a value to reduce jitter."""
    return current_value + (target_value - current_value) * factor

def draw_info_panel(screen, font, data):
    """Draws a panel on the screen with debugging and status information."""
    y_offset = 10
    for key, value in data.items():
        text_surface = font.render(f"{key}: {value}", True, WHITE)
        screen.blit(text_surface, (10, y_offset))
        y_offset += 25

# ==============================================================================
# 4. MAIN APPLICATION LOOP
# ==============================================================================

def main():
    """The main function where the application runs."""
    # Make global variables accessible inside this function.
    global g_smooth_cursor_x, g_smooth_cursor_y, g_center_offset_x, g_center_offset_y
    global BLENDING_INTENSITY, BLENDING_RADIUS, BLENDING_ENABLED
    
    # --- Initialize MediaPipe Face Mesh ---
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Provides more accurate landmark points.
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        face_tracking_active = True
        print("✅ MediaPipe Face Mesh initialized.")
    except Exception as e:
        print(f"⚠️ Could not initialize MediaPipe Face Mesh: {e}. Falling back to mouse control.")
        face_mesh = None
        face_tracking_active = False

    # --- Initialize Webcam ---
    if face_tracking_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam. Disabling face tracking.")
            face_tracking_active = False
            cap = None
        else:
            print("✅ Webcam initialized.")

    # --- Setup Pygame Display and Assets ---
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("HearSee - Face-Controlled Soundscape")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # Generate the landscape image that will be "heard".
    landscape_image = generate_child_landscape()
    # Convert from OpenCV's BGR format to Pygame's RGB format.
    landscape_image = cv2.cvtColor(landscape_image, cv2.COLOR_BGR2RGB)


    print("\n🚀 Starting HearSee...")
    print("Press [ESC] to quit.")
    print("Press [SPACE] to re-calibrate the center position.")
    print("Press [TAB] to toggle sound blending.")
    print("Use [UP/DOWN] keys to change blending intensity.")
    print("Use [LEFT/RIGHT] keys to change blending radius.\n")

    running = True
    while running:
        # --- Event Handling (Keyboard Input) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Re-calibrate the center point based on the current face position.
                    if face_tracking_active and cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            if results.multi_face_landmarks:
                                face_x, face_y = get_face_position(results.multi_face_landmarks[0].landmark)
                                # Store the current face position as the new "zero" point.
                                g_center_offset_x = face_x - 0.5
                                g_center_offset_y = face_y - 0.5
                                print("✨ Center re-calibrated.")
                    # Reset the cursor to the center of the screen.
                    g_smooth_cursor_x, g_smooth_cursor_y = 0.5, 0.5
                elif event.key == pygame.K_TAB:
                    BLENDING_ENABLED = not BLENDING_ENABLED
                    print(f"Blending is now {'ON' if BLENDING_ENABLED else 'OFF'}")
                elif event.key == pygame.K_UP:
                    BLENDING_INTENSITY = min(BLENDING_INTENSITY + 0.2, 5.0)
                elif event.key == pygame.K_DOWN:
                    BLENDING_INTENSITY = max(BLENDING_INTENSITY - 0.2, 0.0)
                elif event.key == pygame.K_RIGHT:
                    BLENDING_RADIUS = min(BLENDING_RADIUS + 0.5, 10.0)
                elif event.key == pygame.K_LEFT:
                    BLENDING_RADIUS = max(BLENDING_RADIUS - 0.5, 0.0)

        # --- Update Cursor Position ---
        cursor_target_x, cursor_target_y = g_smooth_cursor_x, g_smooth_cursor_y
        if face_tracking_active and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Flip the frame horizontally so it acts like a mirror.
                frame = cv2.flip(frame, 1)
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    face_x, face_y = get_face_position(results.multi_face_landmarks[0].landmark)
                    # Apply calibration offset and movement scaling.
                    calibrated_x = (face_x - g_center_offset_x - 0.5) * MOVEMENT_SCALE + 0.5
                    calibrated_y = (face_y - g_center_offset_y - 0.5) * MOVEMENT_SCALE + 0.5
                    cursor_target_x, cursor_target_y = calibrated_x, calibrated_y
        else:
            # If no camera, control with the mouse.
            mouse_x, mouse_y = pygame.mouse.get_pos()
            cursor_target_x = mouse_x / WINDOW_WIDTH
            cursor_target_y = mouse_y / WINDOW_HEIGHT

        # Apply smoothing to the cursor's movement.
        g_smooth_cursor_x = smooth_value(g_smooth_cursor_x, cursor_target_x, SMOOTHING_FACTOR)
        g_smooth_cursor_y = smooth_value(g_smooth_cursor_y, cursor_target_y, SMOOTHING_FACTOR)
        
        # Clamp cursor values to be within the 0.0 to 1.0 range.
        final_cursor_x = np.clip(g_smooth_cursor_x, 0.0, 1.0)
        final_cursor_y = np.clip(g_smooth_cursor_y, 0.0, 1.0)

        # --- Audio Generation ---
        # Pan the sound from left (-1.0) to right (1.0).
        pan_value = (final_cursor_x - 0.5) * 2.0
        # Generate the audio wave based on the current cursor position.
        audio_wave = create_mixed_audio_wave(landscape_image, final_cursor_x, final_cursor_y, pan_value)
        # Play the sound.
        sound = pygame.sndarray.make_sound(audio_wave)
        sound.play()

        # --- Drawing and Display ---
        screen.fill(BLACK)
        
        # Draw the landscape image scaled to fit the window.
        landscape_surface = pygame.surfarray.make_surface(np.transpose(landscape_image, (1, 0, 2)))
        screen.blit(pygame.transform.scale(landscape_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)), (0, 0))

        # Draw the cursor as a white crosshair.
        cursor_pixel_x = int(final_cursor_x * WINDOW_WIDTH)
        cursor_pixel_y = int(final_cursor_y * WINDOW_HEIGHT)
        pygame.draw.line(screen, WHITE, (cursor_pixel_x - 10, cursor_pixel_y), (cursor_pixel_x + 10, cursor_pixel_y), 3)
        pygame.draw.line(screen, WHITE, (cursor_pixel_x, cursor_pixel_y - 10), (cursor_pixel_x, cursor_pixel_y + 10), 3)
        pygame.draw.circle(screen, WHITE, (cursor_pixel_x, cursor_pixel_y), 12, 2)

        # Get the color under the cursor to display its RGB values.
        bgr_color = get_pixel_color_from_image(landscape_image, final_cursor_x, final_cursor_y)
        r, g, b = bgr_color[2], bgr_color[1], bgr_color[0]
        
        # Display all the status information.
        info_data = {
            "Cursor Pos": f"({final_cursor_x:.2f}, {final_cursor_y:.2f})",
            "Color (RGB)": f"({r}, {g}, {b})",
            "Blending": f"{'ON' if BLENDING_ENABLED else 'OFF'} (TAB)",
            "Blend Intensity": f"{BLENDING_INTENSITY:.1f} (UP/DOWN)",
            "Blend Radius": f"{BLENDING_RADIUS:.1f} (LEFT/RIGHT)",
            "Status": "Face Tracking" if face_tracking_active else "Mouse Control"
        }
        draw_info_panel(screen, font, info_data)
        
        pygame.display.flip()
        clock.tick(30) # Limit the loop to 30 frames per second.

    # --- Cleanup ---
    if face_tracking_active and cap is not None:
        cap.release()
    pygame.quit()
    print("Program terminated safely. Goodbye! 👋")


if __name__ == "__main__":
    main()
