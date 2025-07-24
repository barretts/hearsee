import pygame
import numpy as np
import sys
import cv2
import mediapipe as mp
import math

# Initialize pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# Sound settings
FREQ_RED = 440    # A4
FREQ_GREEN = 554  # C#5
FREQ_BLUE = 659   # E5
SAMPLE_RATE = 44100
DURATION = 0.2
MASTER_VOLUME = 0.02  # Very quiet for safety

# Grid settings
GRID_ROWS, GRID_COLS = 10, 10
GRID_CELL_SIZE = 60
WINDOW_WIDTH = GRID_COLS * GRID_CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * GRID_CELL_SIZE

# Face tracking settings
PAN_RANGE_DEGREES = 10  # 10 degrees of panning range
SMOOTHING_FACTOR = 0.3  # Smoothing for face movement
MIN_FACE_CONFIDENCE = 0.5
BASE_PAN_SENSITIVITY = 4.0  # Increased base sensitivity
MOVEMENT_MULTIPLIER = 2.5   # Increased movement range

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Smoothing variables
smooth_x, smooth_y = 0.5, 0.5
smooth_pan_x, smooth_pan_y = 0.0, 0.0
current_sensitivity = BASE_PAN_SENSITIVITY

def generate_color_grid(rows, cols):
    grid = np.zeros((rows, cols, 3), dtype=np.uint8)
    for y in range(rows):
        for x in range(cols):
            grid[y, x] = (
                int(255 * x / (cols - 1)),
                int(255 * y / (rows - 1)),
                int(255 * (x + y) / (rows + cols - 2))
            )
    return grid

def generate_tone(frequency, volume=0.5):
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    audio = (tone * volume * MASTER_VOLUME * 32767).astype(np.int16)
    return audio

def pan_stereo(tone, left_vol, right_vol):
    """Convert mono tone to stereo with panning."""
    stereo = np.zeros((len(tone), 2), dtype=np.int16)
    stereo[:, 0] = (tone * left_vol).astype(np.int16)
    stereo[:, 1] = (tone * right_vol).astype(np.int16)
    return stereo

def calculate_adaptive_sensitivity(landmarks, frame_width, frame_height):
    """Calculate adaptive sensitivity based on face size."""
    global current_sensitivity
    
    # Get face bounding box
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    
    # Calculate face size
    face_width = (max(x_coords) - min(x_coords)) * frame_width
    face_height = (max(y_coords) - min(y_coords)) * frame_height
    face_area = face_width * face_height
    
    # Calculate frame area
    frame_area = frame_width * frame_height
    
    # Calculate face size ratio
    face_ratio = face_area / frame_area
    
    # Adaptive sensitivity: smaller face = higher sensitivity
    # Face ratio typically ranges from 0.01 (far) to 0.1 (close)
    target_sensitivity = BASE_PAN_SENSITIVITY / max(face_ratio * 15, 0.1)  # Increased sensitivity
    target_sensitivity = np.clip(target_sensitivity, 2.0, 12.0)  # Increased range
    
    # Smooth sensitivity changes
    current_sensitivity = smooth_value(current_sensitivity, target_sensitivity, 0.1)
    
    return current_sensitivity

def calculate_pan_from_face_angle(face_angle_x, face_angle_y):
    """Calculate stereo panning based on face angle with adaptive sensitivity."""
    # Apply adaptive sensitivity
    pan_x = np.clip(face_angle_x * current_sensitivity / (PAN_RANGE_DEGREES * np.pi / 180), -1, 1)
    pan_y = np.clip(face_angle_y * current_sensitivity / (PAN_RANGE_DEGREES * np.pi / 180), -1, 1)
    
    # Convert to left/right volumes
    left_vol = max(0, 1 - pan_x)
    right_vol = max(0, 1 + pan_x)
    
    # Normalize
    total_vol = left_vol + right_vol
    if total_vol > 0:
        left_vol /= total_vol
        right_vol /= total_vol
    
    return left_vol, right_vol

def map_face_to_grid_position(nose_x_norm, nose_y_norm, face_angle_x, face_angle_y):
    """Map face position and angle to grid position with increased range."""
    # Apply movement multiplier to increase range
    # Combine nose position with face angle for more range
    grid_x = nose_x_norm + (face_angle_x * MOVEMENT_MULTIPLIER / (PAN_RANGE_DEGREES * np.pi / 180))
    grid_y = nose_y_norm + (face_angle_y * MOVEMENT_MULTIPLIER / (PAN_RANGE_DEGREES * np.pi / 180))
    
    # Clamp to grid bounds with some overflow for edge detection
    grid_x = np.clip(grid_x, -0.2, 1.2)
    grid_y = np.clip(grid_y, -0.2, 1.2)
    
    return grid_x, grid_y

def generate_spatial_mix(grid, cx, cy, pan_x=0, pan_y=0):
    tones = []

    def safe_color(x, y):
        if 0 <= x < GRID_COLS and 0 <= y < GRID_ROWS:
            return grid[y, x]
        return (0, 0, 0)

    # Get center color
    b, g, r = safe_color(cx, cy)
    vol_r = r / 255.0
    vol_g = g / 255.0
    vol_b = b / 255.0

    # Calculate panning based on face angle
    left_vol, right_vol = calculate_pan_from_face_angle(pan_x, pan_y)

    # Generate tones with spatial positioning
    t_r = generate_tone(FREQ_RED, vol_r)
    t_g = generate_tone(FREQ_GREEN, vol_g)
    t_b = generate_tone(FREQ_BLUE, vol_b)
    
    mixed = t_r + t_g + t_b
    mixed = np.clip(mixed, -32768, 32767)
    stereo = pan_stereo(mixed, left_vol, right_vol)
    tones.append(stereo)

    # Add subtle spatial effects for neighboring cells
    neighbors = [
        (cx - 1, cy, 0.8, 0.2),    # left neighbor
        (cx + 1, cy, 0.2, 0.8),    # right neighbor
        (cx, cy - 1, 0.6, 0.6),    # above neighbor
        (cx, cy + 1, 0.6, 0.6),    # below neighbor
    ]

    for nx, ny, lv, rv in neighbors:
        b, g, r = safe_color(nx, ny)
        vol_r = r / 255.0 * 0.3  # Reduced volume for neighbors
        vol_g = g / 255.0 * 0.3
        vol_b = b / 255.0 * 0.3

        t_r = generate_tone(FREQ_RED * 1.05, vol_r)  # Slightly higher pitch
        t_g = generate_tone(FREQ_GREEN * 1.05, vol_g)
        t_b = generate_tone(FREQ_BLUE * 1.05, vol_b)
        
        mixed = t_r + t_g + t_b
        mixed = np.clip(mixed, -32768, 32767)
        stereo = pan_stereo(mixed, lv, rv)
        tones.append(stereo)

    final = np.sum(tones, axis=0)
    final = np.clip(final, -32768, 32767)
    return final.astype(np.int16)

def get_face_angles(landmarks, frame_width, frame_height):
    """Calculate face angles from MediaPipe landmarks."""
    # Get key facial landmarks for angle calculation
    nose_tip = landmarks[1]      # Nose tip
    left_eye = landmarks[33]     # Left eye outer corner
    right_eye = landmarks[263]   # Right eye outer corner
    left_ear = landmarks[234]    # Left ear
    right_ear = landmarks[454]   # Right ear
    
    # Calculate horizontal angle (left-right)
    eye_center_x = (left_eye.x + right_eye.x) / 2
    face_center_x = (left_ear.x + right_ear.x) / 2
    
    # Normalize to frame dimensions
    eye_center_x *= frame_width
    face_center_x *= frame_width
    
    angle_x = math.atan2(eye_center_x - face_center_x, 0.1)
    
    # Calculate vertical angle (up-down)
    nose_y = nose_tip.y * frame_height
    eye_center_y = (left_eye.y + right_eye.y) / 2 * frame_height
    angle_y = math.atan2(nose_y - eye_center_y, 0.1)
    
    return angle_x, angle_y

def smooth_value(current, target, factor):
    """Apply smoothing to reduce jitter."""
    return current + (target - current) * factor

def draw_pan_indicator(screen, pan_x, pan_y, font):
    """Draw a visual indicator for panning."""
    # Draw pan indicator circle
    center_x = WINDOW_WIDTH // 2
    center_y = WINDOW_HEIGHT - 100
    
    # Convert pan values to screen coordinates
    indicator_x = center_x + int(pan_x * 50)
    indicator_y = center_y + int(pan_y * 30)
    
    # Draw background circle
    pygame.draw.circle(screen, WHITE, (center_x, center_y), 60, 2)
    
    # Draw pan indicator
    pygame.draw.circle(screen, YELLOW, (indicator_x, indicator_y), 8)
    
    # Draw center cross
    pygame.draw.line(screen, WHITE, (center_x - 10, center_y), (center_x + 10, center_y), 1)
    pygame.draw.line(screen, WHITE, (center_x, center_y - 10), (center_x, center_y + 10), 1)
    
    # Draw labels
    left_label = font.render("L", True, WHITE)
    right_label = font.render("R", True, WHITE)
    screen.blit(left_label, (center_x - 80, center_y - 10))
    screen.blit(right_label, (center_x + 70, center_y - 10))

def main():
    global smooth_x, smooth_y, smooth_pan_x, smooth_pan_y, current_sensitivity
    
    try:
        # Initialize MediaPipe with enhanced settings
        try:
            mp_face = mp.solutions.face_mesh
            face_mesh = mp_face.FaceMesh(
                static_image_mode=False, 
                max_num_faces=1,
                refine_landmarks=True,  # Enable refined landmarks
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            print("Enhanced MediaPipe face detection initialized successfully")
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            print("Falling back to mouse control mode")
            face_mesh = None

        # Initialize webcam with error handling
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam")
                cap = None
            else:
                print("Webcam initialized successfully")
        except Exception as e:
            print(f"Error initializing webcam: {e}")
            cap = None

        # Set up the display
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("HearSee - Extended Range Face Tracking")
        clock = pygame.time.Clock()

        # Virtual color image
        color_grid = generate_color_grid(GRID_ROWS, GRID_COLS)

        print("Starting HearSee (Extended Range Face Tracking)... Press ESC to quit")
        print("Sound volume has been reduced for safety")
        print("Increased movement range for better edge access!")

        # Font for text
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)

        # Default position (center)
        nose_x_norm, nose_y_norm = 0.5, 0.5
        face_angle_x, face_angle_y = 0.0, 0.0

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Try to get face position and angles from webcam
            if cap is not None and face_mesh is not None:
                try:
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(frame_rgb)
                        
                        if results.multi_face_landmarks:
                            landmarks = results.multi_face_landmarks[0].landmark
                            
                            # Calculate adaptive sensitivity based on face size
                            adaptive_sensitivity = calculate_adaptive_sensitivity(landmarks, frame.shape[1], frame.shape[0])
                            
                            # Get nose position for grid selection
                            nose_tip = landmarks[1]
                            nose_x_norm = nose_tip.x
                            nose_y_norm = nose_tip.y
                            
                            # Get face angles for panning
                            angle_x, angle_y = get_face_angles(landmarks, frame.shape[1], frame.shape[0])
                            
                            # Apply smoothing
                            smooth_x = smooth_value(smooth_x, nose_x_norm, SMOOTHING_FACTOR)
                            smooth_y = smooth_value(smooth_y, nose_y_norm, SMOOTHING_FACTOR)
                            smooth_pan_x = smooth_value(smooth_pan_x, angle_x, SMOOTHING_FACTOR)
                            smooth_pan_y = smooth_value(smooth_pan_y, angle_y, SMOOTHING_FACTOR)
                            
                            # Use smoothed values
                            nose_x_norm, nose_y_norm = smooth_x, smooth_y
                            face_angle_x, face_angle_y = smooth_pan_x, smooth_pan_y
                            
                except Exception as e:
                    print(f"Error processing face: {e}")
                    # Keep previous values

            # Map face position and angle to grid with extended range
            grid_x_norm, grid_y_norm = map_face_to_grid_position(nose_x_norm, nose_y_norm, face_angle_x, face_angle_y)
            
            # Convert to grid coordinates
            grid_x = int(grid_x_norm * GRID_COLS)
            grid_y = int(grid_y_norm * GRID_ROWS)
            grid_x = np.clip(grid_x, 0, GRID_COLS - 1)
            grid_y = np.clip(grid_y, 0, GRID_ROWS - 1)

            # Get color from virtual grid
            b, g, r = color_grid[grid_y, grid_x]

            # Clear screen
            screen.fill(BLACK)

            # Draw the grid
            for y in range(GRID_ROWS):
                for x in range(GRID_COLS):
                    color = color_grid[y, x].tolist()
                    rect = pygame.Rect(x * GRID_CELL_SIZE, y * GRID_CELL_SIZE, 
                                     GRID_CELL_SIZE, GRID_CELL_SIZE)
                    pygame.draw.rect(screen, color, rect)
                    pygame.draw.rect(screen, BLACK, rect, 1)  # Grid lines

            # Draw crosshair at selected cell
            center_x = grid_x * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
            center_y = grid_y * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
            pygame.draw.line(screen, WHITE, (center_x - 10, center_y), (center_x + 10, center_y), 2)
            pygame.draw.line(screen, WHITE, (center_x, center_y - 10), (center_x, center_y + 10), 2)

            # Display RGB text
            text = f"R:{r} G:{g} B:{b}"
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (10, 10))

            # Display face angle information
            angle_text = f"Pan: X:{face_angle_x*180/np.pi:.1f}° Y:{face_angle_y*180/np.pi:.1f}°"
            angle_surface = small_font.render(angle_text, True, WHITE)
            screen.blit(angle_surface, (10, 50))

            # Display adaptive sensitivity info
            sensitivity_text = f"Adaptive Sensitivity: {current_sensitivity:.1f}x"
            sensitivity_surface = small_font.render(sensitivity_text, True, WHITE)
            screen.blit(sensitivity_surface, (10, 80))

            # Display movement range info
            range_text = f"Movement Range: {MOVEMENT_MULTIPLIER:.1f}x"
            range_surface = small_font.render(range_text, True, WHITE)
            screen.blit(range_surface, (10, 110))

            # Add status text
            status_text = "Extended range face tracking active" if cap is not None and face_mesh is not None else "Mouse control mode"
            status_surface = small_font.render(status_text, True, WHITE)
            screen.blit(status_surface, (10, 140))

            # Draw pan indicator
            draw_pan_indicator(screen, face_angle_x, face_angle_y, small_font)

            # Add instructions
            instruction_text = "Extended range - can reach edges - ESC to quit"
            instruction_surface = small_font.render(instruction_text, True, WHITE)
            screen.blit(instruction_surface, (10, WINDOW_HEIGHT - 30))

            # Play the sound with enhanced spatial mixing
            try:
                wave = generate_spatial_mix(color_grid, grid_x, grid_y, face_angle_x, face_angle_y)
                sound_array = pygame.sndarray.make_sound(wave)
                sound_array.play()
            except Exception as e:
                print(f"Error playing sound: {e}")

            # Update display
            pygame.display.flip()
            clock.tick(30)  # 30 FPS

    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        if 'cap' in locals() and cap is not None:
            cap.release()
        pygame.quit()
        print("Program terminated safely")

if __name__ == "__main__":
    main() 