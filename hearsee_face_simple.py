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
SMOOTHING_FACTOR = 0.3  # Smoothing for face movement
MOVEMENT_SCALE = 5.0    # Simple movement scaling

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Smoothing variables
smooth_x, smooth_y = 0.5, 0.5
center_offset_x, center_offset_y = 0.0, 0.0  # Offset from calibrated center

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

    # Simple stereo panning based on face position
    left_vol = max(0, 1 - pan_x)
    right_vol = max(0, 1 + pan_x)
    
    # Normalize
    total_vol = left_vol + right_vol
    if total_vol > 0:
        left_vol /= total_vol
        right_vol /= total_vol

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

def get_face_position(landmarks):
    """Get simple face position from MediaPipe landmarks."""
    # Use nose tip for position
    nose_tip = landmarks[1]
    return nose_tip.x, nose_tip.y

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
    global smooth_x, smooth_y, center_offset_x, center_offset_y
    
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
        pygame.display.set_caption("HearSee - Simple Face Tracking")
        clock = pygame.time.Clock()

        # Virtual color image
        color_grid = generate_color_grid(GRID_ROWS, GRID_COLS)

        print("Starting HearSee (Simple Face Tracking)... Press ESC to quit")
        print("Sound volume has been reduced for safety")
        print("Simple, reliable face tracking with visible cursor!")

        # Font for text
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)

        # Default position (center)
        nose_x_norm, nose_y_norm = 0.5, 0.5

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Move cursor to center and reset tracking
                        nose_x_norm, nose_y_norm = 0.5, 0.5
                        smooth_x, smooth_y = 0.5, 0.5
                        # Reset center offset to current face position
                        if cap is not None and face_mesh is not None:
                            try:
                                ret, frame = cap.read()
                                if ret:
                                    frame = cv2.flip(frame, 1)
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    results = face_mesh.process(frame_rgb)
                                    if results.multi_face_landmarks:
                                        landmarks = results.multi_face_landmarks[0].landmark
                                        face_x, face_y = get_face_position(landmarks)
                                        center_offset_x = face_x - 0.5
                                        center_offset_y = face_y - 0.5
                                        print(f"Cursor moved to center. New center offset: ({center_offset_x:.2f}, {center_offset_y:.2f})")
                            except Exception as e:
                                print(f"Error during spacebar calibration: {e}")
                        else:
                            center_offset_x, center_offset_y = 0.0, 0.0
                            print("Cursor moved to center. Reset center offset.")

            # Try to get face position from webcam
            if cap is not None and face_mesh is not None:
                try:
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(frame_rgb)
                        
                        if results.multi_face_landmarks:
                            landmarks = results.multi_face_landmarks[0].landmark
                            
                            # Get simple face position
                            face_x, face_y = get_face_position(landmarks)
                            
                            # Apply center offset calibration
                            calibrated_x = face_x - center_offset_x
                            calibrated_y = face_y - center_offset_y
                            
                            # Apply movement scaling for better range
                            scaled_x = (calibrated_x - 0.5) * MOVEMENT_SCALE + 0.5
                            scaled_y = (calibrated_y - 0.5) * MOVEMENT_SCALE + 0.5
                            
                            # Apply smoothing
                            smooth_x = smooth_value(smooth_x, scaled_x, SMOOTHING_FACTOR)
                            smooth_y = smooth_value(smooth_y, scaled_y, SMOOTHING_FACTOR)
                            
                            # Use smoothed values
                            nose_x_norm, nose_y_norm = smooth_x, smooth_y
                            
                except Exception as e:
                    print(f"Error processing face: {e}")
                    # Keep previous values

            # Map position to grid with bounds checking
            grid_x = int(nose_x_norm * GRID_COLS)
            grid_y = int(nose_y_norm * GRID_ROWS)
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

            # Draw crosshair at selected cell (always visible)
            center_x = grid_x * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
            center_y = grid_y * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
            
            # Ensure crosshair is within screen bounds
            center_x = np.clip(center_x, 15, WINDOW_WIDTH - 15)
            center_y = np.clip(center_y, 15, WINDOW_HEIGHT - 15)
            
            # Draw crosshair with better visibility
            pygame.draw.line(screen, WHITE, (center_x - 20, center_y), (center_x + 20, center_y), 4)
            pygame.draw.line(screen, WHITE, (center_x, center_y - 20), (center_x, center_y + 20), 4)
            
            # Add a circle around the crosshair for better visibility
            pygame.draw.circle(screen, BLACK, (center_x, center_y), 12, 3)

            # Display RGB text
            text = f"R:{r} G:{g} B:{b}"
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (10, 10))

            # Display face position info
            pos_text = f"Face: ({nose_x_norm:.2f}, {nose_y_norm:.2f})"
            pos_surface = small_font.render(pos_text, True, WHITE)
            screen.blit(pos_surface, (10, 50))

            # Display cursor position info
            cursor_text = f"Cursor: ({grid_x}, {grid_y})"
            cursor_surface = small_font.render(cursor_text, True, WHITE)
            screen.blit(cursor_surface, (10, 80))

            # Display movement scale info
            scale_text = f"Movement Scale: {MOVEMENT_SCALE:.1f}x"
            scale_surface = small_font.render(scale_text, True, WHITE)
            screen.blit(scale_surface, (10, 110))

            # Display calibration info
            cal_text = f"Center Offset: ({center_offset_x:.2f}, {center_offset_y:.2f})"
            cal_surface = small_font.render(cal_text, True, WHITE)
            screen.blit(cal_surface, (10, 140))

            # Add status text
            status_text = "Simple face tracking active" if cap is not None and face_mesh is not None else "Mouse control mode"
            status_surface = small_font.render(status_text, True, WHITE)
            screen.blit(status_surface, (10, 170))

            # Draw pan indicator
            pan_x = (nose_x_norm - 0.5) * 2  # Convert to -1 to 1 range
            pan_y = (nose_y_norm - 0.5) * 2
            draw_pan_indicator(screen, pan_x, pan_y, small_font)

            # Add instructions
            instruction_text = "SPACE to move cursor to center - ESC to quit"
            instruction_surface = small_font.render(instruction_text, True, WHITE)
            screen.blit(instruction_surface, (10, WINDOW_HEIGHT - 30))

            # Play the sound with enhanced spatial mixing
            try:
                wave = generate_spatial_mix(color_grid, grid_x, grid_y, pan_x, pan_y)
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