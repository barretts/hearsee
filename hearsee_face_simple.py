import pygame
import numpy as np
import sys
import cv2
import mediapipe as mp
import math

# Initialize pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# Sound settings - Human color perception based frequencies
FREQ_BASE = 200  # Base frequency
# Make frequencies much more distinct and audible
FREQ_RED = 400    # Red frequency - mid range
FREQ_GREEN = 600  # Green frequency (most sensitive) - higher
FREQ_BLUE = 200   # Blue frequency (least sensitive) - much lower bass
SAMPLE_RATE = 44100
DURATION = 0.2
MASTER_VOLUME = 0.02  # Very quiet for safety

# Grid settings
GRID_ROWS, GRID_COLS = 16, 16  # 16x16 grid for 256x256 image
GRID_CELL_SIZE = 40  # Smaller cells to fit 256x256 image
WINDOW_WIDTH = GRID_COLS * GRID_CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * GRID_CELL_SIZE

# Face tracking settings
SMOOTHING_FACTOR = 0.3  # Smoothing for face movement
MOVEMENT_SCALE = 5.0    # Simple movement scaling
BLENDING_THRESHOLD = 1.0  # Audio blending sensitivity (higher = more dramatic changes)
SPATIAL_BLEND_RADIUS = 2.0  # How much surrounding colors influence audio (like eye focus)
BLENDING_ENABLED = False  # Toggle for all spatial blending - DEFAULT OFF

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
SKY_BLUE = (135, 206, 235)
GRASS_GREEN = (34, 139, 34)
GROUND_BROWN = (139, 69, 19)
SUN_YELLOW = (255, 255, 0)
HOUSE_BROWN = (160, 82, 45)
WINDOW_BLUE = (100, 149, 237)  # Distinct window blue (not sky blue)
CHIMNEY_BROWN = (101, 67, 33)  # Dark brown for chimney
DOOR_BLACK = (50, 50, 50)  # Dark gray for door

# Smoothing variables
smooth_x, smooth_y = 0.5, 0.5
center_offset_x, center_offset_y = 0.0, 0.0  # Offset from calibrated center

def generate_child_landscape():
    """Generate a 256x256 child landscape image."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Sky (top 60% of image)
    img[0:154, :] = SKY_BLUE
    
    # Sun (top right area)
    sun_center = (200, 60)
    sun_radius = 25
    for y in range(max(0, sun_center[1] - sun_radius), min(256, sun_center[1] + sun_radius)):
        for x in range(max(0, sun_center[0] - sun_radius), min(256, sun_center[0] + sun_radius)):
            if (x - sun_center[0])**2 + (y - sun_center[1])**2 <= sun_radius**2:
                img[y, x] = SUN_YELLOW
    
    # Grass (middle 30% of image)
    img[154:230, :] = GRASS_GREEN
    
    # Ground (bottom 10% of image)
    img[230:256, :] = GROUND_BROWN
    
    # Stick house (simple geometric shapes)
    # House base
    house_x, house_y = 128, 180
    house_width, house_height = 60, 40
    
    # House body (brown rectangle)
    img[house_y:house_y+house_height, house_x-house_width//2:house_x+house_width//2] = HOUSE_BROWN
    
    # Roof (triangle)
    roof_points = np.array([
        [house_x, house_y-20],
        [house_x-house_width//2, house_y],
        [house_x+house_width//2, house_y]
    ], dtype=np.int32)
    
    # Draw roof triangle
    for y in range(house_y-20, house_y):
        for x in range(house_x-house_width//2, house_x+house_width//2):
            if (x - house_x)**2 + (y - house_y)**2 <= (house_width//2)**2:
                if y >= house_y - 20:
                    img[y, x] = HOUSE_BROWN
    
    # Door (dark gray)
    door_x, door_y = house_x, house_y + 10
    door_width, door_height = 12, 25
    img[door_y:door_y+door_height, door_x-door_width//2:door_x+door_width//2] = DOOR_BLACK
    
    # Chimney (distinct dark brown)
    chimney_x, chimney_y = house_x + 15, house_y - 35
    img[chimney_y:chimney_y+15, chimney_x:chimney_x+8] = CHIMNEY_BROWN
    
    # Smoke (light gray circles)
    smoke_center = (chimney_x + 4, chimney_y - 5)
    for i in range(3):
        smoke_y = smoke_center[1] - i * 8
        smoke_x = smoke_center[0] + (i % 2) * 4
        for y in range(max(0, smoke_y-3), min(256, smoke_y+3)):
            for x in range(max(0, smoke_x-3), min(256, smoke_x+3)):
                if (x - smoke_x)**2 + (y - smoke_y)**2 <= 9:
                    img[y, x] = (200, 200, 200)
    
    # Add some clouds (white circles) - but avoid sun area
    cloud_positions = [(80, 40), (60, 80)]  # Removed cloud near sun
    for cloud_x, cloud_y in cloud_positions:
        # Check if cloud overlaps with sun area
        sun_distance = ((cloud_x - sun_center[0])**2 + (cloud_y - sun_center[1])**2)**0.5
        if sun_distance > sun_radius + 25:  # Only place clouds far from sun
            for y in range(max(0, cloud_y-15), min(256, cloud_y+15)):
                for x in range(max(0, cloud_x-20), min(256, cloud_x+20)):
                    if (x - cloud_x)**2 + (y - cloud_y)**2 <= 225:
                        img[y, x] = WHITE
    
    # Add some grass details (darker green lines)
    for i in range(0, 256, 20):
        img[230:240, i:i+2] = (0, 100, 0)
    
    return img

def get_pixel_color(img, x, y):
    """Get color from image at normalized coordinates."""
    if 0 <= x < 1 and 0 <= y < 1:
        pixel_x = int(x * 255)
        pixel_y = int(y * 255)
        return img[pixel_y, pixel_x]
    return (0, 0, 0)

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

def color_to_frequency_signature(r, g, b):
    """Convert RGB color to distinct frequency signature based on human color perception."""
    # Normalize RGB values
    r_norm = r / 255.0
    g_norm = g / 255.0  
    b_norm = b / 255.0
    
    # Create VERY distinct frequency signatures for major colors
    # Sky blue vs Sun yellow should be extremely different
    
    # Determine color type and create signature
    if r_norm > 0.8 and g_norm > 0.8 and b_norm < 0.3:  # Yellow (sun)
        # High frequency for bright yellow sun
        primary_freq = 800  # High bright tone
        primary_vol = 1.0
        return [(primary_freq, primary_vol)]
        
    elif b_norm > 0.5 and r_norm < 0.6 and g_norm < 0.8:  # Blue (sky)
        # Low frequency for blue sky
        primary_freq = 150  # Deep bass tone
        primary_vol = 1.0
        return [(primary_freq, primary_vol)]
        
    elif g_norm > 0.5 and r_norm < 0.4 and b_norm < 0.4:  # Green (grass)
        # Mid-high frequency for green
        primary_freq = 500  # Clear mid tone
        primary_vol = 1.0
        return [(primary_freq, primary_vol)]
        
    elif r_norm > 0.4 and g_norm > 0.2 and b_norm < 0.3:  # Brown (house/ground)
        # Mid frequency for brown
        primary_freq = 300  # Warm mid tone
        primary_vol = 1.0
        return [(primary_freq, primary_vol)]
        
    elif r_norm < 0.3 and g_norm < 0.3 and b_norm < 0.3:  # Dark colors (door, chimney)
        # Low-mid frequency for dark colors
        primary_freq = 250  # Dark tone
        primary_vol = 1.0
        return [(primary_freq, primary_vol)]
        
    else:  # Other colors - use RGB dominance
        if r_norm > g_norm and r_norm > b_norm:  # Red dominant
            primary_freq = FREQ_RED + (r_norm * 200)  # 400-600 Hz
        elif g_norm > r_norm and g_norm > b_norm:  # Green dominant  
            primary_freq = FREQ_GREEN + (g_norm * 200)   # 600-800 Hz
        else:  # Blue dominant
            primary_freq = FREQ_BLUE + (b_norm * 100)    # 200-300 Hz
        
        primary_vol = 1.0
        return [(primary_freq, primary_vol)]

def generate_color_tone(r, g, b, volume_scale=1.0):
    """Generate audio tone for a specific RGB color."""
    freq_signature = color_to_frequency_signature(r, g, b)
    
    # Generate combined tone from frequency signature
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    combined_tone = np.zeros_like(t)
    
    for freq, vol in freq_signature:
        if freq > 0:  # Only generate if frequency is valid
            tone = np.sin(freq * t * 2 * np.pi) * vol * volume_scale
            combined_tone += tone
    
    # Normalize and convert to audio format
    if np.max(np.abs(combined_tone)) > 0:
        combined_tone = combined_tone / np.max(np.abs(combined_tone))
    
    audio = (combined_tone * MASTER_VOLUME * 32767).astype(np.int16)
    return audio

def pan_stereo(tone, left_vol, right_vol):
    """Convert mono tone to stereo with panning."""
    stereo = np.zeros((len(tone), 2), dtype=np.int16)
    stereo[:, 0] = (tone * left_vol).astype(np.int16)
    stereo[:, 1] = (tone * right_vol).astype(np.int16)
    return stereo

def generate_spatial_mix(landscape_img, nose_x_norm, nose_y_norm, pan_x=0, pan_y=0):
    """Generate spatial audio mix with proper color distinction and spatial blending."""
    
    def get_landscape_color(x_norm, y_norm):
        """Get color from landscape image at normalized coordinates, return None if out of bounds."""
        if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
            pixel_x = int(x_norm * 255)
            pixel_y = int(y_norm * 255)
            pixel_x = np.clip(pixel_x, 0, 255)
            pixel_y = np.clip(pixel_y, 0, 255)
            return landscape_img[pixel_y, pixel_x]
        return None  # Out of bounds

    # Get center color - this is the PRIMARY sound
    center_color = get_landscape_color(nose_x_norm, nose_y_norm)
    if center_color is None:
        return np.zeros((int(SAMPLE_RATE * DURATION), 2), dtype=np.int16)
    
    center_b, center_g, center_r = center_color
    
    # If blending is disabled, return pure center color
    if not BLENDING_ENABLED or SPATIAL_BLEND_RADIUS <= 0:
        center_tone = generate_color_tone(center_r, center_g, center_b, 1.0)
        left_vol = max(0, 1 - pan_x)
        right_vol = max(0, 1 + pan_x)
        total_vol = left_vol + right_vol
        if total_vol > 0:
            left_vol /= total_vol
            right_vol /= total_vol
        return pan_stereo(center_tone, left_vol, right_vol)
    
    # Calculate center vs surrounding balance based on blending threshold
    # Blending threshold: 0.1 = 90% center, 10% surrounding
    # Blending threshold: 1.0 = 50% center, 50% surrounding  
    # Blending threshold: 5.0 = 20% center, 80% surrounding
    center_weight = 1.0 / (1.0 + BLENDING_THRESHOLD)
    surrounding_weight = BLENDING_THRESHOLD / (1.0 + BLENDING_THRESHOLD)
    
    # Generate center tone
    center_tone = generate_color_tone(center_r, center_g, center_b, center_weight)
    
    # Collect surrounding colors within radius
    pixel_size = 1.0 / 256.0  # Size of one pixel in normalized coordinates
    radius_pixels = SPATIAL_BLEND_RADIUS * (GRID_COLS / 256.0)  # Convert to normalized space
    
    surrounding_tones = []
    surrounding_colors = []
    
    # Sample surrounding pixels
    for dy in range(-int(SPATIAL_BLEND_RADIUS * 2), int(SPATIAL_BLEND_RADIUS * 2) + 1):
        for dx in range(-int(SPATIAL_BLEND_RADIUS * 2), int(SPATIAL_BLEND_RADIUS * 2) + 1):
            if dx == 0 and dy == 0:
                continue  # Skip center
            
            # Convert grid offset to normalized coordinates
            offset_x = dx * pixel_size * 4  # Scale for visibility
            offset_y = dy * pixel_size * 4
            distance = (offset_x**2 + offset_y**2)**0.5
            
            if distance <= radius_pixels:
                sample_x = nose_x_norm + offset_x
                sample_y = nose_y_norm + offset_y
                
                surround_color = get_landscape_color(sample_x, sample_y)
                if surround_color is not None:  # Only include valid colors (not edges)
                    surround_b, surround_g, surround_r = surround_color
                    
                    # Skip if same as center color
                    if (surround_r, surround_g, surround_b) != (center_r, center_g, center_b):
                        # Weight by distance
                        weight = max(0, 1.0 - distance / radius_pixels)
                        surrounding_colors.append((surround_r, surround_g, surround_b, weight, offset_x, offset_y))
    
    # Generate surrounding tones
    if surrounding_colors:
        # Normalize weights
        total_surrounding_weight = sum(color[3] for color in surrounding_colors)
        if total_surrounding_weight > 0:
            for surround_r, surround_g, surround_b, weight, offset_x, offset_y in surrounding_colors:
                normalized_weight = (weight / total_surrounding_weight) * surrounding_weight
                surround_tone = generate_color_tone(surround_r, surround_g, surround_b, normalized_weight)
                
                # Pan based on direction
                pan_offset = offset_x / radius_pixels * 0.5  # Directional panning
                surrounding_tones.append((surround_tone, pan_offset))
    
    # Mix center and surrounding tones
    # Center tone panning
    left_vol = max(0, 1 - pan_x)
    right_vol = max(0, 1 + pan_x)
    total_vol = left_vol + right_vol
    if total_vol > 0:
        left_vol /= total_vol
        right_vol /= total_vol
    
    # Start with center tone
    final_stereo = pan_stereo(center_tone, left_vol, right_vol)
    
    # Add surrounding tones
    for surround_tone, pan_offset in surrounding_tones:
        surround_left = max(0, left_vol - pan_offset)
        surround_right = max(0, right_vol + pan_offset)
        surround_total = surround_left + surround_right
        if surround_total > 0:
            surround_left /= surround_total
            surround_right /= surround_total
        
        surround_stereo = pan_stereo(surround_tone, surround_left, surround_right)
        final_stereo = final_stereo + surround_stereo
    
    # Clip and return
    final_stereo = np.clip(final_stereo, -32768, 32767)
    return final_stereo.astype(np.int16)

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
    global smooth_x, smooth_y, center_offset_x, center_offset_y, BLENDING_THRESHOLD, SPATIAL_BLEND_RADIUS, BLENDING_ENABLED
    
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
        landscape_image = generate_child_landscape()

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
                    elif event.key == pygame.K_UP:
                        # Increase blending threshold (more dramatic changes)
                        BLENDING_THRESHOLD = min(BLENDING_THRESHOLD + 0.2, 5.0)
                        print(f"Blending threshold increased to: {BLENDING_THRESHOLD:.1f}")
                    elif event.key == pygame.K_DOWN:
                        # Decrease blending threshold (softer changes)
                        BLENDING_THRESHOLD = max(BLENDING_THRESHOLD - 0.2, 0.1)
                        print(f"Blending threshold decreased to: {BLENDING_THRESHOLD:.1f}")
                    elif event.key == pygame.K_LEFT:
                        # Decrease spatial blend radius (sharper focus)
                        SPATIAL_BLEND_RADIUS = max(SPATIAL_BLEND_RADIUS - 0.5, 0.0)
                        print(f"Spatial blend radius decreased to: {SPATIAL_BLEND_RADIUS:.1f}")
                    elif event.key == pygame.K_RIGHT:
                        # Increase spatial blend radius (wider focus)
                        SPATIAL_BLEND_RADIUS = min(SPATIAL_BLEND_RADIUS + 0.5, 5.0)
                        print(f"Spatial blend radius increased to: {SPATIAL_BLEND_RADIUS:.1f}")
                    elif event.key == pygame.K_TAB: # Toggle blending
                        BLENDING_ENABLED = not BLENDING_ENABLED
                        print(f"Blending {'enabled' if BLENDING_ENABLED else 'disabled'}")

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

            # Get color from landscape image directly (not from grid)
            pixel_color = get_pixel_color(landscape_image, nose_x_norm, nose_y_norm)
            b, g, r = pixel_color
            
            # Debug: Print color info when it changes significantly
            if abs(r - getattr(main, 'last_r', 0)) > 10 or abs(g - getattr(main, 'last_g', 0)) > 10 or abs(b - getattr(main, 'last_b', 0)) > 10:
                print(f"Color changed: RGB({r}, {g}, {b}) at position ({nose_x_norm:.3f}, {nose_y_norm:.3f})")
                main.last_r, main.last_g, main.last_b = r, g, b

            # Clear screen
            screen.fill(BLACK)

            # Draw the landscape as a grid
            for y in range(GRID_ROWS):
                for x in range(GRID_COLS):
                    # Get color from landscape image for this grid cell
                    cell_x = x / GRID_COLS
                    cell_y = y / GRID_ROWS
                    cell_color = get_pixel_color(landscape_image, cell_x, cell_y)
                    
                    rect = pygame.Rect(x * GRID_CELL_SIZE, y * GRID_CELL_SIZE, 
                                     GRID_CELL_SIZE, GRID_CELL_SIZE)
                    pygame.draw.rect(screen, cell_color, rect)
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

            # Display adjacent colors when blending is enabled
            if BLENDING_ENABLED and SPATIAL_BLEND_RADIUS > 0:
                # Sample adjacent colors
                adjacent_colors = []
                pixel_size = 1.0 / 256.0
                radius_pixels = SPATIAL_BLEND_RADIUS * (GRID_COLS / 256.0)
                
                # Sample 8 adjacent directions
                directions = [
                    (-1, 0, "L"), (1, 0, "R"), (0, -1, "T"), (0, 1, "B"),
                    (-1, -1, "TL"), (1, -1, "TR"), (-1, 1, "BL"), (1, 1, "BR")
                ]
                
                for dx, dy, label in directions:
                    offset_x = dx * pixel_size * 4
                    offset_y = dy * pixel_size * 4
                    sample_x = nose_x_norm + offset_x
                    sample_y = nose_y_norm + offset_y
                    
                    if 0 <= sample_x <= 1 and 0 <= sample_y <= 1:
                        adj_color = get_pixel_color(landscape_image, sample_x, sample_y)
                        # Fix numpy array comparison
                        if adj_color is not None and tuple(adj_color) != (r, g, b):  # Only show if different from center
                            adj_r, adj_g, adj_b = adj_color
                            adjacent_colors.append(f"{label}:({adj_r},{adj_g},{adj_b})")
                
                if adjacent_colors:
                    # Show up to 4 adjacent colors
                    adj_text = "Adj: " + " ".join(adjacent_colors[:4])
                    adj_surface = small_font.render(adj_text, True, WHITE)
                    screen.blit(adj_surface, (10, 40))

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

            # Display blending threshold info
            blend_text = f"Blending: {BLENDING_THRESHOLD:.1f} (UP/DOWN to adjust)"
            blend_surface = small_font.render(blend_text, True, WHITE)
            screen.blit(blend_surface, (10, 170))

            # Display spatial blend radius info
            blend_radius_text = f"Spatial Blend Radius: {SPATIAL_BLEND_RADIUS:.1f} (LEFT/RIGHT to adjust)"
            blend_radius_surface = small_font.render(blend_radius_text, True, WHITE)
            screen.blit(blend_radius_surface, (10, 200))

            # Display blending toggle info
            blend_toggle_text = f"Blending: {'ON' if BLENDING_ENABLED else 'OFF'}"
            blend_toggle_surface = small_font.render(blend_toggle_text, True, WHITE)
            screen.blit(blend_toggle_surface, (10, 230))

            # Add status text
            status_text = "Simple face tracking active" if cap is not None and face_mesh is not None else "Mouse control mode"
            status_surface = small_font.render(status_text, True, WHITE)
            screen.blit(status_surface, (10, 260))

            # Draw pan indicator
            pan_x = (nose_x_norm - 0.5) * 2  # Convert to -1 to 1 range
            pan_y = (nose_y_norm - 0.5) * 2
            draw_pan_indicator(screen, pan_x, pan_y, small_font)

            # Add instructions
            instruction_text = "SPACE to center - UP/DOWN blending - LEFT/RIGHT focus - TAB toggle - ESC quit"
            instruction_surface = small_font.render(instruction_text, True, WHITE)
            screen.blit(instruction_surface, (10, WINDOW_HEIGHT - 30))

            # Play the sound with enhanced spatial mixing
            try:
                wave = generate_spatial_mix(landscape_image, nose_x_norm, nose_y_norm, pan_x, pan_y)
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