import pygame
import numpy as np
import sys

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

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Generate a color grid (could be random or patterned)
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

def generate_spatial_mix(grid, cx, cy):
    tones = []

    def safe_color(x, y):
        if 0 <= x < GRID_COLS and 0 <= y < GRID_ROWS:
            return grid[y, x]
        return (0, 0, 0)

    neighbors = [
        ("center", cx, cy, 0.5, 0.5),      # center: balanced
        ("left", cx - 1, cy, 1.0, 0.0),    # hard left
        ("right", cx + 1, cy, 0.0, 1.0),   # hard right
        ("above", cx, cy - 1, 0.6, 0.6),   # slightly above (slightly higher pitch)
        ("below", cx, cy + 1, 0.4, 0.4),   # slightly below (lower pitch)
    ]

    for label, x, y, lv, rv in neighbors:
        b, g, r = safe_color(x, y)
        vol_r = r / 255.0
        vol_g = g / 255.0
        vol_b = b / 255.0

        # Slight pitch shift to simulate elevation
        shift = 1.1 if label == "above" else 0.9 if label == "below" else 1.0

        t_r = generate_tone(FREQ_RED * shift, vol_r)
        t_g = generate_tone(FREQ_GREEN * shift, vol_g)
        t_b = generate_tone(FREQ_BLUE * shift, vol_b)
        mixed = t_r + t_g + t_b
        mixed = np.clip(mixed, -32768, 32767)
        stereo = pan_stereo(mixed, lv, rv)
        tones.append(stereo)

    final = np.sum(tones, axis=0)
    final = np.clip(final, -32768, 32767)
    return final.astype(np.int16)

def main():
    try:
        # Set up the display
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("HearSee - Virtual Color Grid")
        clock = pygame.time.Clock()

        # Virtual color image
        color_grid = generate_color_grid(GRID_ROWS, GRID_COLS)

        print("Starting HearSee (Mouse Control Mode)... Press ESC to quit")
        print("Sound volume has been reduced for safety")
        print("Move your mouse over the grid to hear colors!")

        # Font for text
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Get mouse position
            mouse_x, mouse_y = pygame.mouse.get_pos()

            # Map mouse position to grid
            grid_x = int(mouse_x / GRID_CELL_SIZE)
            grid_y = int(mouse_y / GRID_CELL_SIZE)
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

            # Add instructions
            instruction_text = "Move mouse over grid - ESC to quit"
            instruction_surface = small_font.render(instruction_text, True, WHITE)
            screen.blit(instruction_surface, (10, WINDOW_HEIGHT - 30))

            # Play the sound
            try:
                wave = generate_spatial_mix(color_grid, grid_x, grid_y)
                # Convert to pygame sound format
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
        pygame.quit()
        print("Program terminated safely")

if __name__ == "__main__":
    main()
