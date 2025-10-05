import numpy as np
import pygame
from colors import COLOR_ACTIVE, COLOR_PASSIVE, COLOR_SORTED
import time

# Sound effect
def play_tone(position, max_value, duration=0.05):
    freq = 200 + int((position / max_value) * 1800)
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(freq * t * 2 * np.pi)
    volume = 0.2  # Scale from 0.0 (silent) to 1.0 (max)
    audio = (wave * 32767 * volume).astype(np.int16).tobytes()
    sound = pygame.mixer.Sound(buffer=audio)
    sound.play()

# Completion sweep animation
def completion_sweep(data, draw_data, speed):
    for i in range(len(data)):
        color_array = []
        for x in range(len(data)):
            if x < i:
                color_array.append(COLOR_SORTED)
            elif x == i:
                color_array.append(COLOR_ACTIVE)
            else:
                color_array.append(COLOR_PASSIVE)
        draw_data(data, color_array)
        play_tone(i, len(data))
        time.sleep(speed / 2)
    draw_data(data, [COLOR_SORTED for _ in range(len(data))])