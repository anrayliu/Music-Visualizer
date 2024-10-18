import librosa
import numpy as np
import pygame
import pygame.gfxdraw
import sys


WIN_X = 1000
WIN_Y = 800
SONG = "song.wav"
HOP_LENGTH = 512
N_FFT = 2048 * 4
HERTZ_RANGE = (100, 6000)
STEP = 100
BAR_WIDTH = (WIN_X * 0.75) // len(np.arange(*HERTZ_RANGE, STEP))
BAR_HEIGHT = 300
LAYERS = 10
WHITE = (0, 0, 0)


class Main:
    def __init__(self):
        pygame.init()

        self.win = pygame.display.set_mode((WIN_X, WIN_Y))
        pygame.display.set_caption(SONG)

    def load(self):
        time_series, sample_rate = librosa.load(SONG)
        stft = np.abs(librosa.stft(time_series, hop_length=HOP_LENGTH, n_fft=N_FFT))
        self.spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
        frequencies = librosa.core.fft_frequencies(n_fft=N_FFT)
        times = librosa.core.frames_to_time(np.arange(self.spectrogram.shape[1]), sr=sample_rate, hop_length=HOP_LENGTH, n_fft=N_FFT)
        self.time_index_ratio = len(times) / times[len(times) - 1]
        self.frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]
        self.bars = []
        for i, f in enumerate(np.arange(*HERTZ_RANGE, STEP)):
            self.bars.append(AudioBar(i * BAR_WIDTH, BAR_HEIGHT, f, max_height=BAR_HEIGHT))

    def get_decibel(self, target_time, freq):
        return self.spectrogram[int(freq * self.frequencies_index_ratio)][int(target_time * self.time_index_ratio)]
    
    def play(self):
        pygame.mixer.music.load(SONG)
        pygame.mixer.music.play(-1)

    def update(self, dt):
        points = []

        for b in self.bars:
            try:
                points.append(b.update(dt, self.get_decibel(pygame.mixer.music.get_pos() / 1000, b.freq)))
            except IndexError:
                self.play()

        self.draw(points)

    def draw(self, points):
        c = 0
        for i in range(LAYERS):
            if i % 2 == 0:
                vals = [(v[0], v[1] * (1 + i * 0.3)) for v in points]
            else:
                vals = [(WIN_X - v[0], v[1] * (1 + i * 0.3)) for v in reversed(points)]

            pygame.gfxdraw.filled_polygon(self.win, (vals[0], vals[-1], (WIN_X, WIN_Y), (0, WIN_Y)), (10 * c, 0, 0))
            c += 1
            pygame.gfxdraw.filled_polygon(self.win, vals, (10 * c, 0, 0))
            c += 1

    def run(self):
        clock = pygame.time.Clock()

        self.load()
        self.play()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            dt = clock.tick(60) / 1000.0

            self.win.fill(WHITE)
            self.update(dt)
            pygame.display.update()


class AudioBar:
    def __init__(self, x, y, freq, min_height=10, max_height=100, min_decibel=-60, max_decibel=0):
        self.x = x 
        self.y = y 
        self.freq = freq

        self.width = BAR_WIDTH
        self.height = self.min_height = min_height
        self.max_height = max_height
        
        self.decibel_height_ratio = (self.max_height - self.min_height) / (max_decibel - min_decibel)

    def update(self, dt, decibel):
        desired_height = decibel * self.decibel_height_ratio + self.max_height

        self.height += (desired_height - self.height) / 0.1 * dt
        self.height = min(self.max_height, max(self.min_height, self.height))

        return self.x, self.y - self.height - 100
    

Main().run()
