import os
import pygame
import random
import math
import struct
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

# ======================================================
# NOKIA 3310 SNAKE VS COCKROACHES - DQN VISUAL VERSION
# ======================================================
# Install:
#   pip install pygame torch
# Run:
#   python worms_dqn_pro_level_v2.py
#
# Controls:
#   S = save neural network model
#   L = load neural network model
#   V = view / hide neural-network side panel
#   M = music on / off
#   N = all sound on / off
#   F = toggle fullscreen / resizable window
#   H = AI / human override
#   T = cycle speed
#   ESC = quit
#
# DQN idea:
#   The neural network receives the game state and outputs Q-values:
#   Q(UP), Q(DOWN), Q(LEFT), Q(RIGHT), Q(SHOOT)
#   The action with the highest Q is what the AI currently believes is best.

# -------------------------
# Audio setup
# -------------------------
SOUND_AVAILABLE = True
try:
    pygame.mixer.pre_init(22050, -16, 1, 512)
except Exception:
    SOUND_AVAILABLE = False

pygame.init()
try:
    pygame.mixer.init(22050, -16, 1, 512)
except Exception:
    SOUND_AVAILABLE = False

# -------------------------
# Display constants
# -------------------------
CELL = 16
COLS, ROWS = 40, 24
GAME_W, GAME_H = COLS * CELL, ROWS * CELL
PANEL_W = 360
BASE_W, BASE_H = GAME_W + PANEL_W, 720
BOTTOM_Y = GAME_H
BOTTOM_H = BASE_H - GAME_H
WIDTH, HEIGHT = BASE_W, BASE_H

# Render to a fixed virtual Nokia screen, then scale to the real display.
# This keeps everything fitting in fullscreen, minimized/restored, and resized windows.
fullscreen = False
display_surface = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
screen = pygame.Surface((BASE_W, BASE_H))
pygame.display.set_caption("Nokia 3310 Snake vs Cockroaches DQN - Responsive Fullscreen Brain")

clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18, bold=True)
small_font = pygame.font.SysFont("consolas", 14)
tiny_font = pygame.font.SysFont("consolas", 11)

# Nokia-like palette
BG = (155, 188, 15)
BG_2 = (139, 172, 15)
DARK = (15, 56, 15)
MID = (48, 98, 48)
LIGHT = (190, 215, 50)

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "SHOOT"]
N_ACTIONS = len(ACTIONS)

# -------------------------
# DQN hyperparameters
# -------------------------
STATE_SIZE = 15
GAMMA = 0.92
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 8_000
MIN_MEMORY_TO_TRAIN = 400
TARGET_UPDATE_EVERY = 1200

EPSILON_START = 0.90
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
DEVICE_LABEL = (torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")[:18] if DEVICE.type == "cuda" else "CPU")


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

    def activations(self, x):
        with torch.no_grad():
            x = torch.tensor([x], dtype=torch.float32, device=DEVICE)
            h1 = self.relu(self.fc1(x))[0].detach().cpu().tolist()
            h2 = self.relu(self.fc2(torch.tensor([h1], dtype=torch.float32, device=DEVICE)))[0].detach().cpu().tolist()
            qv = self.out(torch.tensor([h2], dtype=torch.float32, device=DEVICE))[0].detach().cpu().tolist()
        return h1, h2, qv


@dataclass
class Experience:
    state: list
    action: int
    reward: float
    next_state: list
    done: bool


policy_net = DQN(STATE_SIZE, N_ACTIONS).to(DEVICE)
target_net = DQN(STATE_SIZE, N_ACTIONS).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.SmoothL1Loss()
memory = deque(maxlen=MEMORY_SIZE)

# Learning stats
epsilon = EPSILON_START
train_steps = 0
last_loss = 0.0
last_q_values = [0.0] * N_ACTIONS
q_history = deque(maxlen=90)
reward_history = deque(maxlen=90)
action_history = deque(maxlen=12)
last_reward_value = 0.0
total_steps = 0

# Visual / audio toggles
show_brain_panel = True
sound_on = True
music_on = True
human_mode = False
speed_levels = [8, 14, 24, 32, 40]
TRAIN_EVERY = 8  # base train interval; dynamic interval below protects speed 40
MAX_ROACHES = 8
MAX_BULLETS = 10
MAX_PARTICLES = 45
MAX_STEPS_PER_EPISODE = 1200
LAST_ERROR = ""
speed_index = 0

# -------------------------
# Retro sounds
# -------------------------
def make_square_sound(freq=440, duration=0.08, volume=0.25, sample_rate=22050):
    if not SOUND_AVAILABLE:
        return None
    n_samples = int(sample_rate * duration)
    amplitude = int(32767 * volume)
    period = max(1, int(sample_rate / max(1, freq)))
    buf = bytearray()
    for i in range(n_samples):
        value = amplitude if (i % period) < period // 2 else -amplitude
        buf += struct.pack("<h", value)
    try:
        return pygame.mixer.Sound(buffer=bytes(buf))
    except Exception:
        return None

SFX_MOVE = make_square_sound(180, 0.025, 0.08)
SFX_SHOOT = make_square_sound(760, 0.055, 0.16)
SFX_EAT = make_square_sound(1040, 0.080, 0.18)
SFX_HIT = make_square_sound(90, 0.130, 0.22)
SFX_GAMEOVER = make_square_sound(55, 0.350, 0.22)
MUSIC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "djartmusic-fun-with-my-8-bit-game-301278.mp3")
music_loaded = False
try:
    if SOUND_AVAILABLE and os.path.exists(MUSIC_FILE):
        pygame.mixer.music.load(MUSIC_FILE)
        pygame.mixer.music.set_volume(0.32)
        pygame.mixer.music.play(-1)
        music_loaded = True
except Exception:
    music_loaded = False

# Fallback chip melody if the mp3 is missing.
MELODY_FREQS = [262, 330, 392, 523, 392, 330, 294, 349, 440, 392, 330, 262]
MELODY_SOUNDS = [make_square_sound(f, 0.09, 0.08) for f in MELODY_FREQS]
melody_index = 0
melody_timer = 0


def play(sound):
    if sound_on and SOUND_AVAILABLE and sound is not None:
        try:
            sound.play()
        except Exception:
            pass


def apply_music_state():
    if not SOUND_AVAILABLE or not music_loaded:
        return
    try:
        if sound_on and music_on:
            pygame.mixer.music.unpause()
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
        else:
            pygame.mixer.music.pause()
    except Exception:
        pass


def update_music():
    global melody_index, melody_timer
    if music_loaded:
        apply_music_state()
        return
    if not (sound_on and music_on and SOUND_AVAILABLE):
        return
    melody_timer += 1
    if melody_timer >= 8:
        melody_timer = 0
        snd = MELODY_SOUNDS[melody_index % len(MELODY_SOUNDS)]
        play(snd)
        melody_index += 1

# -------------------------
# Game state
# -------------------------
snake = []
direction = (1, 0)
bullets = []
roaches = []
particles = []

score = 0
lives = 3
slurpee_cups = 0
reward_total = 0.0
level = 1
combo = 0
shoot_cooldown = 0
direction_lock = 0
last_action_text = "NONE"
episode = 1
best_score = 0
episode_steps = 0


def toggle_fullscreen():
    global fullscreen, display_surface
    fullscreen = not fullscreen
    if fullscreen:
        display_surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.RESIZABLE)
    else:
        display_surface = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)


def present_frame():
    """Scale the virtual screen to the current display while preserving aspect ratio."""
    dw, dh = display_surface.get_size()
    if dw <= 0 or dh <= 0:
        return
    scale = min(dw / BASE_W, dh / BASE_H)
    scaled_w = max(1, int(BASE_W * scale))
    scaled_h = max(1, int(BASE_H * scale))
    ox = (dw - scaled_w) // 2
    oy = (dh - scaled_h) // 2
    display_surface.fill((4, 12, 4))
    scaled = pygame.transform.smoothscale(screen, (scaled_w, scaled_h))
    display_surface.blit(scaled, (ox, oy))
    pygame.display.flip()


def draw_toggle_key(x, y, key, label, active):
    """Highlighted hotkey boxes. Active keys look darker and bolder."""
    box_color = DARK if active else BG
    text_color = BG if active else DARK
    pygame.draw.rect(screen, box_color, (x, y, 22, 20))
    pygame.draw.rect(screen, DARK, (x, y, 22, 20), 2)
    screen.blit(small_font.render(key, True, text_color), (x + 6, y + 1))
    screen.blit(tiny_font.render(label, True, DARK), (x + 28, y + 5))


def reset_game():
    global snake, direction, bullets, roaches, particles
    global score, lives, slurpee_cups, reward_total, level, combo
    global shoot_cooldown, direction_lock, last_action_text, roach_timer, screen_shake, episode_steps
    snake = [(5, ROWS // 2), (4, ROWS // 2), (3, ROWS // 2)]
    direction = (1, 0)
    bullets = []
    roaches = []
    particles = []
    score = 0
    lives = 3
    slurpee_cups = 0
    reward_total = 0.0
    level = 1
    combo = 0
    shoot_cooldown = 0
    direction_lock = 0
    last_action_text = "NONE"
    roach_timer = 0
    screen_shake = 0
    episode_steps = 0


def cell_rect(x, y, pad=1):
    return pygame.Rect(x * CELL + pad, y * CELL + pad, CELL - pad * 2, CELL - pad * 2)


def draw_cell(x, y, color=DARK, pad=1):
    pygame.draw.rect(screen, color, cell_rect(x, y, pad))


def draw_background():
    screen.fill(BG)
    # Very subtle Nokia LCD texture, not grid lines.
    for y in range(0, GAME_H, 4):
        pygame.draw.line(screen, BG_2, (0, y), (GAME_W, y), 1)
    pygame.draw.rect(screen, DARK, (0, 0, GAME_W, GAME_H), 4)
    pygame.draw.rect(screen, MID, (5, 61, GAME_W - 10, GAME_H - 66), 2)


def draw_slurpee_cups(x, y, cups):
    """Straight cups. Filled cups become black / DARK from the bottom upward."""
    max_cups = 6
    for i in range(max_cups):
        cx = x + i * 23
        cy = y
        fill = i < cups
        # cup outline, not slanted
        pygame.draw.rect(screen, DARK, (cx + 4, cy + 8, 14, 20), 2)
        pygame.draw.rect(screen, DARK, (cx + 2, cy + 6, 18, 4), 2)
        pygame.draw.line(screen, DARK, (cx + 12, cy), (cx + 12, cy + 7), 2)
        pygame.draw.line(screen, DARK, (cx + 12, cy), (cx + 17, cy - 3), 1)
        if fill:
            pygame.draw.rect(screen, DARK, (cx + 7, cy + 14, 8, 11))
            pygame.draw.rect(screen, DARK, (cx + 6, cy + 20, 10, 6))
        else:
            pygame.draw.rect(screen, MID, (cx + 7, cy + 21, 8, 4))


def draw_lives(x, y):
    text = font.render(f"LIVES {lives}", True, DARK)
    screen.blit(text, (x, y))


def spawn_roach():
    if len(roaches) >= MAX_ROACHES:
        return
    side = random.choice(["right", "top", "bottom"])
    if side == "right":
        x, y = COLS - 2, random.randint(5, ROWS - 3)
    elif side == "top":
        x, y = random.randint(12, COLS - 3), 5
    else:
        x, y = random.randint(12, COLS - 3), ROWS - 3
    roll = random.random()
    if roll < 0.08:
        roach_type, hp = "FAST", 1
    elif roll < 0.18:
        roach_type, hp = "TANK", 2
    else:
        roach_type, hp = "NORMAL", 1
    roaches.append({"x": x, "y": y, "type": roach_type, "hp": hp, "zig": random.choice([-1, 1])})


def nearest_roach():
    if not roaches:
        return None
    hx, hy = snake[0]
    return min(roaches, key=lambda r: abs(r["x"] - hx) + abs(r["y"] - hy))


def danger_at(pos):
    x, y = pos
    if x < 0 or x >= COLS or y < 4 or y >= ROWS:
        return 1
    if pos in snake:
        return 1
    for r in roaches:
        if r["x"] == x and r["y"] == y:
            return 1
    return 0


def normalized_state():
    r = nearest_roach()
    hx, hy = snake[0]
    dx, dy = direction
    front = (hx + dx, hy + dy)
    left = (hx - dy, hy + dx)
    right = (hx + dy, hy - dx)
    danger_front = danger_at(front)
    danger_left = danger_at(left)
    danger_right = danger_at(right)
    if r:
        rx = (r["x"] - hx) / COLS
        ry = (r["y"] - hy) / ROWS
        dist = (abs(r["x"] - hx) + abs(r["y"] - hy)) / (COLS + ROWS)
        aligned = 1.0 if hx == r["x"] or hy == r["y"] else 0.0
        close = 1.0 if abs(r["x"] - hx) + abs(r["y"] - hy) <= 5 else 0.0
        roach_fast = 1.0 if r["type"] == "FAST" else 0.0
        roach_tank = 1.0 if r["type"] == "TANK" else 0.0
    else:
        rx = ry = aligned = close = roach_fast = roach_tank = 0.0
        dist = 1.0
    return [float(v) for v in [
        hx / COLS, hy / ROWS, dx, dy, rx, ry, dist, aligned, close,
        danger_front, danger_left, danger_right, shoot_cooldown / 10.0,
        roach_fast, roach_tank,
    ]]


def get_q_values(state):
    with torch.no_grad():
        state_tensor = torch.tensor([state], dtype=torch.float32, device=DEVICE)
        return policy_net(state_tensor)[0].detach().cpu().tolist()


def choose_action(state):
    global epsilon, last_q_values
    last_q_values = get_q_values(state)
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    return int(max(range(N_ACTIONS), key=lambda i: last_q_values[i]))


def remember(state, action, reward, next_state, done):
    memory.append(Experience(state, action, reward, next_state, done))


def train_dqn():
    global train_steps, last_loss
    dynamic_every = TRAIN_EVERY
    try:
        current_speed = speed_levels[speed_index]
        if current_speed >= 40:
            dynamic_every = 30
        elif current_speed >= 32:
            dynamic_every = 18
        elif current_speed >= 24:
            dynamic_every = 12
    except Exception:
        dynamic_every = TRAIN_EVERY
    if total_steps % dynamic_every != 0:
        return
    if len(memory) < MIN_MEMORY_TO_TRAIN:
        return
    batch_size = min(BATCH_SIZE, len(memory))
    batch = random.sample(memory, batch_size)
    states = torch.tensor([e.state for e in batch], dtype=torch.float32, device=DEVICE)
    actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = torch.tensor([e.next_state for e in batch], dtype=torch.float32, device=DEVICE)
    dones = torch.tensor([e.done for e in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    current_q = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(1, keepdim=True)[0]
        target_q = rewards + GAMMA * next_q * (1.0 - dones)
    loss = loss_fn(current_q, target_q)
    if not torch.isfinite(loss):
        return
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.8)
    optimizer.step()
    last_loss = float(loss.item())
    train_steps += 1
    if train_steps % TARGET_UPDATE_EVERY == 0:
        target_net.load_state_dict(policy_net.state_dict())


def decay_epsilon():
    global epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)


def shoot():
    global shoot_cooldown
    if shoot_cooldown > 0 or len(bullets) >= MAX_BULLETS:
        return False
    hx, hy = snake[0]
    dx, dy = direction
    bullets.append({"x": hx + dx, "y": hy + dy, "dx": dx, "dy": dy})
    shoot_cooldown = 7
    play(SFX_SHOOT)
    return True


def move_snake(action):
    global direction, direction_lock
    old_direction = direction
    new_dir = direction
    if direction_lock <= 0:
        if action == "UP":
            new_dir = (0, -1)
        elif action == "DOWN":
            new_dir = (0, 1)
        elif action == "LEFT":
            new_dir = (-1, 0)
        elif action == "RIGHT":
            new_dir = (1, 0)
        if (new_dir[0] + direction[0], new_dir[1] + direction[1]) != (0, 0):
            if new_dir != direction:
                direction_lock = 3
            direction = new_dir
    hx, hy = snake[0]
    dx, dy = direction
    new_head = (hx + dx, hy + dy)
    if new_head[0] < 0:
        new_head = (COLS - 1, new_head[1])
    elif new_head[0] >= COLS:
        new_head = (0, new_head[1])
    if new_head[1] < 4:
        new_head = (new_head[0], ROWS - 1)
    elif new_head[1] >= ROWS:
        new_head = (new_head[0], 4)
    snake.insert(0, new_head)
    snake.pop()
    if direction != old_direction or random.random() < 0.08:
        play(SFX_MOVE)


def grow_snake():
    snake.append(snake[-1])


def move_roaches(tick):
    hx, hy = snake[0]
    for r in roaches:
        speed_step = 2 if r["type"] == "FAST" and tick % 4 == 0 else 1
        for _ in range(speed_step):
            if random.random() < 0.12:
                r["y"] += r["zig"]
                if r["y"] <= 4 or r["y"] >= ROWS - 1:
                    r["zig"] *= -1
                continue
            if abs(r["x"] - hx) > abs(r["y"] - hy):
                r["x"] += 1 if r["x"] < hx else -1 if r["x"] > hx else 0
            else:
                r["y"] += 1 if r["y"] < hy else -1 if r["y"] > hy else 0
            r["x"] = max(0, min(COLS - 1, r["x"]))
            r["y"] = max(4, min(ROWS - 1, r["y"]))


def add_particles(x, y, amount=8):
    while len(particles) > MAX_PARTICLES:
        particles.pop(0)
    amount = min(amount, max(0, MAX_PARTICLES - len(particles)))
    for _ in range(amount):
        particles.append({
            "x": x * CELL + CELL // 2,
            "y": y * CELL + CELL // 2,
            "vx": random.uniform(-1.5, 1.5),
            "vy": random.uniform(-1.5, 1.5),
            "life": random.randint(8, 16),
        })


def update_particles():
    for p in particles[:]:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1
        if p["life"] <= 0:
            particles.remove(p)


def draw_particles():
    for p in particles:
        size = 2 + int(p["life"] > 10)
        pygame.draw.rect(screen, DARK, (int(p["x"]), int(p["y"]), size, size))


def draw_roach(r):
    x, y = r["x"], r["y"]
    px, py = x * CELL, y * CELL
    if r["type"] == "TANK":
        pygame.draw.rect(screen, DARK, (px + 1, py + 1, CELL - 2, CELL - 2), 2)
        pygame.draw.rect(screen, DARK, (px + 4, py + 5, CELL - 8, CELL - 8))
    elif r["type"] == "FAST":
        pygame.draw.polygon(screen, DARK, [(px + 8, py + 1), (px + 15, py + 14), (px + 2, py + 14)])
    else:
        pygame.draw.rect(screen, DARK, (px + 3, py + 4, 10, 9))
        pygame.draw.line(screen, DARK, (px + 2, py + 5), (px, py + 2), 1)
        pygame.draw.line(screen, DARK, (px + 14, py + 5), (px + 16, py + 2), 1)
        pygame.draw.line(screen, DARK, (px + 2, py + 12), (px, py + 15), 1)
        pygame.draw.line(screen, DARK, (px + 14, py + 12), (px + 16, py + 15), 1)
    pygame.draw.rect(screen, BG, (px + 5, py + 6, 2, 2))
    pygame.draw.rect(screen, BG, (px + 10, py + 6, 2, 2))


def draw_snake():
    for i, (x, y) in enumerate(snake):
        px, py = x * CELL, y * CELL
        pygame.draw.rect(screen, DARK, (px + 1, py + 1, CELL - 2, CELL - 2))
        pygame.draw.rect(screen, MID if i else BG, (px + 4, py + 4, CELL - 8, CELL - 8), 1)
        if i == 0:
            pygame.draw.rect(screen, BG, (px + 10, py + 4, 3, 3))
            pygame.draw.rect(screen, BG, (px + 10, py + 9, 3, 3))


def draw_hud():
    # Three clean rows so the Q text never runs into lives or Slurpee cups.
    pygame.draw.rect(screen, BG, (4, 4, GAME_W - 8, 76))
    pygame.draw.rect(screen, DARK, (4, 4, GAME_W - 8, 76), 2)

    screen.blit(font.render(f"SCORE {score}   BEST {best_score}", True, DARK), (12, 8))
    screen.blit(small_font.render(f"LVL {level}   COMBO {combo}   ACTION {last_action_text}", True, DARK), (12, 34))
    screen.blit(small_font.render(f"EPS {epsilon:.2f}   QAVG {sum(last_q_values)/len(last_q_values):+.2f}   LOSS {last_loss:.4f}", True, DARK), (12, 55))

    draw_lives(GAME_W - 104, 8)
    draw_slurpee_cups(GAME_W - 158, 41, slurpee_cups)


def draw_bar(x, y, w, h, value, label=""):
    pygame.draw.rect(screen, MID, (x, y, w, h), 1)
    if value >= 0:
        fill_w = int(min(1.0, abs(value)) * w)
        pygame.draw.rect(screen, DARK, (x, y, fill_w, h))
    else:
        fill_w = int(min(1.0, abs(value)) * w)
        pygame.draw.rect(screen, DARK, (x + w - fill_w, y, fill_w, h))
    if label:
        screen.blit(tiny_font.render(label, True, DARK), (x, y - 12))


def draw_q_graph(x, y, w, h):
    pygame.draw.rect(screen, DARK, (x, y, w, h), 1)
    if len(q_history) < 2:
        return
    vals = list(q_history)
    mn, mx = min(vals), max(vals)
    if abs(mx - mn) < 1e-6:
        mn -= 1
        mx += 1
    pts = []
    for i, val in enumerate(vals):
        px = x + int(i * (w - 1) / max(1, len(vals) - 1))
        py = y + h - int((val - mn) * (h - 1) / (mx - mn))
        pts.append((px, py))
    if len(pts) > 1:
        pygame.draw.lines(screen, DARK, False, pts, 2)
    screen.blit(tiny_font.render(f"Q avg {vals[-1]:+.2f}", True, DARK), (x + 4, y + 4))


def draw_network_visual(x, y, w, h, state):
    """Draw a small neural network that always fits inside its own panel."""
    panel_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, MID, panel_rect, 1)

    old_clip = screen.get_clip()
    screen.set_clip(panel_rect.inflate(-2, -2))

    title = small_font.render("NEURAL NETWORK", True, DARK)
    screen.blit(title, (x + 10, y + 7))

    h1, h2, qv = policy_net.activations(state)

    # Keep the visual intentionally small. The real network is 15 -> 128 -> 128 -> 5,
    # but drawing all neurons would never fit. These are representative live activations.
    layers = [state[:6], h1[:6], h2[:6], qv]
    labels = ["S", "H1", "H2", "Q"]

    top = y + 42
    bottom = y + h - 48
    usable_h = max(60, bottom - top)
    left = x + 34
    right = x + w - 116
    layer_gap = (right - left) / 3
    layer_xs = [int(left + i * layer_gap) for i in range(4)]

    node_positions = []
    for li, layer in enumerate(layers):
        positions = []
        lx = layer_xs[li]
        screen.blit(tiny_font.render(labels[li], True, DARK), (lx - 8, y + 26))
        count = len(layer)
        if count <= 1:
            ys = [top + usable_h // 2]
        else:
            ys = [int(top + i * usable_h / (count - 1)) for i in range(count)]
        for ni, val in enumerate(layer):
            ly = ys[ni]
            radius = 4 if li < 3 else 5
            active = abs(float(val)) > 0.08
            pygame.draw.circle(screen, DARK if active else MID, (lx, ly), radius, 0 if active else 1)
            positions.append((lx, ly))
        node_positions.append(positions)

    # Draw sparse connections first, then redraw nodes on top.
    for li in range(len(node_positions) - 1):
        for a in node_positions[li]:
            for b in node_positions[li + 1]:
                pygame.draw.line(screen, MID, a, b, 1)

    for li, positions in enumerate(node_positions):
        for ni, pos in enumerate(positions):
            val = layers[li][ni]
            radius = 4 if li < 3 else 5
            active = abs(float(val)) > 0.08
            pygame.draw.circle(screen, DARK if active else MID, pos, radius, 0 if active else 1)

    # Q labels live in their own right column inside this panel.
    qx = x + w - 102
    qy = y + 44
    screen.blit(tiny_font.render("OUTPUT Q", True, DARK), (qx, qy - 18))
    for i, q in enumerate(qv):
        row_y = qy + i * 20
        if row_y + 12 < y + h - 8:
            txt = tiny_font.render(f"{ACTIONS[i]:5s} {q:+.2f}", True, DARK)
            screen.blit(txt, (qx, row_y))

    note = tiny_font.render("compact live view: 6 + 6 neurons shown", True, DARK)
    screen.blit(note, (x + 10, y + h - 24))

    screen.set_clip(old_clip)


def draw_sparkline(values, x, y, w, h, title):
    pygame.draw.rect(screen, MID, (x, y, w, h), 1)
    screen.blit(tiny_font.render(title, True, DARK), (x + 6, y + 4))
    vals = list(values)
    if len(vals) < 2:
        return
    inner_x, inner_y = x + 8, y + 20
    inner_w, inner_h = w - 16, h - 28
    mn, mx = min(vals), max(vals)
    if abs(mx - mn) < 1e-6:
        mn -= 1
        mx += 1
    pts = []
    for i, val in enumerate(vals):
        px = inner_x + int(i * (inner_w - 1) / max(1, len(vals) - 1))
        py = inner_y + inner_h - int((val - mn) * (inner_h - 1) / (mx - mn))
        pts.append((px, py))
    pygame.draw.lines(screen, DARK, False, pts, 2)


def draw_bottom_lab(state):
    x, y, w, h = 0, BOTTOM_Y, GAME_W, BOTTOM_H
    pygame.draw.rect(screen, BG, (x, y, w, h))
    pygame.draw.rect(screen, DARK, (x + 4, y + 8, w - 8, h - 14), 3)
    screen.blit(font.render("AI LAB / TELEMETRY", True, DARK), (x + 14, y + 18))
    cx, cy = x + 14, y + 48
    draw_toggle_key(cx, cy, "H", "HUMAN", human_mode)
    draw_toggle_key(cx + 94, cy, "T", f"SPEED {speed_levels[speed_index]}", True)
    draw_toggle_key(cx + 218, cy, "S", "SAVE", False)
    draw_toggle_key(cx + 294, cy, "L", "LOAD", False)
    draw_toggle_key(cx + 370, cy, "F", "FULL", fullscreen)

    qv = last_q_values if last_q_values else [0.0] * N_ACTIONS
    best_i = int(max(range(len(qv)), key=lambda i: qv[i]))
    q_sorted = sorted(qv, reverse=True)
    margin = q_sorted[0] - q_sorted[1] if len(q_sorted) > 1 else 0.0
    confidence = min(1.0, max(0.0, margin / (abs(q_sorted[0]) + 1.0)))

    ex_x, ex_y = x + 14, y + 88
    pygame.draw.rect(screen, MID, (ex_x, ex_y, 300, 86), 1)
    screen.blit(small_font.render("DECISION", True, DARK), (ex_x + 8, ex_y + 8))
    mode_txt = "HUMAN OVERRIDE" if human_mode else "DQN CONTROL"
    screen.blit(tiny_font.render(f"mode: {mode_txt}", True, DARK), (ex_x + 8, ex_y + 30))
    screen.blit(tiny_font.render(f"best action: {ACTIONS[best_i]}   Q {qv[best_i]:+.2f}", True, DARK), (ex_x + 8, ex_y + 46))
    screen.blit(tiny_font.render(f"confidence {confidence*100:3.0f}%", True, DARK), (ex_x + 8, ex_y + 62))
    pygame.draw.rect(screen, MID, (ex_x + 120, ex_y + 66, 160, 8), 1)
    pygame.draw.rect(screen, DARK, (ex_x + 120, ex_y + 66, int(160 * confidence), 8))

    hx, hy = x + 328, y + 88
    pygame.draw.rect(screen, MID, (hx, hy, 145, 86), 1)
    screen.blit(small_font.render("ACTIONS", True, DARK), (hx + 8, hy + 8))
    recent = list(action_history)[-6:]
    for i, a in enumerate(recent):
        screen.blit(tiny_font.render(f"{i+1:02d} {a}", True, DARK), (hx + 10, hy + 30 + i * 9))

    sx, sy = x + 490, y + 88
    pygame.draw.rect(screen, MID, (sx, sy, 130, 86), 1)
    screen.blit(small_font.render("STATE", True, DARK), (sx + 8, sy + 8))
    screen.blit(tiny_font.render(f"danger F/L/R: {int(state[9])}/{int(state[10])}/{int(state[11])}", True, DARK), (sx + 8, sy + 30))
    screen.blit(tiny_font.render(f"roach dist: {state[6]:.2f}", True, DARK), (sx + 8, sy + 46))
    screen.blit(tiny_font.render(f"last reward: {last_reward_value:+.2f}", True, DARK), (sx + 8, sy + 62))

    draw_sparkline(reward_history, x + 14, y + 190, 285, 96, "REWARD TREND")
    draw_sparkline(q_history, x + 314, y + 190, 305, 96, "Q AVG TREND")


def draw_brain_panel(state):
    x = GAME_W
    pygame.draw.rect(screen, BG, (x, 0, PANEL_W, BASE_H))
    pygame.draw.rect(screen, DARK, (x + 8, 8, PANEL_W - 16, BASE_H - 16), 3)

    if not show_brain_panel:
        txt = font.render("AI PANEL HIDDEN", True, DARK)
        screen.blit(txt, (x + 78, 145))
        draw_toggle_key(x + 64, 205, "V", "VIEW PANEL", show_brain_panel)
        draw_toggle_key(x + 64, 235, "M", "MUSIC", music_on)
        draw_toggle_key(x + 64, 265, "N", "SOUND", sound_on)
        draw_toggle_key(x + 64, 295, "F", "FULLSCREEN", fullscreen)
        return

    screen.blit(font.render("DQN BRAIN", True, DARK), (x + 20, 18))
    draw_toggle_key(x + 20, 44, "V", "PANEL", show_brain_panel)
    draw_toggle_key(x + 108, 44, "M", "MUSIC", music_on)
    draw_toggle_key(x + 204, 44, "N", "SFX", sound_on)
    draw_toggle_key(x + 286, 44, "F", "FULL", fullscreen)

    screen.blit(tiny_font.render("Music: Krzysztof Szymanski from Pixabay", True, DARK), (x + 20, 73))

    stats_box = pygame.Rect(x + 18, 94, PANEL_W - 36, 96)
    pygame.draw.rect(screen, MID, stats_box, 1)
    screen.blit(small_font.render(f"episode {episode}   train {train_steps}", True, DARK), (x + 28, 104))
    screen.blit(small_font.render(f"memory {len(memory)}/{MEMORY_SIZE}", True, DARK), (x + 28, 124))
    screen.blit(tiny_font.render(f"device {DEVICE_LABEL}", True, DARK), (x + 210, 124))
    if LAST_ERROR:
        screen.blit(tiny_font.render(f"last error: {LAST_ERROR}", True, DARK), (x + 28, 170))
    screen.blit(small_font.render(f"epsilon {epsilon:.3f}   loss {last_loss:.4f}", True, DARK), (x + 28, 148))

    q_box = pygame.Rect(x + 18, 206, PANEL_W - 36, 138)
    pygame.draw.rect(screen, MID, q_box, 1)
    screen.blit(small_font.render("Q VALUES", True, DARK), (q_box.x + 10, q_box.y + 8))
    q_min = min(last_q_values) if last_q_values else -1
    q_max = max(last_q_values) if last_q_values else 1
    scale = max(abs(q_min), abs(q_max), 1.0)
    for i, qv in enumerate(last_q_values):
        row_y = q_box.y + 32 + i * 20
        label = f"{ACTIONS[i]:5s} {qv:+.2f}"
        screen.blit(tiny_font.render(label, True, DARK), (q_box.x + 10, row_y - 2))
        draw_bar(q_box.x + 118, row_y, 178, 10, qv / scale)

    graph_box = pygame.Rect(x + 18, 354, PANEL_W - 36, 78)
    pygame.draw.rect(screen, MID, graph_box, 1)
    screen.blit(small_font.render("Q TREND", True, DARK), (graph_box.x + 10, graph_box.y + 6))
    draw_q_graph(graph_box.x + 10, graph_box.y + 28, graph_box.w - 20, 44)

    network_box = pygame.Rect(x + 18, 448, PANEL_W - 36, 238)
    draw_network_visual(network_box.x, network_box.y, network_box.w, network_box.h, state)


def draw_game_over():
    overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
    overlay.fill((15, 56, 15, 50))
    screen.blit(overlay, (0, 0))
    msg = font.render("GAME OVER - DQN TRAINING CONTINUES", True, DARK)
    msg2 = small_font.render("Restarting episode...", True, DARK)
    pygame.draw.rect(screen, BG, (GAME_W // 2 - 205, GAME_H // 2 - 45, 410, 90))
    pygame.draw.rect(screen, DARK, (GAME_W // 2 - 205, GAME_H // 2 - 45, 410, 90), 3)
    screen.blit(msg, (GAME_W // 2 - msg.get_width() // 2, GAME_H // 2 - 20))
    screen.blit(msg2, (GAME_W // 2 - msg2.get_width() // 2, GAME_H // 2 + 10))


def update_level():
    global level
    level = 1 + score // 10


def save_model():
    torch.save({
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epsilon": epsilon,
        "train_steps": train_steps,
        "best_score": best_score,
    }, "snake_dqn_model.pt")


def load_model():
    global epsilon, train_steps, best_score
    try:
        data = torch.load("snake_dqn_model.pt", map_location=DEVICE)
        policy_net.load_state_dict(data["policy_net"])
        target_net.load_state_dict(data["target_net"])
        optimizer.load_state_dict(data["optimizer"])
        epsilon = float(data.get("epsilon", EPSILON_START))
        train_steps = int(data.get("train_steps", 0))
        best_score = int(data.get("best_score", 0))
        print("Loaded snake_dqn_model.pt")
    except FileNotFoundError:
        print("No saved model found. Starting fresh.")


reset_game()
running = True
tick = 0
roach_timer = 0
screen_shake = 0
autosave_timer = 0

while running:
    clock.tick(speed_levels[speed_index])
    update_music()

    if screen_shake > 0:
        shake_x = random.randint(-1, 1)
        shake_y = random.randint(-1, 1)
        screen_shake -= 1
    else:
        shake_x = shake_y = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            save_model()
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                save_model()
                running = False
            elif event.key == pygame.K_s:
                save_model()
            elif event.key == pygame.K_l:
                load_model()
            elif event.key == pygame.K_f:
                toggle_fullscreen()
            elif event.key == pygame.K_v:
                show_brain_panel = not show_brain_panel
            elif event.key == pygame.K_m:
                music_on = not music_on
                apply_music_state()
            elif event.key == pygame.K_n:
                sound_on = not sound_on
                apply_music_state()
            elif event.key == pygame.K_h:
                human_mode = not human_mode
            elif event.key == pygame.K_t:
                speed_index = (speed_index + 1) % len(speed_levels)

    old_state = normalized_state()
    if human_mode:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action_i = ACTIONS.index("UP")
        elif keys[pygame.K_DOWN]:
            action_i = ACTIONS.index("DOWN")
        elif keys[pygame.K_LEFT]:
            action_i = ACTIONS.index("LEFT")
        elif keys[pygame.K_RIGHT]:
            action_i = ACTIONS.index("RIGHT")
        elif keys[pygame.K_SPACE]:
            action_i = ACTIONS.index("SHOOT")
        else:
            action_i = ACTIONS.index(last_action_text) if last_action_text in ACTIONS else ACTIONS.index("RIGHT")
        last_q_values = get_q_values(old_state)
    else:
        action_i = choose_action(old_state)
    action = ACTIONS[action_i]
    last_action_text = action
    action_history.append(action)

    reward = -0.03
    done = False

    if shoot_cooldown > 0:
        shoot_cooldown -= 1
    if direction_lock > 0:
        direction_lock -= 1

    if action == "SHOOT":
        fired = shoot()
        reward += -0.05 if fired else -0.12

    old_direction = direction
    move_snake(action)
    if direction != old_direction:
        reward -= 0.015

    roach_timer += 1
    spawn_speed = max(5, 13 - level)
    if roach_timer >= spawn_speed:
        spawn_roach()
        roach_timer = 0

    for b in bullets[:]:
        b["x"] += b["dx"]
        b["y"] += b["dy"]
        if b["x"] < 0 or b["x"] >= COLS or b["y"] < 4 or b["y"] >= ROWS:
            bullets.remove(b)

    if tick % 3 == 0:
        move_roaches(tick)

    hit_this_tick = False
    for b in bullets[:]:
        for r in roaches[:]:
            if b["x"] == r["x"] and b["y"] == r["y"]:
                if b in bullets:
                    bullets.remove(b)
                r["hp"] -= 1
                if r["hp"] <= 0:
                    roaches.remove(r)
                    score += 1
                    combo += 1
                    slurpee_cups = min(6, slurpee_cups + 1)
                    reward_gain = 8 + min(combo, 10)
                    reward += reward_gain
                    grow_snake()
                    add_particles(r["x"], r["y"], 12)
                    screen_shake = 3
                    hit_this_tick = True
                    play(SFX_EAT)
                else:
                    reward += 2
                    add_particles(r["x"], r["y"], 5)
                    play(SFX_HIT)
                break

    if not hit_this_tick:
        combo = max(0, combo - 1)

    for r in roaches[:]:
        if (r["x"], r["y"]) == snake[0]:
            roaches.remove(r)
            lives -= 1
            combo = 0
            reward -= 12
            add_particles(r["x"], r["y"], 14)
            screen_shake = 5
            play(SFX_HIT)

    if snake[0] in snake[1:]:
        lives -= 1
        reward -= 15
        combo = 0
        add_particles(snake[0][0], snake[0][1], 14)
        snake[:] = snake[:3]
        screen_shake = 5
        play(SFX_HIT)

    if lives <= 0:
        done = True
        reward -= 20
        play(SFX_GAMEOVER)

    episode_steps += 1
    if episode_steps >= MAX_STEPS_PER_EPISODE:
        done = True

    update_particles()
    update_level()

    new_state = normalized_state()
    remember(old_state, action_i, reward, new_state, done)
    try:
        train_dqn()
    except RuntimeError as e:
        LAST_ERROR = str(e)[:38]
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    decay_epsilon()

    reward_total += reward
    last_reward_value = reward
    total_steps += 1
    q_history.append(sum(last_q_values) / len(last_q_values))
    reward_history.append(reward)

    draw_background()
    if shake_x or shake_y:
        # Shake only the active objects slightly.
        pass

    draw_snake()
    for b in bullets:
        draw_cell(b["x"], b["y"], MID, 3)
    for r in roaches:
        draw_roach(r)
    draw_particles()
    draw_hud()
    draw_bottom_lab(new_state)
    draw_brain_panel(new_state)

    if done:
        best_score = max(best_score, score)
        draw_game_over()
        present_frame()
        pygame.time.wait(1200)
        episode += 1
        if episode % 5 == 0:
            save_model()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        reset_game()

    present_frame()
    tick += 1
    autosave_timer += 1
    if autosave_timer >= 3600:
        save_model()
        autosave_timer = 0

pygame.quit()
