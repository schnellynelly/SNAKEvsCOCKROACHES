# Worms DQN Retro Game

A Nokia inspired 8 bit Snake vs Cockroaches game powered by Deep Q Learning.

The agent learns using a Deep Q Network, replay memory, epsilon greedy exploration, and live Q value visualization.

<img width="1215" height="852" alt="Animation" src="https://github.com/user-attachments/assets/678912ed-6f66-4b73-a84f-2c02801979aa" />


## Features

- Deep Q Network gameplay agent
- Live Q value panel
- Neural network visualization
- Fullscreen and windowed mode
- Human override mode
- Retro 8 bit music and sound effects
- GPU support with CUDA PyTorch when available
- Speed control for faster training

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Controls

```text
F = fullscreen/windowed
H = human override
T = speed
V = show/hide brain panel
M = music on/off
N = sound effects on/off
S = save model
L = load model
ESC = quit
```

## GPU Notes

The game automatically uses CUDA if your installed PyTorch build supports it.

For NVIDIA GPUs, install CUDA PyTorch with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pygame numpy
```

Then run:

```bash
python main.py
```

## Music Credit

Music by [Krzysztof Szymanski](https://pixabay.com/users/djartmusic-46653586/) from [Pixabay](https://pixabay.com/).

Track file included:

```text
djartmusic-fun-with-my-8-bit-game-301278.mp3
```

## License

Add your preferred license before publishing publicly.
