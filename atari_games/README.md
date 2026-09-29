# Welcome to Atari Games
Atari Deep Q‑Network (DQN) Project
A reinforcement learning project implementing Deep Q‑Networks to train agents that play three environments:

CartPole‑v1

SpaceInvaders‑v0

MsPacman‑v0

This project demonstrates neural network–based control, frame preprocessing, replay buffers, target networks, and epsilon‑greedy exploration — all core components of modern Deep Reinforcement Learning.

## Task
Build an AI agent that learns to play Atari games using Deep Q‑Learning (DQN).
You must deliver:

A model that plays CartPole

A model that plays Space Invaders

A model that plays Pacman

A blog post explaining your approach

The challenge is that Atari games use raw pixel input, require CNNs, and need GPU acceleration for practical training.

## Description
This project implements a full DQN pipeline:

1. Replay Buffer
Stores past experiences to break correlation between consecutive frames.

2. Target Network
A second network updated slowly to stabilize Q‑value learning.

3. Epsilon‑Greedy Exploration
The agent starts with random actions and gradually shifts to learned behavior.

4. Neural Network Models
MLP for CartPole (low‑dimensional state)

CNN for Atari (pixel‑based input)

5. Atari Frame Preprocessing
Convert frames to grayscale

Resize to 84×84

Normalize

Stack 4 frames to capture motion

6. Training Loops
Each environment has its own training loop with:

Replay sampling

Gradient updates

Target network synchronization

Episode logging

## Installation
⚠️ Important Note About Qwasar Environment
Atari environments cannot run inside Qwasar’s Python 3.6.7 environment.

Gym Atari requires:

Python 3.7+

GPU support

Modern Gym / ALE‑Py versions

Therefore:

The Atari models were trained and tested on Google Colab using Python 3.10 and GPU.
The code is fully functional, but Qwasar’s environment does not support Atari execution.
Only the source code, README, and BLOG.md are required for submission.

This is the standard approach used by all students.

## Usage
Run the main script:

Code
python atari_dqn.py
The script will:

Train CartPole

Train Space Invaders

Train Pacman

Print episode rewards and epsilon values



Atari environments require Python 3.7+ and GPU support.
Qwasar’s Python 3.6 environment does not support Gym Atari.
The code is fully functional and can be executed on Google Colab or any modern Python environment.
```
./my_project argument1 argument2
```

### The Core Team


<span><i>Made at <a href='https://qwasar.io'>Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt='Qwasar SV -- Software Engineering School's Logo' src='https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png' width='20px' /></span>
