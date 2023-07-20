import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

def distancia_euclidiana(punto_origen, lista_puntos):
    # Calcular las diferencias entre el punto de origen y los puntos de la lista
    diferencias = lista_puntos - punto_origen
    # Elevar al cuadrado las diferencias y sumarlas
    distancias_cuadradas = np.sum(diferencias**2, axis=1)
    # Tomar la raíz cuadrada para obtener la distancia euclidiana
    distancias_euclidianas = np.sqrt(distancias_cuadradas)
    return distancias_euclidianas


class SingleAgent(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1
    STOP = 2

    def __init__(self, puntos_cercanos, grid_size, render_mode="console"):
        super(SingleAgent, self).__init__()
        self.render_mode = render_mode

        # Size of the 1D-grid
        self.grid_size = grid_size
        # Initialize the agent at the right of the grid
        self.agent_pos = np.random.randint(0, grid_size)
        self.puntos_cercanos = puntos_cercanos
        self.prev_distance = None
        self.visitados = np.ones((self.grid_size))
        print(len(self.visitados), grid_size)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 3
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        # Initialize the agent at the right of the grid
        # self.agent_pos = self.grid_size - 1
        self.agent_pos = np.random.randint(0, high=self.grid_size)
        self.visitados = np.ones((self.grid_size))
        self.prev_distance = None
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32), {}  # empty info dict

    def step(self, action):
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        elif action == self.STOP:
            pass
        else:
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )

        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size-1)

        # Are we at the left of the grid?
        # terminated = bool(self.agent_pos == 0)
        truncated = False  # we do not limit the number of steps here

        # Null reward everywhere except when reaching the goal (left of the grid)
        # reward = 1 if self.agent_pos == 0 else 0

        # centroide
        # print(len(self.puntos_cercanos))
        actual_distance = np.average(distancia_euclidiana(self.agent_pos, self.puntos_cercanos))
        if self.prev_distance is None: self.prev_distance = actual_distance
        # if self.prev_distance > actual_distance:
        #     reward = 1
        # else:
        #     reward = 0
        # print(actual_distance.shape)
        reward = np.abs(self.prev_distance - actual_distance) - self.visitados[self.agent_pos]
        # print(self.prev_distance, actual_distance, reward)
        # terminated = bool(self.prev_distance == actual_distance)
        terminated = False
        # if reward != 0:
        self.prev_distance = actual_distance
        self.visitados[self.agent_pos] += 1

        # Optionally we can pass additional info, we are not using that for now
        # info = {"pos":self.agent_pos}
        info = {}

        return (
            np.array([self.agent_pos]).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print("." * self.agent_pos, end="")
            print("x", end="")
            print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass
