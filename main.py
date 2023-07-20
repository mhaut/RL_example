import numpy as np
from scipy.spatial import Delaunay
from entorno import SingleAgent, distancia_euclidiana
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt


# Lista de puntos cercanos con sus posiciones (x, y)
# puntos_cercanos = [
#     (0, 0),
#     (2, 1),
#     (1, 3),
#     (3, 3),
#     (4, 2),
#     (4, 0.5),
# ]

def generar_lista_pares(N):
    lista_pares = []
    for _ in range(N):
        x = np.random.random()  # Genera un número aleatorio entre 0 y 1 para la coordenada x
        y = np.random.random()  # Genera un número aleatorio entre 0 y 1 para la coordenada y
        lista_pares.append((x, y))
    return lista_pares

def update(frame, centroides, puntos_cercanos, triangulacion, obs):
    ax.clear()
    # Aquí coloca todo tu código de gráficas
    # Asegúrate de usar 'frame' para acceder a la posición actual en indices_ordenados
    plt.triplot(puntos_cercanos[:,0], puntos_cercanos[:,1], triangulacion.simplices)
    plt.scatter(puntos_cercanos[:, 0], puntos_cercanos[:, 1], marker='o', color='green', label='Puntos Cercanos')
    plt.plot(centroides[:, 0], centroides[:, 1], marker='x', color='red', label='Contorno')
    plt.plot(centroides[indices_ordenados[frame]][0], centroides[indices_ordenados[frame]][1], marker='o', color='red', label='Puntos Minimo')
    plt.plot(centroides[int(obs[0])][0], centroides[int(obs[0])][1], marker='o', color='blue', label='Puntos Actual')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Contorno de Puntos Cercanos')
    plt.legend()
    plt.grid(True)


puntos_cercanos = generar_lista_pares(100)
# Convertir la lista de puntos cercanos a un array de NumPy
puntos_cercanos = np.array(puntos_cercanos)
# Crear un objeto Delaunay para realizar la triangulación de Delaunay
triangulacion = Delaunay(puntos_cercanos)
# Definir el tamaño de la grilla para la interpolación
tamanio_grilla = 0.1

# Crear una malla de puntos (x, y) para la interpolación
x_min, y_min = np.min(puntos_cercanos, axis=0)
x_max, y_max = np.max(puntos_cercanos, axis=0)
x, y = np.meshgrid(np.arange(x_min, x_max, tamanio_grilla),
                   np.arange(y_min, y_max, tamanio_grilla))
# Obtener los índices de los triángulos que contienen cada punto de la malla
indice_triangulo = triangulacion.find_simplex(np.c_[x.ravel(), y.ravel()])

# Calcular los centroides de los triángulos
centroides = np.array([puntos_cercanos[triangulacion.simplices[i]].mean(axis=0) for i in indice_triangulo])
centroides = np.unique(centroides, axis=0)


dist = np.array([distancia_euclidiana(a, puntos_cercanos).sum() for a in centroides])
indices_ordenados = np.argsort(dist)
print("minimos", indices_ordenados[:10])
for a in indices_ordenados:
    print(a, dist[a])


# Crear una figura y representar los puntos de las puntos cercanos y el contorno
plt.triplot(puntos_cercanos[:,0], puntos_cercanos[:,1], triangulacion.simplices)
plt.scatter(puntos_cercanos[:, 0], puntos_cercanos[:, 1], marker='o', color='green', label='Puntos Cercanos')
plt.plot(centroides[indices_ordenados[0]][0], centroides[indices_ordenados[0]][1], marker='o', color='red', label='Puntos Minimo')
plt.plot(centroides[:, 0], centroides[:, 1], marker='x', color='red', label='Contorno')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Contorno de Puntos Cercanos')
plt.legend()
plt.grid(True)
plt.show()


env = SingleAgent(puntos_cercanos=puntos_cercanos, grid_size=len(centroides))
check_env(env, warn=True)

obs, _ = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())



# Train the agent
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env(SingleAgent, n_envs=1, env_kwargs=dict(puntos_cercanos=puntos_cercanos, grid_size=len(centroides)))
model = DQN("MlpPolicy", env, verbose=1).learn(100000, log_interval=4, progress_bar=True)
# Test the trained agent
# using the vecenv
obs = vec_env.reset()
n_steps = 30
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    print("obsobsobsobsobs", obs)
    print("obs=", obs, "reward=", reward, "done=", done)#, "position=", centroides[int(obs)])
    # vec_env.render()
    # if done:
    #     # Note that the VecEnv resets automatically
    #     # when a done signal is encountered
    #     print("Goal reached!", "reward=", reward)
    #     break
    update(frame, centroides, puntos_cercanos, triangulacion, obs)
    plt.triplot(puntos_cercanos[:,0], puntos_cercanos[:,1], triangulacion.simplices)
    plt.scatter(puntos_cercanos[:, 0], puntos_cercanos[:, 1], marker='o', color='green', label='Puntos Cercanos')
    plt.plot(centroides[indices_ordenados[0]][0], centroides[indices_ordenados[0]][1], marker='o', color='red', label='Puntos Minimo')
    plt.plot(centroides[int(obs[0])][0], centroides[int(obs[0])][1], marker='o', color='blue', label='Puntos Actual')
    plt.plot(centroides[:, 0], centroides[:, 1], marker='x', color='red', label='Contorno')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Contorno de Puntos Cercanos')
    plt.legend()
    plt.grid(True)
    plt.show()

ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, repeat=False)
ani.save('animacion_contorno.gif', writer='pillow')
plt.show()


exit()

