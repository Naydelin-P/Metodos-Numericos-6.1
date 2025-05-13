import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del problema
g = 9.81  # Gravedad
m = 2  # Masa
k = 0.5  # Coeficiente de fricción
v0 = 0  # Velocidad inicial
t0 = 0  # Tiempo inicial
tf = 10  # Tiempo final
n = 50  # Número de pasos
h = (tf - t0) / n  # Tamaño del paso

# Definición de la ecuación diferencial
def f(t, v):
    return g - (k / m) * v

# Inicialización de listas para almacenar resultados
t_vals = [t0]
v_vals = [v0]

# Método de Euler
t = t0
v = v0
for i in range(n):
    v = v + h * f(t, v)  # Paso de Euler
    t = t + h
    t_vals.append(t)
    v_vals.append(v)

# Solución exacta
v_exacta = [(m * g / k) * (1 - np.exp(- (k / m) * t)) for t in t_vals]

# Guardar resultados en CSV
data = {
    "t": t_vals,
    "v_aproximado": v_vals,
    "v_exacta": v_exacta
}
df = pd.DataFrame(data)
df.to_csv("caida_con_friccion.csv", index=False)

# Graficar resultados
plt.figure(figsize=(8, 5))
plt.plot(t_vals, v_vals, 'o-', label="Euler", color="blue")
plt.plot(t_vals, v_exacta, '-', label="Solución exacta", color="red")
plt.title("Caída con fricción (Método de Euler)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.grid(True)
plt.legend()
plt.savefig("caida_con_friccion.png")
plt.show()
