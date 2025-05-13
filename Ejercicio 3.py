import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del problema
T0 = 90  # Temperatura inicial en °C
T_amb = 25  # Temperatura ambiente en °C
k = 0.07  # Constante de enfriamiento
t0 = 0  # Tiempo inicial en minutos
tf = 30  # Tiempo final en minutos
n = 30  # Número de pasos
h = (tf - t0) / n  # Tamaño de paso

# Definición de la ecuación diferencial
def f(t, T):
    return -k * (T - T_amb)

# Inicialización de listas
t_vals = [t0]
T_vals = [T0]

# Método de Euler
t = t0
T = T0
for i in range(n):
    T = T + h * f(t, T)  # Paso de Euler
    t = t + h
    t_vals.append(t)
    T_vals.append(T)

# Solución exacta
T_exacta = [T_amb + (T0 - T_amb) * np.exp(-k * t) for t in t_vals]

# Guardar resultados en CSV
data = {
    "t": t_vals,
    "T_aproximado": T_vals,
    "T_exacto": T_exacta
}
df = pd.DataFrame(data)
df.to_csv("enfriamiento_newton.csv", index=False)

# Graficar resultados
plt.figure(figsize=(8, 5))
plt.plot(t_vals, T_vals, 'o-', label="Euler", color="blue")
plt.plot(t_vals, T_exacta, '-', label="Solución exacta", color="red")
plt.title("Ley de Enfriamiento de Newton (Método de Euler)")
plt.xlabel("Tiempo (min)")
plt.ylabel("Temperatura (°C)")
plt.grid(True)
plt.legend()
plt.savefig("enfriamiento_newton.png")
plt.show()
