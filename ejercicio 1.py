import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del circuito
R = 1000  # Resistencia en ohmios
C = 0.001  # Capacitancia en faradios
V_fuente = 5  # Voltaje de la fuente en volts

# Definición de la ecuación diferencial
def f(t, V):
    return (1 / (R * C)) * (V_fuente - V)

# Condiciones iniciales
t0 = 0  # Tiempo inicial
V0 = 0  # Voltaje inicial
tf = 5  # Tiempo final
n = 20  # Número de pasos
h = (tf - t0) / n  # Paso

# Inicialización de listas para almacenar resultados
t_vals = [t0]
V_vals = [V0]

# Método de Euler
t = t0
V = V0
for i in range(n):
    V = V + h * f(t, V)  # Actualización usando Euler
    t = t + h
    t_vals.append(t)
    V_vals.append(V)

# Comparación con la solución analítica
V_analitica = [V_fuente * (1 - np.exp(-t / (R * C))) for t in t_vals]

# Guardar resultados en archivo CSV
data = {
    "t": t_vals,
    "V_aproximado": V_vals,
    "V_analitico": V_analitica
}
df = pd.DataFrame(data)
csv_path = "circuito_rc_euler.csv"
df.to_csv(csv_path, index=False)

# Graficar la solución aproximada y analítica
plt.figure(figsize=(8, 5))
plt.plot(t_vals, V_vals, 'o-', label='Solución aproximada (Euler)', color='blue')
plt.plot(t_vals, V_analitica, '-', label='Solución analítica', color='red')
plt.title('Modelo de Voltaje en Circuito RC (Método de Euler)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.grid(True)
plt.legend()
plt.savefig("circuito_rc_solucion.png")
plt.show()
