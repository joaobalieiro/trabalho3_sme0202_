import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ======================================
# PARTE 1: metodo ftcs explicito para adveccao e difusao
# ======================================

# parametros do dominio
L = 15.0
T = 12.0
dx = 0.1
x = np.arange(0, L + dx, dx)
nx = len(x)

# condicao inicial
u0 = np.exp(-20 * (x - 2) ** 2) + np.exp(-(x - 5) ** 2)

# numeros de peclet para os tres regimes
Pe_values = [0.1, 1.0, 100.0]

fig = plt.figure(figsize=(18, 5))

for idx, Pe in enumerate(Pe_values, start=1):
    # restricao de estabilidade: dt < min{dx^2/(2/Pe + dx), dx}
    dt1 = dx ** 2 / (2 / Pe + dx)
    # fator de seguranca
    dt = 0.9 * min(dt1, dx)
    t = np.arange(0, T + dt, dt)
    nt = len(t)

    # inicializacao do array de solucao
    u = np.zeros((nx, nt))
    u[:, 0] = u0.copy()

    # integracao temporal: ftcs explicito
    for n in range(nt - 1):
        # condicoes periodicas: roll para vizinhos
        up = np.roll(u[:, n], -1)
        um = np.roll(u[:, n], 1)

        # atualizacao explicita
        u[:, n + 1] = u[:, n] - (dt / dx) * (u[:, n] - um) \
                      + (dt / (Pe * dx ** 2)) * (up - 2 * u[:, n] + um)

    # plot em 3d
    ax = fig.add_subplot(1, 3, idx, projection='3d')
    X, Y = np.meshgrid(x, t)
    ax.plot_surface(X, Y, u.T, cmap='viridis', edgecolor='none')
    ax.set_title(f'Pe = {Pe}')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')

plt.tight_layout()
plt.show()

# ======================================
# PARTE 2: comparacao de esquemas centralizado e upwind
# ======================================

# esquema upwind para termo advectivo e centered para difusao
def solver_upwind_centered(Pe):
    # restricao de estabilidade
    dt1 = dx**2 / (2/Pe + dx)
    dt = 0.9 * min(dt1, dx)
    t = np.arange(0, T+dt, dt)
    nt = len(t)
    u = np.zeros((nt, nx))
    u[0] = u0.copy()
    for n in range(nt-1):
        um = np.roll(u[n], 1)
        up = np.roll(u[n], -1)
        # derivada advectiva por upwind (a>0)
        ux = (u[n] - um) / dx
        # derivada segunda centralizada para difusao
        uxx = (up - 2*u[n] + um) / dx**2
        u[n+1] = u[n] - dt * ux + dt/Pe * uxx
    return t, u

# valores de peclet para estudo
Pe_vals = [0.1, 1.0, 10.0, 100.0]
labels = ['Pe << 1', 'Pe = 1', 'Pe na ordem de dezenas', 'Pe na ordem de centenas']

# tempos de amostragem (6 tempos igualmente espaçados)
t_samples = np.linspace(0, T, 6)

# preparacao da figura
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

for ax, Pe, label in zip(axes.flatten(), Pe_vals, labels):
    t, u = solver_upwind_centered(Pe)
    # acha indices proximos aos tempos de amostragem
    idxs = [np.argmin(np.abs(t - ts)) for ts in t_samples]
    for i, ts in zip(idxs, t_samples):
        ax.plot(x, u[i], label=f"t={ts:.1f}")
    ax.set_title(label)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend(fontsize='small')

plt.tight_layout()
plt.show()

# ======================================
# PARTE 3: condicoes de estabilidade em funcao de pe
# ======================================

# gera vetor de pe de 0.1 a 1000 em escala logaritmica
pe = np.logspace(-1, 3, 200)

# restricao combinada difusao/advectiva
dt_diff_adv = dx**2 / (2/pe + dx)

# restricao advectiva pura (cfl)
dt_adv = dx * np.ones_like(pe)

# valor critico de pe onde 2/pe = dx
pe_critico = 2 / dx

# plot em escala log-log
plt.figure(figsize=(8, 5))
plt.loglog(pe, dt_diff_adv, label='dt < dx**2 / (2/pe + dx)')
plt.loglog(pe, dt_adv, '--', label='dt < dx')
plt.axvline(pe_critico, color='gray', linestyle=':', label=f'Pe = {pe_critico:.1f}')
plt.xlabel('Número de Peclet (Pe)')
plt.ylabel('Restrição de dt')
plt.title('Condições de estabilidade em função de Pe')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.show()
