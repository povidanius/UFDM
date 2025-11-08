import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Δ(α,β) for general σx, σy, ρ ------------------------
def delta(alpha, beta, sigma_x, sigma_y, rho):
    phi_xy = np.exp(-0.5 * (sigma_x**2 * alpha**2 +
                             sigma_y**2 * beta**2 +
                             2 * rho * sigma_x * sigma_y * alpha * beta))
    phi_x  = np.exp(-0.5 * sigma_x**2 * alpha**2)
    phi_y  = np.exp(-0.5 * sigma_y**2 * beta**2)
    return phi_xy - phi_x * phi_y

# --- grid setup ------------------------------------------
rng = 3.0
n_points = 250
alpha = np.linspace(-rng, rng, n_points)
beta  = np.linspace(-rng, rng, n_points)
A, B = np.meshgrid(alpha, beta)

# initial parameters
sigma_x0, sigma_y0, rho0 = 1.0, 1.0, 0.5
Z = np.abs(delta(A, B, sigma_x0, sigma_y0, rho0))

# --- figure ----------------------------------------------
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)
im = ax.imshow(Z, extent=[-rng, rng, -rng, rng],
               origin='lower', cmap='magma')
ax.set_title(r"$|\Delta(\alpha,\beta)|$ for Gaussian $(\sigma_X,\sigma_Y,\rho)$")
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"$|\Delta|$")

# --- sliders ---------------------------------------------
axcolor = 'lightgoldenrodyellow'
ax_rho = plt.axes([0.25, 0.23, 0.65, 0.03], facecolor=axcolor)
ax_sx  = plt.axes([0.25, 0.17, 0.65, 0.03], facecolor=axcolor)
ax_sy  = plt.axes([0.25, 0.11, 0.65, 0.03], facecolor=axcolor)

s_rho = Slider(ax_rho, r'$\rho$', -0.99, 0.99, valinit=rho0)
s_sx  = Slider(ax_sx,  r'$\sigma_X$', 0.1, 3.0, valinit=sigma_x0)
s_sy  = Slider(ax_sy,  r'$\sigma_Y$', 0.1, 3.0, valinit=sigma_y0)

# --- update ----------------------------------------------
def update(val):
    rho = s_rho.val
    sx  = s_sx.val
    sy  = s_sy.val
    Z = np.abs(delta(A, B, sx, sy, rho))
    im.set_data(Z)
    im.set_clim(vmin=np.min(Z), vmax=np.max(Z))
    fig.canvas.draw_idle()

s_rho.on_changed(update)
s_sx.on_changed(update)
s_sy.on_changed(update)

plt.show()
