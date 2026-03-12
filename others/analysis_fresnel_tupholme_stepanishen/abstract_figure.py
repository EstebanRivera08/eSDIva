import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

mainpath = ""

N_continue = 1000
N_derivative = 20
t0 = 0  # Start time (us)
t_max = 4.5  # Maximum time (us)

t_continue = np.linspace(t0, t_max, N_continue)  # Upsampled time points
t_derivative = np.linspace(t0, t_max, N_derivative)  # Original time points
f_s_derivative = (N_derivative - 1) / (t_max - t0)

# Characteristics
l_c = 3

# Initial parameters and its range
area = [1.5, 0.6, 0.75]
Dt1 = [0.5, 0.1, 1]
Dt2 = [2, 1.5, 1]
shift = [-1.5, -1, -0.15]

# Marker size
SAVE_FIGURE = False
figsize = (4.3, 4.3)
font_size = 15
plt.rcParams.update({"font.size": font_size})
continuous_color = "b"
linewidth = 2.5
linewidth2 = 2
marker_size = 3.5
marker_color = "r"
xlim = (0, 4.5)
ylim = (-0.05, 1.5)
ylim2 = (-3, 3)


def compute_trapezoid(Dt1, Dt2, shift, area):
    t1 = l_c + shift - (Dt1 + Dt2) / 2
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + Dt1 + Dt2

    hmax = area / Dt2
    s1 = hmax / Dt1

    h_continue = np.zeros(N_continue)

    # Compute continue version
    for i in range(N_continue):
        if t_continue[i] < t1 or t_continue[i] > t4:
            continue
        elif t_continue[i] < t2:
            h_continue[i] = s1 * (t_continue[i] - t1)
        elif t_continue[i] < t3:
            h_continue[i] = hmax
        else:
            h_continue[i] = s1 * (t4 - t_continue[i])

    # Compute derivative version
    def middle_interp_delta(ti, sign, d2h):
        k_t_ = (ti - t0) * f_s_derivative + 1

        k_t_f = np.floor(k_t_).astype(int)
        w_f = 1 - (k_t_ - k_t_f)

        k_t_c = k_t_f + 1
        w_c = k_t_ - k_t_f

        d2h[k_t_f] += w_f * s1 * sign
        d2h[k_t_c] += w_c * s1 * sign
        return d2h

    d2h = np.zeros(N_derivative)
    d2h = middle_interp_delta(t1, +1, d2h)
    d2h = middle_interp_delta(t2, -1, d2h)
    d2h = middle_interp_delta(t3, -1, d2h)
    d2h = middle_interp_delta(t4, +1, d2h)

    dh = np.cumsum(d2h)

    h_derivative = np.cumsum(dh) / f_s_derivative

    times = (t1, t2, t3, t4)

    return times, h_continue, d2h, dh, h_derivative


# Create the figure and axes
n_rows = len(shift) + 1  # One row for the sum of trapezoids and deltas
n_cols = 2
fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

# Print independent trapezoids
times = np.zeros((len(shift), 4))
h_continue = np.zeros((len(shift), N_continue))
d2h = np.zeros((len(shift), N_derivative))
dh = np.zeros((len(shift), N_derivative))
h_derivative = np.zeros((len(shift), N_derivative))
h_sum = np.zeros(N_continue)
d2h_sum = np.zeros(N_derivative)

for i, _ in enumerate(shift):
    times, h_continue, d2h, dh, h_derivative = compute_trapezoid(
        Dt1[i], Dt2[i], shift[i], area[i]
    )

    # Cumulate the trapezoids and deltas
    h_sum += h_continue
    d2h2 = d2h.copy()
    d2h = np.sign(d2h) * np.log10(np.abs(100 * d2h))
    d2h[d2h2 == 0] = 0
    d2h_sum += d2h

    # Make each line of the subplot
    ax[i, 0].plot(
        t_continue,
        h_continue,
        "-",
        label=r"$h_{SIR}$",
        color=continuous_color,
        linewidth=linewidth,
    )

    markerline, stemlines, baseline = ax[i, 1].stem(
        t_derivative,
        d2h,
        linefmt=marker_color + "-",
        markerfmt=marker_color + "s",
        basefmt=" ",
        label=r"$\partial^2 h / \partial t^2$",
    )

    markerline.set_markersize(marker_size)
    stemlines.set_linewidth(linewidth2)

    ax[i, 0].set_yticks([])
    ax[i, 0].set_xticks([])

    ax[i, 1].set_yticks([])
    ax[i, 1].set_xticks([])

    if i == 1:
        ax[i, 0].set_ylabel("$h_m$")
        ax[i, 1].set_ylabel("$\partial^2 h_m / \partial t^2$")

i += 1
ax[i, 0].plot(
    t_continue,
    h_sum,
    "-",
    label=r"$\sum h_{SIR}$",
    color=continuous_color,
    linewidth=linewidth,
)
ax[i, 0].set_xlabel("$t$")
ax[i, 0].set_ylabel(r"$h_{tx}=\sum h_{m}$")
ax[i, 0].set_yticks([])
ax[i, 0].set_xticks([0, 1, 2, 3, 4])
ax[i, 0].tick_params(labelbottom=False)

markerline, stemlines, baseline = ax[i, 1].stem(
    t_derivative,
    d2h_sum,
    linefmt=marker_color + "-",
    markerfmt=marker_color + "s",
    basefmt=" ",
    label=r"$\partial^2 h / \partial t^2$",
)
markerline.set_markersize(marker_size)
stemlines.set_linewidth(linewidth2)
ax[i, 1].set_xlabel("$t$")
ax[i, 1].set_ylabel(
    r"$\frac{\partial^2 h_{tx}}{\partial t^2} = \sum \frac{\partial^2 h_m}{\partial t^2}$"
)
ax[i, 1].set_yticks([])
ax[i, 1].set_xticks([0, 1, 2, 3, 4])
ax[i, 1].tick_params(labelbottom=False)

fig.tight_layout()

for j in range(n_rows):
    ax[j, 0].set_ylim(ylim)
    ax[j, 0].set_xlim(xlim)
    ax[j, 1].set_ylim(ylim2)
    ax[j, 1].set_xlim(xlim)

if SAVE_FIGURE:
    fig.savefig(mainpath + "abstract_figure.svg", dpi=800)
plt.show()
