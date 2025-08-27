import matplotlib.pyplot as plt
import numpy as np

mainpath = r"C:\Users\INSERM\Documents\Esteban\Ressources" + "/"

N_continue = 1000
N_naive = 10
N_derivative = 10

t0 = 0  # Start time (us)
t_max = 10  # Maximum time (us)

t_continue = np.linspace(t0, t_max, N_continue)  # Upsampled time points
t_naive = np.linspace(t0, t_max, N_naive)  # Original time points
t_derivative = np.linspace(t0, t_max, N_derivative)  # Original time points
f_s_naive = (N_naive - 1) / (t_max - t0)
f_s_derivative = (N_derivative - 1) / (t_max - t0)

Dt1 = 0.1
Dt2 = 0.2
area = 1
shift = 3.4

name_fig = f"trap_Dt1{Dt1}_Dt2{Dt2}_shift{shift}_Nnaive{N_naive}_Nder{N_derivative}"

t1 = shift - (Dt1 + Dt2) / 2
t2 = t1 + Dt1
t3 = t1 + Dt2
t4 = t1 + Dt1 + Dt2

hmax = area / Dt2
s1 = hmax / Dt1

h_continue = np.zeros(N_continue)
h_naive = np.zeros(N_naive)
h_derivative = np.zeros(N_derivative)

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

# Compute naive version
for i in range(N_naive):
    if t_naive[i] < t1 or t_naive[i] > t4:
        continue
    elif t_naive[i] < t2:
        h_naive[i] = s1 * (t_naive[i] - t1)
    elif t_naive[i] < t3:
        h_naive[i] = hmax
    else:
        h_naive[i] = s1 * (t4 - t_naive[i])


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


# ---------------------------- Plotting ---------------------------
fig = plt.figure(figsize=(10, 4))
order = 1
plt.plot(t_continue, h_continue, "-", label="Continue", color="k", zorder=order)
order += 1
plt.vlines(
    t1,
    -hmax,
    hmax,
    linewidth=1,
    label="t1",
    color="gray",
    linestyles="--",
    zorder=order,
)
order += 1
plt.vlines(
    t2,
    -hmax,
    hmax,
    linewidth=1,
    label="t2",
    color="gray",
    linestyles="--",
    zorder=order,
)
order += 1
plt.vlines(
    t3,
    -hmax,
    hmax,
    linewidth=1,
    label="t3",
    color="gray",
    linestyles="--",
    zorder=order,
)
order += 1
plt.vlines(
    t4,
    -hmax,
    hmax,
    linewidth=1,
    label="t4",
    color="gray",
    linestyles="--",
    zorder=order,
)
order += 1
stem_container = plt.stem(
    t_derivative,
    d2h,
    linefmt="g-",
    markerfmt="gs",
    basefmt=" ",
    label="d2h",
)
# Set zorder for the components of the stem plot
for component in stem_container:
    if isinstance(component, list):  # Handles lines and markers
        for item in component:
            item.set_zorder(order)
    else:  # Handles the baseline
        component.set_zorder(order)
order += 1
plt.plot(t_derivative, dh, "y-s", label="dh", ms=4, zorder=order)
order += 1
plt.plot(t_derivative, h_derivative, "s", label="h_deriv", color="b", zorder=order)
order += 1
plt.plot(t_naive, h_naive, "o", label="h_naive", color="r", ms=5, zorder=order)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
# Place the legend outside the plot on the right
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid()
plt.tight_layout()  # Adjust layout to make space for the legend
plt.savefig(mainpath + name_fig + ".png", dpi=300)
plt.show()
