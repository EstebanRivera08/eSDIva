"""Learn 12 — Under the hood: the SDI trick behind eSDIva.

Seen from a field point, one small rectangular patch has a trapezoid-shaped spatial
impulse response (SIR) h(t): a ramp up, a plateau, a ramp down. Filling every time
sample of the trapezoid costs as many operations as it is wide. But its second
derivative is just four spikes (deltas) at the corners t1..t4, with signs + - - +.
SDI (sparse delta integration) places those 4 spikes, each split between its two
neighbouring samples, then integrates twice (two cumulative sums). The cost no
longer depends on the trapezoid width, which pays off for wide patches.

You will see: the 4 corner spikes, and the trapezoid rebuilt from them landing on
exactly the same samples as direct sampling. No eSDIva call here, only the idea.
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np

# %% One trapezoid (times in samples)
t1, dt_rise, dt_long = 2.3, 2.2, 5.1  # Start, ramp duration, plateau end - start.
t2, t3, t4 = t1 + dt_rise, t1 + dt_long, t1 + dt_rise + dt_long
h_max = 1.0
slope = h_max / dt_rise

n = np.arange(14)  # Sample indices.
t_fine = np.linspace(0, 13, 1000)
exact = np.interp(t_fine, [t1, t2, t3, t4], [0, h_max, h_max, 0], left=0, right=0)
sampled = np.interp(n, [t1, t2, t3, t4], [0, h_max, h_max, 0], left=0, right=0)

# %% SDI: 4 spikes, each split linearly between its two neighbouring samples
# They go one sample late, because the double cumulative sum lands one sample early.
spikes = np.zeros(n.size + 2)
for t_corner, sign in [(t1, +1), (t2, -1), (t3, -1), (t4, +1)]:
    k = int(np.floor(t_corner)) + 1
    frac = t_corner + 1 - k
    spikes[k] += sign * slope * (1 - frac)
    spikes[k + 1] += sign * slope * frac
h_sdi = np.cumsum(np.cumsum(spikes))[: n.size]  # Integrate twice.

# %% Plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.stem(np.arange(spikes.size) - 1, spikes, basefmt=" ")  # Undo the 1-sample shift.
ax1.set(ylabel="d²h/dt²", title="Second derivative: 4 corner spikes (+ - - +)")
ax2.plot(t_fine, exact, "k", label="exact trapezoid")
ax2.plot(n, sampled, "o", ms=9, mfc="none", label="direct sampling (every sample)")
ax2.plot(n, h_sdi, "x", ms=8, label="SDI (4 spikes + 2 cumulative sums)")
ax2.set(xlabel="Time (samples)", ylabel="h(t)")
ax2.legend()
plt.show()
