import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# plt.rcParams["figure.figsize"] = [10, 6]
# plt.rcParams.update({"font.size": 20})
# plt.style.use("seaborn-v0_8")
# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.weight"] = "bold"

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


# Characteristics
def plot_trapezoid_methods(
    area, l_c, Dt1, Dt2, shift, range_Dt1, range_Dt2, range_shift
):
    def compute_trapezoid(Dt1, Dt2, shift):
        t1 = l_c + shift - (Dt1 + Dt2) / 2
        t2 = t1 + Dt1
        t3 = t1 + Dt2
        t4 = t1 + Dt1 + Dt2

        hmax = area / Dt2
        s1 = hmax / Dt1

        h_continue = np.zeros(N_continue)
        h_naive = np.zeros(N_naive)

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

        times = (t1, t2, t3, t4)

        return times, h_continue, h_naive, d2h, dh, h_derivative

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(11, 5))
    plt.subplots_adjust(bottom=0.3)
    plt.subplots_adjust(right=0.8)

    times, h_continue, h_naive, d2h, dh, h_derivative = compute_trapezoid(
        Dt1, Dt2, shift
    )
    t1, t2, t3, t4 = times
    # Initial plot
    (line_continue,) = ax.plot(
        t_continue, h_continue, "-", label=r"$h_{continue}$", color="k"
    )
    vline_t1 = ax.axvline(t1, color="gray", linestyle="--", label=r"$t_1$")
    vline_t2 = ax.axvline(t2, color="gray", linestyle="--", label=r"$t_2$")
    vline_t3 = ax.axvline(t3, color="gray", linestyle="--", label=r"$t_3$")
    vline_t4 = ax.axvline(t4, color="gray", linestyle="--", label=r"$t_4$")
    (line_h_naive,) = ax.plot(
        t_naive, h_naive, "o", label=r"$h_{naive}$", color="r", ms=5, zorder=8
    )
    stem_container = ax.stem(
        t_derivative,
        d2h,
        linefmt="g-",
        markerfmt="gs",
        basefmt=" ",
        label=r"$\partial^2 h / \partial t^2$",
    )
    (line_h_derivative,) = ax.plot(
        t_derivative, h_derivative, "s", label=r"$h_{SDI}$", color="b"
    )
    (line_dh,) = ax.plot(
        t_derivative, dh, "y--s", label=r"$\partial h / \partial t$", ms=4, zorder=5
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    leg = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    for text in leg.get_texts():
        text.set_fontweight("bold")
        text.set_fontsize(15)
    ax.grid("minor")

    # Function to update the plot
    def update_plot(Dt1, Dt2, shift):
        times, h_continue, h_naive, d2h, dh, h_derivative = compute_trapezoid(
            Dt1, Dt2, shift
        )
        t1, t2, t3, t4 = times
        # Update plot data
        line_continue.set_data(t_continue, h_continue)
        vline_t1.set_xdata([t1, t1])
        vline_t2.set_xdata([t2, t2])
        vline_t3.set_xdata([t3, t3])
        vline_t4.set_xdata([t4, t4])
        # Update stem plot components
        stem_container[0].set_data(t_derivative, d2h)  # Markers
        stem_container[1].set_segments(
            [[[x, 0], [x, y]] for x, y in zip(t_derivative, d2h)]
        )  # Vertical lines

        line_dh.set_data(t_derivative, dh)
        line_h_derivative.set_data(t_derivative, h_derivative)
        line_h_naive.set_data(t_naive, h_naive)

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw_idle()

    # Add sliders
    axcolor = "lightgoldenrodyellow"
    ax_Dt1 = plt.axes([0.25, 0.1, 0.4, 0.02], facecolor=axcolor)  # Reduced height
    ax_Dt2 = plt.axes([0.25, 0.05, 0.4, 0.02], facecolor=axcolor)  # Reduced height
    ax_shift = plt.axes([0.25, 0.15, 0.4, 0.02], facecolor=axcolor)  # Reduced height

    color = "black"
    track_color = "lightgray"
    handle_style = {"edgecolor": "black", "facecolor": "darkgray"}

    slider_Dt1 = Slider(
        ax_Dt1,
        r"$\Delta t1$",
        range_Dt1[0],
        range_Dt1[1],
        valinit=Dt1,
        valstep=0.05,
        color=color,
        track_color=track_color,
        handle_style=handle_style,
    )
    slider_Dt2 = Slider(
        ax_Dt2,
        r"$\Delta t2$",
        range_Dt2[0],
        range_Dt2[1],
        valinit=Dt2,
        valstep=0.05,
        color=color,
        track_color=track_color,
        handle_style=handle_style,
    )
    slider_shift = Slider(
        ax_shift,
        "Offset",
        range_shift[0],
        range_shift[1],
        valinit=shift,
        valstep=0.1,
        color=color,
        track_color=track_color,
        handle_style=handle_style,
    )

    fontsize_widgets = 12
    slider_Dt1.label.set_fontsize(fontsize_widgets)
    slider_Dt2.label.set_fontsize(fontsize_widgets)
    slider_shift.label.set_fontsize(fontsize_widgets)
    slider_Dt1.valtext.set_fontsize(fontsize_widgets)
    slider_Dt2.valtext.set_fontsize(fontsize_widgets)
    slider_shift.valtext.set_fontsize(fontsize_widgets)

    # Update the plot when sliders are changed
    def update(val):
        Dt1 = slider_Dt1.val
        Dt2 = slider_Dt2.val
        shift = slider_shift.val
        update_plot(Dt1, Dt2, shift)

    slider_Dt1.on_changed(update)
    slider_Dt2.on_changed(update)
    slider_shift.on_changed(update)

    plt.show()
