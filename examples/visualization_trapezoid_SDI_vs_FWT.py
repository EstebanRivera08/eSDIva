import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Slider

# plt.rcParams["figure.figsize"] = [10, 6]
# plt.style.use("seaborn-v0_8")
plt.style.use("default")
plt.rcParams.update({"font.size": 18})
# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.weight"] = "bold"


# Characteristics
def plot_trapezoid_methods(
    area, l_c, Dt1, Dt2, shift, range_Dt1, range_Dt2, range_shift
):
    N_continue = 1000
    N_FST = 11
    N_derivative = 11

    t0 = 0  # Start time (us)
    t_max = 10  # Maximum time (us)

    t_continue = np.linspace(t0, t_max, N_continue)  # Upsampled time points
    t_FST = np.linspace(t0, t_max, N_FST)  # Original time points
    t_derivative = np.linspace(t0, t_max, N_derivative)  # Original time points
    f_s_FST = (N_FST - 1) / (t_max - t0)
    f_s_derivative = (N_derivative - 1) / (t_max - t0)

    def compute_trapezoid(Dt1, Dt2, shift):
        t1 = l_c + shift - (Dt1 + Dt2) / 2
        t2 = t1 + Dt1
        t3 = t1 + Dt2
        t4 = t1 + Dt1 + Dt2

        hmax = area / Dt2
        s1 = hmax / Dt1

        h_continue = np.zeros(N_continue)
        h_FST = np.zeros(N_FST)

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

        # Compute FST version
        for i in range(N_FST):
            if t_FST[i] < t1 or t_FST[i] > t4:
                continue
            elif t_FST[i] < t2:
                h_FST[i] = s1 * (t_FST[i] - t1)
            elif t_FST[i] < t3:
                h_FST[i] = hmax
            else:
                h_FST[i] = s1 * (t4 - t_FST[i])

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

        return times, h_continue, h_FST, d2h, dh, h_derivative

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(13, 5))
    plt.subplots_adjust(bottom=0.3)
    plt.subplots_adjust(right=0.8)
    # plt.subplots_adjust(left=0.2)

    times, h_continue, h_FST, d2h, dh, h_derivative = compute_trapezoid(
        Dt1, Dt2, shift
    )
    t1, t2, t3, t4 = times
    # Initial plot
    (line_continue,) = ax.plot(
        t_continue, h_continue, "-", label=r"$h_{continue}$", color="k"
    )
    vline_t1 = ax.axvline(t1, color="gray", linestyle="--", linewidth=1.5)
    vline_t2 = ax.axvline(t2, color="gray", linestyle="--", linewidth=1.5)
    vline_t3 = ax.axvline(t3, color="gray", linestyle="--", linewidth=1.5)
    vline_t4 = ax.axvline(t4, color="gray", linestyle="--", linewidth=1.5)
    (line_h_FST,) = ax.plot(
        t_FST, h_FST, "s", label=r"$h_{FST}$", color="g", ms=10, zorder=7
    )
    stem_container = ax.stem(
        t_derivative,
        d2h,
        linefmt="#dd9e00ff",
        markerfmt="s",
        basefmt=" ",
        label=r"$\partial^2 h / \partial t^2$",
    )
    stem_container[1].set_linewidth(2.5)  # vertical lines
    stem_container[0].set_markersize(7)  # marker squares (optional)
    # stem_container[0].set_color("#dd9e00ff")  # Colors the markers

    (line_h_derivative,) = ax.plot(
        t_derivative,
        h_derivative,
        "d",
        label=r"$h_{SDI}$",
        color="r",
        ms=8,
        zorder=8,
    )
    (line_dh,) = ax.plot(
        t_derivative,
        dh,
        "b:s",
        label=r"$\partial h / \partial t$",
        ms=6,
        zorder=5,
        linewidth=1,
    )
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("$ t $  (samples)")
    ax.set_ylabel("$h_{sir}$ (a.u.) ")
    ax.set_xlim(t0, t_max)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    for text in leg.get_texts():
        text.set_fontweight("bold")
        text.set_fontsize(15)
    # ax.grid("minor")
    ax.hlines(0, t0, t_max, colors="lightgray", linestyles="-", linewidth=1)
    ax.set_xticks(range(0, 11))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Store text and arrow objects for updating
    text_annotations = {}
    arrow_annotations = {}

    def add_annotations(t1, t2, t3, t4):
        """Create or update text labels for time points."""
        y_text = -0.4

        for key, t, label in [
            ("t1", t1, r"$t_1$"),
            ("t2", t2, r"$t_2$"),
            ("t3", t3, r"$t_3$"),
            ("t4", t4, r"$t_4$"),
        ]:
            if key not in text_annotations:
                text_annotations[key] = ax.text(
                    t + 0.25, y_text, label, ha="center", fontsize=18, color="k"
                )
            else:
                text_annotations[key].set_position((t + 0.25, y_text))
                text_annotations[key].set_text(label)

    # Initial annotations
    add_annotations(t1, t2, t3, t4)

    # Function to update the plot
    def update_plot(Dt1, Dt2, shift):
        times, h_continue, h_FST, d2h, dh, h_derivative = compute_trapezoid(
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
        line_h_FST.set_data(t_FST, h_FST)

        # # Update arrows and labels
        # y_arrow_dt1 = 0.75
        # arrow_annotations["dt1_arrow"].set_positions(
        #     (t1, y_arrow_dt1), (t2, y_arrow_dt1)
        # )
        # arrow_annotations["dt1_arrow_2"].set_positions(
        #     (t3, y_arrow_dt1), (t4, y_arrow_dt1)
        # )
        # text_annotations["dt1_label"].set_position(((t1 + t2) / 2, y_arrow_dt1 + 0.08))

        # y_arrow_dt2 = -0.35
        # arrow_annotations["dt2_arrow"].set_positions(
        #     (t1, y_arrow_dt2), (t3, y_arrow_dt2)
        # )
        # text_annotations["dt2_label"].set_position(((t1 + t3) / 2, y_arrow_dt2 + 0.08))

        # Update annotations
        y_text = -0.4
        text_annotations["t1"].set_position((t1 + 0.25, y_text))
        text_annotations["t2"].set_position((t2 + 0.25, y_text))
        text_annotations["t3"].set_position((t3 + 0.25, y_text))
        text_annotations["t4"].set_position((t4 + 0.25, y_text))

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw_idle()

    # Add sliders
    axcolor = "lightgoldenrodyellow"
    ax_Dt1 = plt.axes([0.25, 0.06, 0.4, 0.02], facecolor=axcolor)  # Reduced height
    ax_Dt2 = plt.axes([0.25, 0.01, 0.4, 0.02], facecolor=axcolor)  # Reduced height
    ax_shift = plt.axes([0.25, 0.11, 0.4, 0.02], facecolor=axcolor)  # Reduced height

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
    return fig


if __name__ == "__main__":
    area = 2
    l_c = 3
    Dt1 = 2
    Dt2 = 5
    shift = 1.5
    range_Dt1 = (0.5, 5.0)
    range_Dt2 = (0.5, 5.0)
    range_shift = (-2.0, 2.0)

    plot_trapezoid_methods(
        area, l_c, Dt1, Dt2, shift, range_Dt1, range_Dt2, range_shift
    )
