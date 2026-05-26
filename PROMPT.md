

\subsubsection{Receive Signals} 

Under the Born approximation - valid for weakly scattering media where multiple
scattering is neglected \cite{jensen_model_1991} - the received pressure at
position $\mathbf{r_m}$ is expressed as a surface integral of the scattered pressure
field over the aperture $S$:

\begin{equation}
    p_r(\mathbf{r_{m}}, t) = E_e(t) \overset{t}{*} \int_S p_s(\mathbf{r_{p,m}}, t)\, dS
\label{eq:received_pr_origin}
\end{equation}

where \(E_e(t)\) denotes the electromechanical impulse response of the receiving
elements. Using the scattering formalism introduced by
\cite{Angelsen1980} and \cite{jensen_model_1991}, (\ref{eq:received_pr_origin})
can be rewritten compactly as

\begin{equation}
    p_r(\mathbf{r_{m}, t}) = v_{pe}(t) \overset{t}{*} f_m(\mathbf{r_p})
    \overset{\mathbf{r}}{*} h_{pe}(\mathbf{r}_{m,p}, t)
    \label{eq:received_pr}
\end{equation}

With,

\begin{align}
    f_m(\mathbf{r}) &= \frac{\Delta\rho(\mathbf{r})}{\rho_0} - 2 \frac{\Delta c(\mathbf{r})}{c_0} \\
    \label{eq:f_perturbations}
    h_{pe}(\mathbf{r_{m,p}},t) &= \frac{\partial^2}{\partial t^2}
    \left(h(\mathbf{r}_{p,m},t) \overset{t}{*} h(\mathbf{r}_{m,p},t) \right) \\
    \label{eq:pe_hsir}
    v_{pe}(t) &= \frac{\rho_0}{2c_0^2} E_e(t) \overset{t}{*} \frac{\partial v(t)}{\partial t}
    \label{eq:pe_velocity}
\end{align}

Here, \(f_m\) is the medium scattering function, which encodes local density and
sound-speed perturbations; \(h_{pe}\) is the modified pulse-echo spatial impulse
response (SIR), relating the transducer geometry to the spatial distribution of the
scattered field; and \(v_{pe}\) is the pulse-echo velocity waveform incorporating the
transducer excitation and the electromechanical impulse responses during emission and
reception. \\

Substituting (\ref{eq:f_perturbations})--(\ref{eq:pe_velocity}) into
(\ref{eq:received_pr}) and using the associativity of convolution together with the
redistribution property of temporal derivatives over convolved terms, the received
pressure field can be expressed as:

\begin{multline}
    p_r(\mathbf{r}_{m}, t) = \frac{\rho_0}{2c_0^2} f_m(\mathbf{r}_p)
    \overset{\mathbf{r}}{*} \left(E_m(t) \overset{t}{*} v(t)\right) \overset{t}{*} \\
    \left(\frac{\partial
        h(\mathbf{r}_{m,p},t)}{\partial t} \overset{t}{*} \frac{\partial^2
        h(\mathbf{r}_{p,m},t)}{\partial t^2}\right)
\end{multline}

Furthermore, the first- and second-order temporal derivatives of the SIR imply that the
SDI method can again be truncated before the integration stage. This reduces the
computational cost associated with each omitted integration step, yielding an approximate
saving of \(\mathcal{O}(T)\) operations per truncation, where \(T\) is the temporal
sampling length. \\

In addition, the first and second derivates, as showed before corresponds to the
represenation with step functions (\ref{eq:first_deriv_sir}) and the sparse deltas
distribution (\ref{eq:second_deriv_sir}). Applying the sifting property of the Dirac
delta, \(f(t) \overset{t}{*} \delta(t - \tau) = f(t - \tau)\), to the convolution of
\(\partial h_{tx}/\partial t\) (a sum of step functions over \(M_{tx}\) transmit patches)
with \(\partial^2 h_{rx}/\partial t^2\) (a sum of weighted Dirac deltas over \(M_{rx}\)
receive patches), each delta event produces a shifted, signed copy of the step-function
representation. This yields:

\begin{multline}
    \frac{\partial h_{tx}}{\partial t}(\mathbf{r}_p, t) \overset{t}{*}
    \frac{\partial^2 h_{rx}}{\partial t^2}(\mathbf{r}_p, t) = \\
    \sum_{n=1}^{M_{rx}} \sum_{m=1}^{M_{tx}} a_n^{rx}\, s_n^{rx}\, a_m^{tx}\, s_m^{tx}\;
    \sum_{j=1}^{4} \sum_{i=1}^{4} \sigma_j\, \sigma_i\;
    u\!\left(t -
    \underbrace{(t_i^m + \tau_m^{tx})}_{\text{TX corner}} -
    \underbrace{(t_j^n + \tau_n^{rx})}_{\text{RX corner}}\right)
    \label{eq:pe_conv_steps_deltas}
\end{multline}

where \(\sigma_1 = +1,\; \sigma_2 = -1,\; \sigma_3 = -1,\; \sigma_4 = +1\) are the
corner signs, \(u(t)\) denotes the Heaviside step function, and \(a_m, s_m, \tau_m\) are
respectively the apodization weight, trapezoid slope, and beamforming delay of patch \(m\).
The result is a superposition of \(16 \times M_{tx} \times M_{rx}\) weighted step
functions — a piecewise-constant signal representing the third-order temporal derivative
of the raw pulse-echo SIR:

\begin{equation}
    \frac{\partial h_{tx}}{\partial t} \overset{t}{*}
    \frac{\partial^2 h_{rx}}{\partial t^2}
    = \frac{\partial^3}{\partial t^3}
    \left(h_{tx} \overset{t}{*} h_{rx}\right)
    = \frac{\partial\, h_{pe}}{\partial t}
    \label{eq:pe_third_deriv}
\end{equation}

This identity clarifies why the excitation \(v(t)\) appears undifferentiated in the
redistributed form: the original formulation contains three temporal derivatives in total
— one from \(v_{pe}\) via (\ref{eq:pe_velocity}) and two from \(h_{pe}\) via
(\ref{eq:pe_hsir}) — all of which are absorbed into the SIR derivative levels (1 on
\(h_{tx}\), 2 on \(h_{rx}\)).

\subsubsection{Discrete SDI Formulation for Pulse-Echo}

In the discrete-time SDI framework with sampling frequency \(f_s\) and time step
\(\Delta t = 1/f_s\), the raw SDI events for a single rectangular patch \(m\) at field
point \(\mathbf{r}_p\) are:

\begin{equation}
    d^2h_m[k] = s_m \sum_{i=1}^{4} \sigma_i
    \Big[(1 - \alpha_i)\,\delta[k - \lfloor k_i \rfloor]
    + \alpha_i\,\delta[k - \lfloor k_i \rfloor - 1]\Big]
    \label{eq:discrete_d2h}
\end{equation}

where \(k_i = (t_i - t_0)\,f_s + 1\) is the (non-integer) sample index of the \(i\)-th
corner time, \(\alpha_i = k_i - \lfloor k_i \rfloor\) is the fractional part used for
linear interpolation between adjacent samples, and \(s_m = h_{\max,m}/(t_2^m - t_1^m)\)
is the trapezoid slope. Each patch produces exactly 8 weighted sample contributions
(4 corners \(\times\) 2 interpolation weights).

The multi-patch SIR derivatives are recovered via cumulative summation:

\begin{align}
    d^2h[k] &= \sum_{m=1}^{M} d^2h_m[k]
    && \text{(raw SDI events)}
    \label{eq:d2h_sum} \\[4pt]
    dh[k] &= \sum_{n=0}^{k} d^2h[n]
    && \text{(first cumsum — step functions)}
    \label{eq:dh_cumsum} \\[4pt]
    h[k] &= \Delta t \sum_{n=0}^{k} dh[n]
    && \text{(second cumsum} \times \Delta t \text{ — full SIR)}
    \label{eq:h_cumsum}
\end{align}

For the pulse-echo kernel with the 1+2 derivative redistribution, only partial integration
is required:

\begin{align}
    dh_{tx}[k] &= \sum_{n=0}^{k} d^2h_{tx}[n]
    && \text{(1 cumsum — saves 1 vs.\ } h_{tx}\text{)}
    \label{eq:dh_tx_pe} \\[4pt]
    d^2h_{rx}[k] &= \sum_{n=1}^{M_{rx}} d^2h_n^{rx}[k]
    && \text{(0 cumsums — saves 2 vs.\ } h_{rx}\text{)}
    \label{eq:d2h_rx_pe}
\end{align}

\subsubsection{Frequency-Domain Implementation}

The received pressure for a discrete phantom of \(Q\) point scatterers at positions
\(\mathbf{r}_q\) with scattering strengths \(f_q\) is, in the frequency domain:

\begin{equation}
    P_r(\mathbf{r}_m, f) = \frac{\rho_0}{2c_0^2}
    \sum_{q=1}^{Q} f_q \;
    \underbrace{E_m(f) \cdot V(f)}_{\text{undifferentiated}}
    \cdot\;
    \underbrace{\widehat{dh}_{tx}(\mathbf{r}_q, f)
    \cdot \widehat{d^2h}_{rx}(\mathbf{r}_q, f)}_{\text{SDI pulse-echo kernel}}
    \label{eq:pr_freq}
\end{equation}

where \(\widehat{dh}_{tx}(f) = \mathcal{F}\{dh_{tx}[k]\}\) and
\(\widehat{d^2h}_{rx}(f) = \mathcal{F}\{d^2h_{rx}[k]\}\) are the DFTs of the partially
integrated SDI arrays. The three temporal derivatives (1 from the original
\(\partial v / \partial t\) in \(v_{pe}\), 2 from \(\partial^2/\partial t^2\) in
\(h_{pe}\)) are encoded in the SIR derivative levels rather than applied to the
excitation:

\begin{equation}
    \underbrace{(j2\pi f)^1 \cdot H_{tx}(f)}_{\widehat{dh}_{tx}(f)}
    \;\cdot\;
    \underbrace{(j2\pi f)^2 \cdot H_{rx}(f)}_{\widehat{d^2h}_{rx}(f)}
    = (j2\pi f)^3 \cdot H_{tx}(f) \cdot H_{rx}(f)
    \label{eq:derivative_redistribution_freq}
\end{equation}

This contrasts with the emission path, where the excitation derivative is applied in the
frequency domain as \(j2\pi f \cdot V(f)\). In the receive formulation, the excitation
appears undifferentiated and the \(j2\pi f\) factors are implicit in the SDI truncation
levels.

When attenuation is enabled, the causal power-law transfer function is applied
multiplicatively for each propagation leg:

\begin{equation}
    P_r(f) \leftarrow P_r(f)
    \cdot H_{\text{att}}(f,\, d_{tx})
    \cdot H_{\text{att}}(f,\, d_{rx})
    \label{eq:pr_attenuation}
\end{equation}

where \(d_{tx}\) and \(d_{rx}\) are the transmit and receive propagation distances.

\subsubsection{Per-Element Extension}

For per-element excitation \(v_e(t)\) or per-element attenuation (using element-center
distances for near-field accuracy), the frequency-domain accumulation extends over both
transmit elements \(e_{tx}\) and receive elements \(e_{rx}\):

\begin{equation}
    P_r(f) = \frac{\rho_0}{2c_0^2}\, F_m(f)
    \sum_{e_{tx}} \sum_{e_{rx}}
    E_{m,e_{rx}}(f)\; V_{e_{tx}}(f)\;
    \widehat{dh}_{tx,e_{tx}}(f)\;
    \widehat{d^2h}_{rx,e_{rx}}(f)\;
    H_{\text{att}}(f, d_{e_{tx}})\;
    H_{\text{att}}(f, d_{e_{rx}})
    \label{eq:pr_per_element}
\end{equation}

where \(\widehat{dh}_{tx,e_{tx}}(f)\) is the DFT of the first-derivative SIR for transmit
element \(e_{tx}\), computed by placing SDI events only for patches belonging to that
element and integrating once. Similarly, \(\widehat{d^2h}_{rx,e_{rx}}(f)\) uses only
patches from receive element \(e_{rx}\) with no integration. Accumulation in the
frequency domain before the inverse FFT ensures peak memory of
\(\mathcal{O}(P_b \times N_{\text{fft}})\) per batch, independent of the number of
elements.

\subsubsection{Optimality of the 1+2 Derivative Split}

The choice of placing 1 derivative on \(h_{tx}\) and 2 on \(h_{rx}\) (or equivalently
2+1 by symmetry) is the optimal redistribution for SDI because:

\begin{enumerate}
    \item \textbf{3+0 or 0+3 splits}: The third derivative
    \(\partial^3 h / \partial t^3\) of a trapezoid involves derivatives of Dirac deltas
    (doublets), which have no natural discrete SDI representation via sample placement.
    Meanwhile, the undifferentiated SIR \(h\) requires two full cumulative sums — no net
    savings.

    \item \textbf{1+2 or 2+1 splits}: One side uses raw SDI events (0 cumsums), the other
    uses a single cumsum. Three cumulative-sum operations are saved in total compared to
    computing both full SIRs.
\end{enumerate}

\subsubsection{Computational Savings}

Compared to computing the full SIRs \(h_{tx}\) and \(h_{rx}\), convolving them, and
differentiating twice, the SDI derivative redistribution avoids three cumulative-sum
passes:

\begin{center}
\begin{tabular}{lcc}
    \hline
    \textbf{Quantity} & \textbf{Full approach} & \textbf{SDI receive} \\
    \hline
    TX SIR & \(d^2h_{tx}
        \xrightarrow{\text{cumsum}} dh_{tx}
        \xrightarrow{\text{cumsum}\times\Delta t} h_{tx}\)
    & \(d^2h_{tx} \xrightarrow{\text{cumsum}} dh_{tx}\) \\[4pt]
    RX SIR & \(d^2h_{rx}
        \xrightarrow{\text{cumsum}} dh_{rx}
        \xrightarrow{\text{cumsum}\times\Delta t} h_{rx}\)
    & \(d^2h_{rx}\) (raw events) \\[4pt]
    \hline
    Cumsums saved & — & \(3 \times \mathcal{O}(P \times T)\) \\
    \hline
\end{tabular}
\end{center}

The total cost per scatterer (or batch of \(P_b\) scatterers) is:

\begin{enumerate}
    \item Compute \(d^2h_{tx}\) via SDI event placement:
          \(\mathcal{O}(8\,M_{tx})\) operations.
    \item Integrate once:
          \(dh_{tx} = \text{cumsum}(d^2h_{tx})\): \(\mathcal{O}(T)\).
    \item Compute \(d^2h_{rx}\) via SDI event placement:
          \(\mathcal{O}(8\,M_{rx})\) operations.
    \item FFT both arrays:
          \(\mathcal{O}(N \log N)\) where
          \(N = 2^{\lceil\log_2(T_{tx} + T_{rx} - 1)\rceil}\).
    \item Multiply all frequency-domain factors
          (\(\widehat{dh}_{tx},\; \widehat{d^2h}_{rx},\; V,\; E_m,\;
          H_{\text{att}}\)): \(\mathcal{O}(N)\).
    \item Inverse FFT to recover \(p_r(t)\): \(\mathcal{O}(N \log N)\).
\end{enumerate}

Total per scatterer: \(\mathcal{O}\!\big(8(M_{tx} + M_{rx}) + T + N\log N\big)\). The
SDI event placement (\(\mathcal{O}(8M)\)) dominates over the naive patch-filling cost
(\(\mathcal{O}(M \times \overline{\Delta k})\)) only when the average trapezoid width
\(\overline{\Delta k}\) exceeds the SDI break-even threshold, following the same heuristic
as in emission.

