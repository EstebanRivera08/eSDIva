
\subsection{SDI analytic expression for Reception}
\label{appendix:SDI_pulse_echo}

The conventional formulation of Section~\ref{sec:reception} first builds the two one-way
spatial impulse responses, the emission SIR $h^{e}$ and the reception SIR $h^{r}$, and
then evaluates the received pressure (\ref{eq:received_pr}) by convolving them. The
Sparse Delta Integration (SDI) method offers an alternative route: instead of
constructing the two SIRs explicitly and convolving them, it assembles the \emph{two-way}
(pulse-echo) SIR directly from a small set of Dirac deltas. This subsection derives that
analytic expression step by step and shows that it can be evaluated in two equivalent
ways, which we call the \emph{paired SDI} form and the \emph{spectral SDI} form. \\

The pulse-echo SIR is the convolution of the emission and reception SIRs. Substituting
(\ref{eq:sir_from_deltas}) for each one-way SIR and using the fact that integration
operators commute and add ($I^2$ applied twice is $I^4$), we obtain

\begin{align}
    h_{pe}(t)
    &= h^{e}(\mathbf{r}_{m_e,p},t) \overset{t}{*}
       h^{r}(\mathbf{r}_{p,m_r},t) \\
    &= \big[I^2 D^2 h^{e}\big] \overset{t}{*} \big[I^2 D^2 h^{r}\big] \\
    &= I^4 \big[\, D^2 h^{e}(\mathbf{r}_{m_e,p},t) \overset{t}{*}
                   D^2 h^{r}(\mathbf{r}_{p,m_r},t) \,\big] \\
    &= I^4 \big[\, \Delta\delta^{e}(\mathbf{r}_{m_e,p},t) \overset{t}{*}
                   \Delta\delta^{r}(\mathbf{r}_{p,m_r},t) \,\big]
       \label{eq:conv_deltas_reception} \\
    &= I^4\, \Delta\delta_{pe}(\mathbf{r}_{m_e,p}, \mathbf{r}_{p,m_r}, t).
       \label{eq:h_pe_SDI}
\end{align}

The decisive simplification is that the convolution of two Dirac deltas is again a Dirac
delta, shifted to the sum of their positions: $\delta(t-a) \overset{t}{*} \delta(t-b) =
\delta(t-a-b)$. Applying this to the convolution in (\ref{eq:conv_deltas_reception})
turns the four emission deltas and four reception deltas into a single sparse two-way
delta train,

\begin{multline}
    \Delta\delta_{pe}(\mathbf{r}_{m_e,p}, \mathbf{r}_{p,m_r}, t) = \\
    s_{m_e,p}\, s_{p,m_r} \sum_{i=1}^{4}\sum_{j=1}^{4} \sigma_i \sigma_j\,
    \delta\!\big(t - t^e_i - t^r_j\big).
    \label{eq:delta_pe}
\end{multline}

Each emission corner ($i$) pairs with each reception corner ($j$), so a single
transmit-receive patch pair produces $4 \times 4 = 16$ pulse-echo deltas, located at the
summed corner times $t^e_i + t^r_j$ and weighted by the products of the corner slopes
$\sigma_i \sigma_j$. Equation~(\ref{eq:h_pe_SDI}) then states the central result: the
two-way SIR is the \emph{fourth} time-integral of this sparse 16-delta train. The two
double integrations of the one-way SIRs have merged into one fourth-order integration
$I^4$. \\

Inserting the two-way SIR into the received-signal equation (\ref{eq:received_pr})
gives the SDI in reception. Because $I^n$ denotes $n$-fold time integration, which in 
the Fourier domain is simply the division $\div (j\omega)^n$, two equivalent 
computational forms arise by arraging the convolution either in the time domain or in
the Fourier domain. They
produce identical results but distribute the work differently, and, as
Section~\ref{appendix:rf_heuristic} shows, they have opposite cost behaviour. We call them
the \emph{paired SDI} form (time-domain delta placement, one term per patch pair) and
the \emph{spectral SDI} form (Fourier-domain, transmit and receive contributions
separately).

\subsubsection{Paired SDI}

Using (\ref{eq:delta_pe}) and (\ref{eq:h_pe_SDI}), the received-signal contribution
(\ref{eq:RF_signal}) of a single scatterer and a single transmit-receive patch pair
($P = M_e = M_r = 1$) reads

\begin{multline}
    RF^{SDI}(p, m_e, m_r, t) = \\
    \gamma_p \left[\, \nu_{pe}(t) \overset{t}{*}
    I^4 \Delta\delta_{pe}(\mathbf{r}_{m_e,p}, \mathbf{r}_{p,m_r}, t) \right].
    \label{eq:pulseecho_SDI_paired}
\end{multline}

Here we move the fourth integration onto the excitation/impulse-response waveform,
defining the integrated pulse-echo excitation $w(t) \equiv I^4 \nu_{pe}(t)$, which is
computed once per excitation. Convolving $w$ with the 16 deltas of
(\ref{eq:delta_pe}) simply places 16 shifted, scaled copies of $w$,

\begin{multline}
    RF^{SDI}(p, m_e, m_r, t) = \\
    \gamma_p\, s_{m_e,p}\, s_{p,m_r} \sum_{i=1}^{4}\sum_{j=1}^{4} \sigma_i \sigma_j\;
    w\!\big(t - t^e_i - t^r_j\big).
    \label{eq:RF_SDI_paired}
\end{multline}

The total signal at receive element $e_{rx}$ is the sum over all scatterers and all
transmit-receive patch pairs,

\begin{equation}
    RF(e_{rx},t) = \sum_{p=1}^{P} \sum_{m_e = 1}^{M_e} \sum_{m_r = 1}^{E_{r}}
    RF^{SDI}(p, m_e, m_r, t).
    \label{eq:RF_signal_SDI_annexes}
\end{equation}

Equations~(\ref{eq:delta_pe}), (\ref{eq:pulseecho_SDI_paired}) and
(\ref{eq:RF_SDI_paired}) define the paired-delta form. For each scatterer $p$ and each
patch pair $(m_e, m_r)$ it places the $16$ deltas of (\ref{eq:delta_pe}) in continuous
time. After discretization (Section~\ref{sec:discrete_implementation}) each delta is
split between its two neighbouring samples by linear interpolation, so each patch pair
costs $16 \times 2 = 32$ sample updates. The form is called \emph{paired} because the
computation iterates over transmit-receive patch \emph{pairs}: its cost therefore grows
with the \emph{product} $M_e M_r$ of the transmit and receive patch counts.

\subsubsection{Factored form}

Using (\ref{eq:conv_deltas_reception}) instead, the same single-scatterer, single-pair
contribution is

\begin{multline}
    RF^{SDI}(p, m_e, m_r, t) = \\
    \gamma_p \left[\, \nu_{pe}(t) \overset{t}{*}
    I^4 \big[\, \Delta\delta^{e}(t) \overset{t}{*} \Delta\delta^{r}(t) \,\big] \right].
    \label{eq:pulseecho_SDI_factored}
\end{multline}

We now evaluate this in the Fourier domain. The convolutions become products and the
integration operator $I^4$ becomes the division $1/(j\omega)^4$,

\begin{equation}
    \mathcal{F}\{RF\}_{(\omega)} = \frac{\gamma_p}{(j\omega)^4}\,
    V_{(\omega)} \times \mathcal{F}\{\Delta\delta^e\}_{(\omega)}
    \times \mathcal{F}\{\Delta\delta^r\}_{(\omega)},
    \label{eq:FT_pulseecho_SDI_factored}
\end{equation}

where $V_{(\omega)} = \mathcal{F}\{\nu_{pe}\}$. The key advantage appears here: the
Fourier transform of a sparse delta train is a closed-form sum of complex exponentials,

\begin{equation}
    \mathcal{F}\{\Delta\delta^{e/r}\}_{(\omega)} = s_{p,m_{e/r}} \sum_{i=1}^{4} \sigma_i
    \exp\!\big(-j\omega t_i^{e/r}\big),
    \label{eq:FT_delta}
\end{equation}

So no time-domain sampling of the one-way SIR is needed at all. Using
(\ref{eq:TOFs1})-(\ref{eq:TOFs}) to compute the corner times $t_i^{e/r}$, equation 
(\ref{eq:FT_delta}) can be expressed as

\begin{multline}
    \mathcal{F}\{\Delta\delta^{e/r}\}_{(\omega)} = s_{p,m_{e/r}} 
    e^{-j\omega t_1^{e/r}} \\
    \left( 1 -  e^{-j\omega \Delta t_1^{e/r}} \right) 
    \left(1- e^{-j\omega \Delta t_2^{e/r}}\right) 
\end{multline}

\begin{equation}
    = -4s_{p,m_{e/r}}\sin\left(\frac{\omega\Delta t_1^{e/r}}{2}\right)
    \sin\left(\frac{\omega\Delta t_2^{e/r}}{2}\right) e^{-j \omega t_1^{e/r}}
    \label{eq:FT_delta_envelope}
\end{equation}

Equation (\ref{eq:FT_delta_envelope}) is numerically more stable than (\ref{eq:FT_delta}). When 
$\Delta t_1 = \Delta t_2 \rightarrow 0$, the $s_{p,m}$ can get big and needs more
accuracy to be sampled correctly on sums. Because the
Fourier transform is linear, the sums over the transmit patches and over the receive
patches can therefore be carried out \emph{independently} before they are multiplied.
Summing (\ref{eq:RF_signal_SDI_annexes}) over patches and points and grouping the
transmit and receive contributions gives the factored received signal,

\begin{equation}
    RF(e_{rx},t) = \mathcal{F}^{-1}\Biggl\{ \frac{\gamma_p}{(j\omega)^4}\, V_{(\omega)}
        \times \sum_{p=1}^{P}
        \Big[\, \Sigma_{TX} \times \Sigma_{RX} \,\Big] \Biggr\},
    \label{eq:RF_signal_SDI_TF_annexes}
\end{equation}

with the transmit and receive spectral sums

\begin{align}
    \Sigma_{TX} &= \sum_{m_e = 1}^{M_e} s_{p,m_e} 
    \mathcal{F}\{\Delta\delta^{e}\}_{(\omega)}
    \Sigma_{RX} &= \sum_{m_r = 1}^{E_r} s_{p,m_r} \sum_{i=1}^{4} \sigma_i
                   \exp\!\big(-j\omega t_i^{r}\big).
\end{align}

The form is called \emph{spectral} because thanks to the convolution theorem
the transmit sum $\Sigma_{TX}$ ($M_e$ patches) and the receive sum
$\Sigma_{RX}$ ($M_r$ patches) are built separately and only then multiplied. The cost
therefore grows with the \emph{sum} $M_e + M_r$ of the patch counts rather than their
product. Structurally, (\ref{eq:RF_signal_SDI_TF_annexes}) mirrors the conventional
formulation (\ref{eq:RF_conventional_signal}), with the one difference that the one-way
responses are now closed-form delta spectra (\ref{eq:FT_delta}) instead of explicitly
sampled SIRs and no numerical integration is needed.\\

The \emph{paired} and \emph{spectral} forms are mathematically identical: both compute
the same $RF(e_{rx},t)$. They differ only in \emph{where} the fourth integration and the
convolution are evaluated. The paired-delta form places $I^4 \nu_{pe}$ as shifted copies
in the time domain, looping over patch pairs and paying a cost proportional to $M_e M_r$.
The factored form performs the integration and convolution as a spectral product,
exploiting the additive separability of the two-way delay to reduce the cost to
$M_e + M_r$. Which form is cheaper depends on the patch counts, the time-record length
and the number of scatterers, as analysed next.

\subsection{Performance heuristic for SDI formulations in reception compared to conventional}
\label{appendix:rf_heuristic}

We count the cost per scatterer $p$ and per receive element $e_r$; all formulations are
assumed to scale
linearly with the scatterer count $P$, which therefore cancels in the comparison. We
compare three options: the conventional formulation, the paired SDI form
(\ref{eq:RF_SDI_paired}) and the spectral SDI form (\ref{eq:RF_signal_SDI_TF_annexes}).

\subsubsection{Conventional formulation}

Equation~(\ref{eq:RF_conventional_signal}) builds the two one-way SIRs and convolves them.
Each build is an emission-type evaluation at the cost of
(\ref{eq:heuristic_condition}),

\begin{align}
    \mathrm{CT}^{\mathrm{build}}_{tx} &= \min\!\big(M_e\,\overline{\Delta k},\;
        8M_e + 2T\big), \\
    \mathrm{CT}^{\mathrm{build}}_{rx} &= \min\!\big(M_{E_r}\,\overline{\Delta k},\;
        8M_{E_r} + 2T\big),
\end{align}

so each one-way SIR independently selects the fully-sampled trapezoid (FST) or SDI
representation according to (\ref{eq:heuristic_condition}).

The two-way response
$h_{pe}=h_{tx}\overset{t}{*}h_{E_r}$, together with the excitation and impulse-response
filtering, is evaluated in the Fourier domain as a product of length-$N_{\mathrm{fft}}$
spectra, with $N_{\mathrm{fft}}\approx T$ (next power of two). With batched forward and
inverse transforms,

\begin{multline}
    \mathrm{CT}^{\mathrm{conv}}_{\mathrm{fft}}
    =
    N_{\mathrm{tr}}^{\mathrm{conv}}
    c_{\mathrm{fft}}
    N_{\mathrm{fft}}
    \log_2 N_{\mathrm{fft}}
    \\
    \approx
    N_{\mathrm{tr}}^{\mathrm{conv}}
    c_{\mathrm{fft}}
    T
    \log_2 T,
\end{multline}

where $c_{\mathrm{fft}}$ denotes the cost of a length-$N_{\mathrm{fft}}$ FFT (including
the associated spectral multiplication), and $N_{\mathrm{tr}}^{\mathrm{conv}}$ is the
number of transforms involved in the convolution: two forward transforms for the transmit
and receive SIRs and one inverse transform for the final signal
($N_{\mathrm{tr}}^{\mathrm{conv}}\approx 3$).

The conventional cost therefore consists of two independent one-way SIR builds followed by
a transform cost that is independent of the patch counts $M_e$ and $M_{E_r}$.

\subsubsection{Paired SDI}

The paired-delta form (\ref{eq:RF_SDI_paired}) skips the construction of the one-way SIRs
and their subsequent convolution. The convolution of the corner-delta trains is performed
analytically, producing the $32$ two-way deltas of (\ref{eq:delta_pe}) directly at the
summed corner times. The four integrations and the excitation filtering collapse into a
single operator
$I^4\nu_{pe}=\div(j\omega)^4\,V$,
applied once (one FFT pair per receive element) and amortized over all scatterers and
patch pairs.

The dominant cost is therefore the delta placement,

\begin{equation}
    \mathrm{CT}^{\mathrm{paired}}_{\mathrm{place}}
    =
    32\,M_e\,M_{E_r},
\end{equation}

which scales quadratically with the patch count. The paired-delta formulation therefore
replaces the conventional convolution cost by a quadratic patch-pair placement cost.

\subsubsection{Spectral SDI}

The factored form (\ref{eq:RF_signal_SDI_TF_annexes}) keeps the convolution in the Fourier
domain but constructs the transmit and receive spectral sums,
$\Sigma_{TX}$ and $\Sigma_{RX}$, independently.

To build the one-way spectra, we evaluate the closed-form delta spectra
(\ref{eq:FT_delta}) directly. Each spectrum is computed only over the $N_b$ frequency
samples contained in the excitation/impulse-response pass-band
($N_b\ll N_{\mathrm{fft}}$). With one shared inverse transform per receive element,

\begin{multline}
    \mathrm{CT}^{\mathrm{fact}}_{\mathrm{analytic}}
    =
    4(M_e+M_{E_r})N_b
    \\
    +
    \underbrace{1}_{\text{inverse only}}
    c_{\mathrm{fft}}
    N_{\mathrm{fft}}
    \log_2 N_{\mathrm{fft}} .
\end{multline}

This implementation is exact (no interpolation) and never computes a sampled SIR. Its
spectrum build therefore scales as

\begin{equation}
    \mathcal{O}\!\big((M_e+M_{E_r})N_b\big).
\end{equation}

It is competitive only because band-limiting keeps $N_b$ small; for a near-delta
(wideband) excitation, $N_b\rightarrow N_{\mathrm{fft}}$ and the build cost degrades
accordingly.

\paragraph{Sharing the transmit sum across channels.}
The transmit spectral sum $\Sigma_{TX}$ depends only on the transmit aperture and the
scatterer position, not on which receive channel is being formed. It is therefore built
\emph{once} per scatterer and reused for all $E_r$ receive channels; only the receive sum
$\Sigma_{RX}$ is rebuilt per channel. The transmit build cost $4M_eN_b$ is thus shared
across the channels, leaving a per-channel cost dominated by the receive term
$4M_{E_r}N_b$. In practice all channels are produced in a single pass that keeps
$\Sigma_{TX}$ in fast memory and never stores the full per-scatterer spectra; removing this
memory traffic is what makes the method efficient at high scatterer counts.

\paragraph{Keeping the band size small with depth binning.}
Band-limiting fixes the \emph{fraction} $\beta$ of useful frequencies, but the \emph{number}
$N_b=\beta N_{\mathrm{fft}}$ still grows with the length of the time record, and that record
must span the arrival times of \emph{all} scatterers: a deep or wide field lengthens it and
inflates $N_b$. To keep $N_b$ small, the scatterers are grouped into depth bins. Each bin
spans only a short arrival-time window, so it uses a short record (small $N_{\mathrm{fft}}$,
hence small $N_b$); the bins share a common sample grid, so their signals are simply added
back at the correct sample offset. The number of bins is chosen so that each window is only
as short as useful---shrinking it further brings no gain once it reaches the fixed length of
the excitation. Depth binning therefore ties the spectral build cost to the small, fixed
per-bin band size rather than to the overall extent of the scatterer field.

Per scatterer and receive element, the three formulations differ only in how they combine
the transmit and receive contributions:

\begin{itemize}

\item \textbf{Spectral SDI vs paired SDI.}

Using the dominant build terms,

\begin{equation}
    4(M_e+M_{E_r})N_b
    \ll
    32M_eM_{E_r},
\end{equation}

or, neglecting constant factors,

\begin{equation}
    (M_e+M_{E_r})N_b
    \ll
    M_eM_{E_r}.
\end{equation}

The spectral formulation therefore becomes increasingly advantageous as the aperture
discretization grows, whereas the paired formulation remains attractive when
$M_eM_{E_r}$ is small or when the transmit and receive paths cannot be factorized (for
example, if attenuation depends explicitly on the transmit--receive patch pair).

\item \textbf{Spectral SDI vs conventional.}

The conventional formulation builds and samples the transmit and receive SIRs before
applying forward and inverse FFTs. In contrast, the spectral SDI formulation evaluates
the analytic spectrum directly over the occupied frequency band and requires only a single
inverse transform.

The spectral formulation is therefore advantageous when

\begin{multline}
    4(M_e+M_{E_r})N_b
    +
    c_{\mathrm{fft}}T\log_2 T
    \ll
    \\
    8(M_e+M_{E_r})
    +
    4T
    +
    2c_{\mathrm{fft}}T\log_2 T.
    \label{eq:heuristic_spectral}
\end{multline}

The SDI-build terms are typically secondary compared with the transform costs. Neglecting
the common inverse FFT yields

\begin{equation}
    4(M_e+M_{E_r})N_b
    \ll
    c_{\mathrm{fft}}T\log_2 T.
\end{equation}

Introducing the fractional bandwidth

\begin{equation}
    \beta
    =
    \frac{N_b}{N_{\mathrm{fft}}},
\end{equation}

and using $N_{\mathrm{fft}}\approx T$ gives

\begin{equation}
    4(M_e+M_{E_r})\beta
    \ll
    c_{\mathrm{fft}}\log_2 T.
    \label{eq:spectral_clean}
\end{equation}

Equation~(\ref{eq:spectral_clean}) shows that the dominant parameter is the fractional
bandwidth $\beta$. The patch count enters only through the product
$(M_e+M_{E_r})\beta$, whereas $T$ contributes only through the slowly varying logarithmic
term. Consequently, narrowband excitations ($\beta\ll1$) strongly favor the spectral
formulation, whereas broadband or near-delta excitations ($\beta\rightarrow1$) reduce its
advantage and eventually favor the conventional approach.

\end{itemize}

In summary, the paired and spectral SDI formulations eliminate the explicit construction
and convolution of sampled one-way SIRs, but they do so differently. The paired
formulation replaces the convolution by a quadratic patch-pair interaction and is
advantageous when $M_eM_{E_r}$ remains small or when transmit and receive paths cannot be
factorized. The spectral formulation preserves linear scaling with patch count through
separate transmit and receive sums and benefits from band-limiting through the factor
$N_b$. Its principal advantages are the exact analytic representation, support for
path-dependent attenuation, and a computation pattern that is particularly amenable to GPU
implementation. Together with the shared transmit sum and depth binning, which keep both
the per-channel work and the band size small regardless of the field extent, these
properties make the spectral formulation the most efficient option for large apertures and
high scatterer counts, where it outperforms the conventional convolution.

