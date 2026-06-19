

\subsection{SDI analytic expression for the pulse-echo SIR}
\label{appendix:SDI_pulse_echo}

The conventional formulation of Section~\ref{sec:reception} first builds the two one-way
spatial impulse responses, the emission SIR $h^{e}$ and the reception SIR $h^{r}$, and
then evaluates the received pressure (\ref{eq:received_pr}) by convolving them. The
Sparse Delta Integration (SDI) method offers an alternative route: instead of
constructing the two SIRs explicitly and convolving them, it assembles the \emph{two-way}
(pulse-echo) SIR directly from a small set of Dirac deltas. This subsection derives that
analytic expression step by step and shows that it can be evaluated in two equivalent
ways, which we call the \emph{paired-delta} form and the \emph{factored} form.

\paragraph{From trapezoids to corner deltas.}
The starting point is a property of the far-field SIR of a rectangular patch. Its
waveform $h_m(t)$ is a trapezoid in time, so its second time derivative collapses to four
Dirac deltas, one at each corner of the trapezoid,

\begin{equation}
    D^2 h_m(t) \;=\; \frac{d^2 h_m}{dt^2}
    \;=\; s_m \sum_{i=1}^{4} \sigma_i\, \delta(t - t_i),
    \label{eq:corner_deltas}
\end{equation}

where $t_i$ are the four corner arrival times, $\sigma_i = \mathrm{slope}\times[+1,-1,-1,+1]$
are the signed corner weights, and $s_m$ is the patch sign. We denote this sparse
four-delta train $\Delta\delta_m(t) \equiv D^2 h_m(t)$. Conversely, the SIR is recovered
from its corner deltas by integrating twice in time,

\begin{equation}
    h_m(t) \;=\; I^2\, D^2 h_m(t) \;=\; I^2 \Delta\delta_m(t),
    \label{eq:sir_from_deltas}
\end{equation}

where $I^n$ denotes $n$-fold time integration; in the Fourier domain $I^n$ is simply the
division $\div (j\omega)^n$. Equation~(\ref{eq:sir_from_deltas}) is the core idea of SDI:
a continuous SIR is stored as a handful of deltas and reconstructed by integration, which
is far cheaper than sampling the full trapezoid when the deltas are sparse.

\paragraph{Building the two-way SIR from deltas.}
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
$I^4$.

\paragraph{Two ways to evaluate the received signal.}
Inserting the two-way SIR into the received-pressure equation (\ref{eq:received_pr})
gives the SDI received signal. Because $I^4$ and the convolution can be arranged either in
the time domain or in the Fourier domain, two equivalent computational forms arise. They
produce identical results but distribute the work differently, and, as
Section~\ref{appendix:rf_heuristic} shows, they have opposite cost behaviour. We call them
the \emph{paired-delta} form (time-domain delta placement, one term per patch pair) and
the \emph{factored} form (Fourier-domain, transmit and receive contributions
factorised).

\subsubsection{Paired-delta form}

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

so no time-domain sampling of the one-way SIR is needed at all. Crucially, the two-way
delay separates additively, $t^e_i + t^r_j = t^e_i + t^r_j$, which in the frequency
domain factorises the joint exponential into a transmit factor times a receive factor:
$\exp(-j\omega(t^e_i + t^r_j)) = \exp(-j\omega t^e_i)\,\exp(-j\omega t^r_j)$. Because the
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
    \Sigma_{TX} &= \sum_{m_e = 1}^{M_e} s_{p,m_e} \sum_{i=1}^{4} \sigma_i
                   \exp\!\big(-j\omega t_i^{e}\big), \\
    \Sigma_{RX} &= \sum_{m_r = 1}^{E_r} s_{p,m_r} \sum_{i=1}^{4} \sigma_i
                   \exp\!\big(-j\omega t_i^{r}\big).
\end{align}

The form is called \emph{factored} because the $M_e M_r$ patch pairs never appear
together: the transmit sum $\Sigma_{TX}$ ($M_e$ patches) and the receive sum
$\Sigma_{RX}$ ($M_r$ patches) are built separately and only then multiplied. The cost
therefore grows with the \emph{sum} $M_e + M_r$ of the patch counts rather than their
product. Structurally, (\ref{eq:RF_signal_SDI_TF_annexes}) mirrors the conventional
formulation (\ref{eq:RF_conventional_signal}), with the one difference that the one-way
responses are now closed-form delta spectra (\ref{eq:FT_delta}) instead of explicitly
sampled SIRs.

\paragraph{Equivalence.}
The paired-delta and factored forms are mathematically identical: both compute the same
$RF(e_{rx},t)$. They differ only in \emph{where} the fourth integration and the
convolution are evaluated. The paired-delta form places $I^4 \nu_{pe}$ as shifted copies
in the time domain, looping over patch pairs and paying a cost proportional to $M_e M_r$.
The factored form performs the integration and convolution as a spectral product,
exploiting the additive separability of the two-way delay to reduce the cost to
$M_e + M_r$. Which form is cheaper depends on the patch counts, the time-record length
and the number of scatterers, as analysed next.


\subsection{Heuristic inequalities for the SDI analytic forms compared to the
conventional convolution}
\label{appendix:rf_heuristic}

We count the cost per scatterer $p$ and per receive element $e_r$; all formulations scale
linearly with the scatterer count $P$, which therefore cancels in the comparison. We
compare three options: the conventional formulation, the paired-delta SDI form
(\ref{eq:RF_SDI_paired}) and the factored SDI form (\ref{eq:RF_signal_SDI_TF_annexes}).
Throughout, $T$ is the time-record length in samples, $M_e$ and $M_{E_r}$ are the
transmit and receive patch counts, and $\overline{\Delta k}$ is the mean number of
samples between corner arrivals (the SIR "spread").

\paragraph{Conventional formulation.}
Equation~(\ref{eq:RF_conventional_signal}) builds the two one-way SIRs and convolves them.
Each build is an emission-type evaluation at the cost of \S\ref{appendix:emission_cost},

\begin{align}
    \mathrm{CT}^{\mathrm{build}}_{tx} &= \min\!\big(M_e\,\overline{\Delta k},\;
        8M_e + 2T\big), \\
    \mathrm{CT}^{\mathrm{build}}_{rx} &= \min\!\big(M_{E_r}\,\overline{\Delta k},\;
        8M_{E_r} + 2T\big),
\end{align}

so each one-way SIR independently selects the fully-sampled trapezoid (FST) or SDI by
(\ref{eq:heuristic_condition}). The two-way response $h_{pe} = h_{tx}\overset{t}{*}h_{E_r}$
together with the excitation/impulse-response filtering is then evaluated as a product of
length-$N_{\mathrm{fft}}$ spectra, with $N_{\mathrm{fft}}\approx T$ (next power of two).
With batched forward and inverse transforms,

\begin{equation}
    \mathrm{CT}^{\mathrm{conv}}_{\mathrm{fft}} =
        N_{\mathrm{tr}}^{\mathrm{conv}}\,c_{\mathrm{fft}}\,N_{\mathrm{fft}}\log_2 N_{\mathrm{fft}}
    \approx N_{\mathrm{tr}}^{\mathrm{conv}}\,c_{\mathrm{fft}}\,T\log_2 T,
    \qquad N_{\mathrm{tr}}^{\mathrm{conv}}\ge 2,
\end{equation}

where $c_{\mathrm{fft}}$ is the cost of a single length-$N_{\mathrm{fft}}$ transform (plus its
spectral multiply) and $N_{\mathrm{tr}}^{\mathrm{conv}}$ is the number of transforms the
conventional convolution pays: the \emph{forward} transforms of the two one-way SIR spectra
plus the shared \emph{inverse} ($N_{\mathrm{tr}}^{\mathrm{conv}}\ge 2$; the forward transforms
are precisely what the factored form removes). Writing the count explicitly --- rather than
hiding it in a lumped constant --- keeps it visible in the factored-vs-conventional comparison
below, where the two forms pay a \emph{different} number of transforms. This convolution cost
is \emph{independent} of the patch counts $M_e, M_{E_r}$. The conventional cost is thus a
linear one-way build plus a patch-independent transform.

\paragraph{Paired-delta SDI form.}
The paired-delta form (\ref{eq:RF_SDI_paired}) skips building the one-way SIRs and skips
the convolution that combines them. The convolution of the corner-delta trains is
performed analytically, producing the $16$ two-way deltas of (\ref{eq:delta_pe}) directly
at the summed corner times. The four integrations and the excitation filtering collapse
into a single operator $I^4\nu_{pe} = \div(j\omega)^4\, V$, applied \emph{once} (one FFT
pair per receive element) and amortised over all scatterers and pairs. The dominant cost
is therefore the delta placement, which iterates over patch pairs,

\begin{equation}
    \mathrm{CT}^{\mathrm{paired}}_{\mathrm{place}} = 16\,M_e\,M_{E_r},
\end{equation}

\emph{quadratic} in the patch count. The paired-delta form trades the conventional
per-scatterer convolution for this quadratic placement. The integration $I^4$ can be
realised two ways. \emph{(a)} Place the $16$ deltas of every pair into one time buffer and
apply $\div(j\omega)^4$ in a \emph{single} shared transform (one FFT pair per receive
element); the cost is the placement above plus $c_{\mathrm{fft}}\,T\log_2 T$. \emph{(b)} Precompute
$w = I^4\nu_{pe}$ once and convolve it with the delta train by depositing a shifted, scaled
copy of $w$ per delta; this needs no FFT but costs $\mathcal{O}(N_{\mathrm{fft}})$
\emph{per pair}, i.e.
$\mathrm{CT}^{\mathrm{splat}} = 16\,M_e\,M_{E_r}\,N_{\mathrm{fft}}$ — exact, but viable only
for very small apertures (a single-point / reference cross-check).

\paragraph{Factored SDI form.}
The factored form (\ref{eq:RF_signal_SDI_TF_annexes}) keeps the convolution in the
Fourier domain but exploits the additive separability of the two-way delay
$t_i^e + t_j^r$ to build the transmit and receive spectral sums $\Sigma_{TX}$,
$\Sigma_{RX}$ independently. The $16$ pairs never appear together. There are two ways to
build the one-way spectra, with \emph{different} costs.

\emph{(i) Placed-train realisation.} Place each one-way corner-delta train once on the
sample grid ($8$ writes per patch, with linear interpolation) and forward-transform it.
The placement is

\begin{equation}
    \mathrm{CT}^{\mathrm{fact}}_{\mathrm{place}} = 8\,(M_e + M_{E_r}),
\end{equation}

\emph{linear} in the patch count, followed by the single $I^4$/excitation transform
$1\cdot c_{\mathrm{fft}}\,T\log_2 T$ (one inverse only). This recovers the conventional cost
(linear build $+$ transform) but
inherits the two-bin interpolation error of the placement.

\emph{(ii) Closed-form (analytic) realisation} (the one implemented in PyField). Evaluate
the closed-form delta spectra (\ref{eq:FT_delta}) \emph{directly}: no time-domain
placement and \emph{no forward transform} at all. Each one-way spectrum is a non-uniform
DFT of its $4M$ corner phasors, evaluated only on the $N_b$ frequencies inside the
excitation/impulse-response pass-band (the out-of-band bins are annihilated by the shared
filter $G = I^4\,V\,\mathrm{IR}$, so they need not be formed). With one shared inverse
transform per receive element,

\begin{equation}
    \mathrm{CT}^{\mathrm{fact}}_{\mathrm{analytic}}
        = 4\,(M_e + M_{E_r})\,N_b
        \;+\; \underbrace{1}_{\text{inverse only}}\cdot c_{\mathrm{fft}}\,
          N_{\mathrm{fft}}\log_2 N_{\mathrm{fft}},
    \qquad N_b \ll N_{\mathrm{fft}}.
\end{equation}

This realisation is \emph{exact} (no interpolation) and never materialises a sampled SIR.
Its spectrum build is $\mathcal{O}\!\big((M_e+M_{E_r})\,N_b\big)$ — a non-uniform DFT:
linear in the patch count but carrying the per-bin factor $N_b$. It is competitive only
because band-limiting makes $N_b$ small; for a near-delta (wideband) excitation
$N_b \to N_{\mathrm{fft}}$ and the build degrades to
$\mathcal{O}\!\big((M_e+M_{E_r})\,N_{\mathrm{fft}}\big)$, i.e. the off-grid DFT cost the
FFT is designed to avoid.

\paragraph{Comparison.}
Per scatterer and element, the three options differ only in how they pay for combining the
transmit and receive sides:

\begin{itemize}
    \item \textbf{Paired-delta vs conventional.} The paired-delta form replaces the
    patch-independent convolution $c_{\mathrm{fft}}\,T\log_2 T$ by the patch-quadratic placement
    $16\,M_e M_{E_r}$. It is cheaper iff
    \begin{equation}
        16\,M_e\,M_{E_r} \;\ll\; c_{\mathrm{fft}}\,T\log_2 T,
        \tag{\ref{eq:rf_heuristic}}
    \end{equation}
    which holds for compact apertures, few patches, and single-point responses
    (point-spread functions or mono-element transducers), where the $16\,M_e M_{E_r}$
    placement is negligible against the transform.

    \item \textbf{Factored vs paired-delta.} Both avoid materialising the one-way SIRs,
    but the factored placement $8(M_e + M_{E_r})$ beats the paired placement
    $16\,M_e M_{E_r}$ whenever
    \begin{equation}
        M_e + M_{E_r} \;\ll\; 2\,M_e\,M_{E_r},
    \end{equation}
    i.e. as soon as both apertures have more than one patch. The factored form is thus
    the better default for extended apertures; the paired-delta form is preferable only
    when $M_e M_{E_r}$ is small, or when the two-way path cannot be separated (for
    example when the attenuation depends on the full transmit-receive path and
    $\Sigma_{TX}$, $\Sigma_{RX}$ no longer factorise).

    \item \textbf{Factored vs conventional.} Both scale as a linear patch build plus a
    patch-independent transform, and the factored form avoids sampling and storing the
    one-way SIR arrays, is exact, and additionally removes the \emph{forward} transform,
    replacing it with the band-limited DFT $4(M_e+M_{E_r})\,N_b$. The comparison is
    \emph{not} purely asymptotic, however: it depends on the scatterer count $P$. The
    conventional formulation groups the $P$ scatterers into depth bins and amortises the
    one-way build across all scatterers in a bin, so its build is \emph{sublinear} in $P$,
    whereas the factored build is strictly \emph{linear} in $P$. The two therefore cross
    over at a problem-dependent $P^{*}$ (Sec.~\ref{appendix:rf_resources}): below it the
    factored form is faster (it pays no forward FFT), above it the depth-binned conventional
    form pulls ahead. On CPU the factored form's robust advantages are thus exactness (no
    interpolation) and cheap per-path attenuation; its decisive speed-up is expected on GPU
    (Sec.~\ref{appendix:rf_resources}) rather than on CPU.
\end{itemize}

\paragraph{Equal-$P$ comparison and the role of the bandwidth $N_b$.}
If the conventional form is \emph{not} depth-binned --- every scatterer builds its own
one-way SIRs --- both forms are linear in the scatterer count $P$, the factor $P$ cancels,
and the comparison reduces to the per-scatterer, per-element cost. Keeping \emph{every} term
--- the conventional SDI builds $8M_e+2T$ and $8M_{E_r}+2T$ of the two one-way SIRs and its
forward$+$inverse transform pair ($N_{\mathrm{tr}}^{\mathrm{conv}}=2$), against the spectral
band DFT and its single inverse --- the spectral form is cheaper iff
\begin{equation}
    \underbrace{4\,(M_e+M_{E_r})\,N_b}_{\text{spectral band DFT}}
    + \underbrace{c_{\mathrm{fft}}\,T\log_2 T}_{\text{1 inverse}}
    \;<\;
    \underbrace{8\,(M_e+M_{E_r}) + 4T}_{\text{2 SDI builds}}
    + \underbrace{N_{\mathrm{tr}}^{\mathrm{conv}}\,c_{\mathrm{fft}}\,T\log_2 T}_{\text{forward $+$ inverse}}.
\end{equation}
Isolating the in-band sample count $N_b$, with $M\equiv M_e+M_{E_r}$,
\begin{equation}
    N_b \;<\; 2 \;+\; \frac{4T + (N_{\mathrm{tr}}^{\mathrm{conv}}-1)\,c_{\mathrm{fft}}\,T\log_2 T}{4\,M}.
    \label{eq:heuristic_spectral}
\end{equation}
Two effects govern this bound, and together they explain the benchmark below. \emph{(i) The
patch count $M$ nearly cancels.} Both sides carry a term \emph{linear} in $M$ --- the spectral
band DFT $4MN_b$ on the left, the conventional sampling build $8M$ on the right --- so once $M$
is large these dominate their respective ($M$-independent) transform terms and the cost
\emph{ratio} becomes almost independent of $M$. Crucially, this is only true because the
conventional build $8M$ was \emph{kept}: had it been dropped, the left side would scale with
$M$ while the right did not, predicting a spurious $M$-driven blow-up of the spectral form.
\emph{(ii) The bandwidth $N_b$ sets the level.} What survives the cancellation is $N_b$: a
narrow band keeps $4MN_b$ below the build plus the saved forward transform and the spectral
form wins; a wide (near-delta) excitation, $N_b\to N_{\mathrm{fft}}$, inflates the band DFT and
the inequality flips.

This is exactly what a CPU sweep shows. Using the same 1-D array for transmit and receive,
$P=200$ scatterers, and $4\times6$ sub-patches per element (so $M=24\,n_{\mathrm{el}}$), the
spectral/conventional wall-clock ratio across a $16\times$ range of $M$ is:

\begin{center}
\begin{tabular}{r r r}
\hline
$M$ & spectral/conv (narrowband, small $N_b$) & spectral/conv (wideband, $N_b\!\to\!N_{\mathrm{fft}}$) \\
\hline
$384$  & $0.61$ & $1.29$ \\
$768$  & $0.61$ & $1.27$ \\
$1536$ & $0.56$ & $1.26$ \\
$3072$ & $0.60$ & $1.22$ \\
$6144$ & $0.61$ & $1.35$ \\
\hline
\end{tabular}
\end{center}

As predicted by (\ref{eq:heuristic_spectral}) the ratio is essentially \emph{flat} in $M$ ---
both forms scale linearly in $M$, so it cancels --- while the \emph{bandwidth} sets its level:
$\approx0.6$ (spectral $\sim1.7\times$ faster) for the band-limited tone burst, and
$\approx1.3$ (spectral $\sim1.3\times$ slower) for the near-delta excitation. The two outputs
are numerically identical (correlation $\geq0.9999$ throughout), so this is a pure cost
trade, not an accuracy one. Practical rule: prefer the spectral form for band-limited
excitations and the conventional form for wideband/near-delta pulses --- the patch count
itself barely shifts the balance. (Separately, depth-binning makes the conventional build
\emph{sublinear} in $P$; that is what creates the scatterer-count crossover $P^{*}$ of the
next subsection, an effect orthogonal to the bandwidth trade described here.)

In summary, the analytic SDI fuses the SIR build and the convolution into a delta
product. The paired-delta form pays a quadratic patch-pair price to avoid the transform
entirely and wins for compact, few-patch or single-point problems
(\ref{eq:rf_heuristic}); the factored form keeps the cost linear in the patch count by
separating the transmit and receive sums (so it never suffers the $M_e M_{E_r}$ blow-up of
the paired form on large arrays), and the closed-form realisation makes it exact and
band-limited. On CPU it is competitive with — not decisively faster than — the
depth-binned conventional formulation; its principal gains are exactness, per-path
attenuation, and a dense, scatter-free arithmetic structure well suited to GPU
acceleration, where the absence of any forward FFT and of irregular memory access is
expected to be decisive.


\subsection{Resource considerations: scatterer count, memory, and hardware}
\label{appendix:rf_resources}

The per-scatterer, per-element cost counts of Sec.~\ref{appendix:rf_heuristic} settle the
patch-count scaling but deliberately cancel the scatterer count $P$. That cancellation is
only valid when every formulation is linear in $P$, which is \emph{not} the case once the
conventional form is depth-binned. This subsection completes the picture along the three
axes a practitioner actually trades — scatterer count, memory, and target hardware —
comparing the conventional formulation against the factored SDI form (exposed in the
implementation as \texttt{method="spectral"}).

\paragraph{Scatterer-count scaling and the crossover $P^{*}$.}
The factored form evaluates each scatterer's one-way spectra independently, so its build is
strictly linear, $\mathcal{O}\!\big(P\,(M_e+M_{E_r})\,N_b\big)$. The conventional form groups
the $P$ scatterers into $B$ depth bins and builds each one-way SIR \emph{once per bin},
amortising the build across every scatterer that falls in the bin; its cost is
$\mathcal{O}\!\big(B\,(\text{build}+T\log T)\big)$ plus an $\mathcal{O}(P)$ accumulation, i.e.
\emph{sublinear} in $P$ until the bins saturate. Consequently the two cross over at a
problem-dependent $P^{*}$: for $P<P^{*}$ the factored form is faster (it pays no forward
transform), and for $P>P^{*}$ the conventional form pulls ahead, the gap widening until the
factored cost becomes purely build-bound. Measured on an 8-core CPU with a 128-element array
(\,$M_e=M_{E_r}=2304$ patches, $N_b\approx250$\,), summing a random scatterer cloud:

\begin{center}
\begin{tabular}{r r r r}
\hline
$P$ & conventional [s] & factored [s] & factored / conv \\
\hline
$100$    & $2.97$  & $2.34$   & $0.79$ \\
$500$    & $6.82$  & $11.4$   & $1.67$ \\
$2000$   & $18.9$  & $44.8$   & $2.37$ \\
$10000$  & $80.1$  & $222$    & $2.77$ \\
\hline
\end{tabular}
\end{center}

so $P^{*}\approx250$ here; the ratio plateaus near $2.8\times$ once the factored form is
build-bound. The agreement is unaffected (correlation $\geq 0.9997$ throughout). Practical
guidance: sparse targets, point-spread functions and calibration grids ($P\lesssim$ a few
hundred) favour the factored form; dense speckle phantoms for B-mode ($P\sim 10^3$–$10^6$)
favour the depth-binned conventional form.

\paragraph{Memory.}
The factored form's batched receive build forms an $(E_r, P, N_b)$ complex spectrum
($\Sigma_{RX}$ for every element and scatterer); at $E_r=128$, $P=10^4$, $N_b\approx250$ this
is $\sim 5$\,GB in double precision and overflows memory if formed at once. Because the
two-way response is summed over scatterers, the scatterer axis is an outer sum and is
\emph{chunk-decomposable}: the implementation tiles $P$ into chunks and accumulates the
$(E_r, N_b)$ per-element spectrum across chunks, bounding the peak working set to
$(E_r,\text{chunk},N_b)$ independently of $P$ (and, as a side effect, removing the
cache-thrashing super-linear slowdown a single giant buffer caused). The accumulation must
be done in double precision: the summed corner-delta \emph{areas} are large (reaching
$\sim\!10^{20}$ for sub-sample patches) and must cancel almost completely to leave the small
in-band signal, which single precision would erode. The conventional form never forms a
$P$-sized spectrum tensor — depth-binning holds the working set to one bin's SIRs and
transform buffers, $\mathcal{O}(B\,T)$ plus the $(E_r, T)$ output — so it is naturally
bounded. Per-scatterer (point-spread) output $(P, E_r, N_t)$ is intrinsically large for both
forms and is a property of the requested output, not of the method.

\paragraph{Vectorisation and GPU.}
The factored form is a dense, regular, scatter-free computation: each one-way spectrum is a
sum of corner phasors swept across a uniform frequency grid by a constant complex
multiplication (a tight, branch-light inner loop that vectorises cleanly), and the two-way
combine is a batched complex contraction over $(E_r, P, N_b)$ — a GEMM-like operation with
high arithmetic intensity, no irregular memory writes, and \emph{no forward FFT}. This maps
almost directly onto SIMD lanes and onto a GPU map-reduce: build $\Sigma_{TX}$ and
$\Sigma_{RX}$, contract them, and take one inverse FFT batched over elements. The
conventional form is harder to accelerate: it depends on many \emph{short} per-depth-bin
FFTs and on gather/scatter of sampled SIRs into time buffers — short batched FFTs
under-utilise wide vector units and GPUs, and the depth-bin gather/scatter is irregular.
The conventional form's CPU advantage (build amortisation by depth-binning) therefore does
not translate cleanly to massively parallel hardware. The expectation is that the CPU
crossover \emph{inverts} on GPU: for the dense-$P$ regime where the conventional form wins
on CPU, the factored form's regular dense structure and absence of a forward transform
should make it the faster path on GPU. This — together with exactness and free per-path
attenuation — is the principal motivation for the factored form, and the strongest
follow-up to implement and measure.
