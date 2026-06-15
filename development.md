

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
        \kappa\,N_{\mathrm{fft}}\log_2 N_{\mathrm{fft}}
    \approx \kappa\,T\log_2 T,
\end{equation}

where $\kappa$ collects the (constant number of) transforms and spectral multiplies. This
convolution cost is \emph{independent} of the patch counts $M_e, M_{E_r}$. The
conventional cost is thus a linear one-way build plus a patch-independent transform.

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
per-scatterer convolution for this quadratic placement.

\paragraph{Factored SDI form.}
The factored form (\ref{eq:RF_signal_SDI_TF_annexes}) keeps the convolution in the
Fourier domain but exploits the additive separability of the two-way delay
$t_i^e + t_j^r$ to build the transmit and receive spectral sums $\Sigma_{TX}$,
$\Sigma_{RX}$ independently. The $16$ pairs never appear together: each one-way delta
train is placed once (or summed analytically), so the placement cost is

\begin{equation}
    \mathrm{CT}^{\mathrm{fact}}_{\mathrm{place}} = 8\,(M_e + M_{E_r}),
\end{equation}

\emph{linear} in the patch count, followed by the same single $I^4$/excitation transform
$\kappa\,T\log_2 T$ as the conventional formulation. Structurally the factored form
recovers the conventional cost (linear build $+$ transform), but it never materialises the
sampled one-way SIRs: it uses the closed-form delta spectra (\ref{eq:FT_delta}) instead.

\paragraph{Comparison.}
Per scatterer and element, the three options differ only in how they pay for combining the
transmit and receive sides:

\begin{itemize}
    \item \textbf{Paired-delta vs conventional.} The paired-delta form replaces the
    patch-independent convolution $\kappa\,T\log_2 T$ by the patch-quadratic placement
    $16\,M_e M_{E_r}$. It is cheaper iff
    \begin{equation}
        16\,M_e\,M_{E_r} \;\ll\; \kappa\,T\log_2 T,
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
    patch-independent transform $\kappa\,T\log_2 T$, so they are asymptotically
    equivalent; the factored form simply avoids sampling and storing the one-way SIR
    arrays.
\end{itemize}

In summary, the analytic SDI fuses the SIR build and the convolution into a delta
product. The paired-delta form pays a quadratic patch-pair price to avoid the transform
entirely and wins for compact, few-patch or single-point problems
(\ref{eq:rf_heuristic}); the factored form restores the linear patch cost by separating
the transmit and receive sums and is the method of choice for large arrays, where
$M_e M_{E_r}$ would otherwise exceed the $T\log_2 T$ convolution cost.
