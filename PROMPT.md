

\subsubsection{Receive Signals} 

Under the Born approximation - valid for weakly scattering media where multiple
scattering is neglected \cite{jensen_model_1991} - the received pressure at
position $\mathbf{r_m}$ is expressed as a surface integral of the scattered pressure
field over the aperture $S$:

\begin{equation}
    p_r(\mathbf{r_{m}}, t) = E(t) \overset{t}{*} \int_S p_s(\mathbf{r_{p,m}}, t)\, dS
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
    f_m(\mathbf{r_p}) &= \frac{\Delta\rho(\mathbf{r_p})}{\rho_0} - 2 \frac{\Delta
    c(\mathbf{r_p})}{c_0} \\
    \label{eq:f_perturbations}
    h_{pe}(\mathbf{r_{m,p}},t) &= \frac{\partial^2}{\partial t^2}
    \left(h^e(\mathbf{r}_{p,m},t) \overset{t}{*} h^r(\mathbf{r}_{m,p},t) \right) \\
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

\begin{equation}
    p_r(\mathbf{r}_{m}, t) = f_m(\mathbf{r}_p)
    \overset{\mathbf{r}}{*} v'_{pe}(t) \overset{t}{*}
    Dh_{pe}(\mathbf{r}_{m,p},t)
    \label{eq:pulseecho_SDI_reform}
\end{equation}

where \( D h_{pe} \) denotes the differentiated modified pulse-echo SIR, and
\( v'_{pe} \) the modified pulse-echo velocity waveform, defined as

\begin{align}
    Dh_{pe} &= \left(\frac{\partial
     h^{e,r}(\mathbf{r}_{m,p},t)}{\partial t} \overset{t}{*} \frac{\partial^2
h^{r,e}(\mathbf{r}_{p,m},t)}{\partial t^2}\right)
        \label{eq:diff_pulseecho_sir} \\
    v'_{pe} &=\frac{\rho_0}{2c_0^2} \left(E_m(t) \overset{t}{*} v(t)\right) 
        \label{eq:modif_pulseecho_velocity}
\end{align}

where the superscripts \(e\) and \(r\) denote the apertures used for emission and
reception, respectively. For simplicity, the emission term is expressed using the first
temporal derivative and the reception term using the second, although the order may be
interchanged if required. \\

To simplify the following development of the equivalent SDI for reception signals, 
the apertures are assumed to consist of a single
rectangular patch $M=M_e=M_r=1$, thereby avoiding the summation over patches. As shown
previously, the second temporal derivative of the trapezoidal SIR corresponds to a
sparse distribution of Dirac delta functions (\ref{eq:second_deriv_sir}), allowing the
convolution to be rewritten as

\begin{align}
    Dh_{pe}^{M=1} 
        &= \frac{\partial h^{e}(\mathbf{r}_{m_e,p},t)}{\partial t} 
        \overset{t}{*} \left( s^r_{p,m_r} \Delta \delta^r(t) \right) \\ \label{}
        &= \sum_{t_i^r \in \left\{ t_1, t_2, t_3, t_4 \right\}}
           \operatorname{sign}(t_i^r)\, s^r_{p,m_r}\,
           \frac{\partial h^{e}(\mathbf{r}_{m_e,p},t-t_i^r)}{\partial t}
       \label{eq:diff_pulseecho_sir_deltas}
\end{align}


Substituting the first-derivative formulation
(\ref{eq:first_deriv_sir}) into
(\ref{eq:diff_pulseecho_sir_deltas}) yields

\begin{equation}
    Dh_{pe}^{M=1}= \sum_{t_i^r \in \left\{ t_1, t_2, t_3, t_4 \right\}}
           \operatorname{sign}(t_i^r)\, s^r_{p,m_r}\, s^e_{m_e,p}\,
           \Delta u^e(t-t_i^r)
\label{eq:diff_pulseecho_sir_deltau}
\end{equation}

This expression can be further reduced to an integral of Dirac delta functions as
follows:

\begin{equation}
    Dh_{pe}^{M=1} = \int_{-\infty}^{t'} \zeta_{pe}(\mathbf{r}_{m_r,p}
    \mathbf{r}_{m_e,p}, t) dt'
    \label{eq:diff_pulseecho_sir_SDI_m1}
\end{equation}

With,

\begin{multline}
    \zeta_{pe}(\mathbf{r}_{m_e,p}, \mathbf{r}_{p, m_r}, t) = \\
           \sum_{t_i^r \in \left\{ t_1, t_2, t_3, t_4 \right\}}
           \operatorname{sign}(t_i^r)\, s^r_{p,m_r}\, s^e_{m_e,p}\,
           \Delta \delta^e(t-t_i^r) \\
\label{eq:zeta_SDI_pulseecho}
\end{multline}

Generalizing to emission and reception apertures composed of \(M_e\) and \(M_r\)
patches, respectively, (\ref{eq:diff_pulseecho_sir_SDI_m1}) becomes

\begin{equation}
    Dh_{pe} = \int_{-\infty}^{t'}  \sum_{m_r=1}^{M_r} \sum_{m_e=1}^{M_e}
    \zeta_{pe}(\mathbf{r}_{m_e,p}, \mathbf{r}_{p, m_r}, t) dt'
    \label{eq:diff_pulseecho_sir_SDI}
\end{equation}

The (\ref{eq:pulseecho_SDI_reform}), (\ref{eq:zeta_SDI_pulseecho}) and
(\ref{eq:diff_pulseecho_sir_SDI}) establish the foundation of the SDI formulation for
received signal computation. As discussed in
Section~\ref{sec:discrete_implementation}, the discrete implementation of
(\ref{eq:diff_pulseecho_sir_SDI}) corresponds to placing thirty-two temporal samples (16 delta positions × 2 discrete bins each) for each
emission--reception patch pair, followed by numerical integration at each field point.


