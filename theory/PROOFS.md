# Proofs of the exact properties

The properties below follow from the construction of ankyra-lite's two parts (identities and elementary bounds), not
from fitted relationships; the historical estimates, the anchoring weights and the carrier's weights themselves are
fitted to the data. This page states them with proofs and counterexamples:

- **P1–P20** belong to the anchoring layer, `ankyra_lite.anchor`, a byte-identical copy of the ANKYRA 2.2.1 package.
  They are numbered in the order of its pipeline, and their statements and proofs are those of the ANKYRA 2.2.1
  theory. The data evidence of that theory (measurements from the method's earlier development and from the ANKYRA
  study, made with TimesFM 2.5 as the carrier) is not a result of this repository and is left out; where this
  repository's result files measure a consequence, the statement names the file.
- **C1–C10** belong to the carrier network, `ankyra_lite.carrier`
  ([Exact properties of the carrier network](#exact-properties-of-the-carrier-network)). They hold for any weights,
  trained or not, except C9(a), which is about the initialisation.

They make the construction auditable:

- any division of labour between two forecasters can be scored from block losses;
- every weight has a stated support requirement and a bound on how far it can move;
- every readout has a guarantee or a counterexample that marks where the guarantee stops;
- the carrier's forecast is a nonnegative energy budget built only from data before its issue time, and every
  pseudo-origin forecast that the anchoring layer scores is out of sample for the carrier.

The index, with the condition, use, code and check of each property, is in [README.md](README.md). P1–P20 are
implemented in [`operators.py`](operators.py) (or in the anchoring layer's `anchor.blocks`, `anchor.readouts` and
`anchor.metrics`) and checked in [`test_operators.py`](test_operators.py); C1–C10 are properties of the carrier's own
code and are checked in [`test_carrier_properties.py`](test_carrier_properties.py); counterexamples are included in
both. Where this repository's result files measure a property's consequence on data, the statement names the file.

P1–P10 are identities and one-step bounds. Their arguments are short and are given with the statements, not as
separate proofs; the numerical checks cover each of them.

- Unit-equal log RMS ratios are reported with 95% unit-and-month (UM) bootstrap intervals, and percentages are
  $100[1-\exp(r)]$.

Several statements are standard, and they are marked as such. None of them is an accuracy guarantee: accuracy is
measured in [results/SUMMARY.md](../results/SUMMARY.md).

## Blocks

**P1. Block identity (standard).** A 744-hour trajectory covers $D=31$ origin-aligned days of $H=24$ hours. Its level
$\ell$, centred daily path $b_d$ and within-day path $w_{d,h}$ are orthogonal projections of ranks 1, 30 and 713, so for
an error $e=F-y$

$$\frac1T\sum_{d,h}e_{d,h}^2=\ell(e)^2+\frac1D\sum_d b_d(e)^2+\frac1T\sum_{d,h}w_{d,h}(e)^2 .$$

**P2. Four blocks, and no weekly Fourier bin.** Split the within-day error into its hour-of-day mean
$\bar w_h=D^{-1}\sum_d w_{d,h}$ and the day-by-hour interaction $u_{d,h}=w_{d,h}-\bar w_h$. With $J_n=n^{-1}\mathbf 1\mathbf 1^\top$
and the day index outermost, consider the four projections

$$J_D\otimes J_H,\qquad (I-J_D)\otimes J_H,\qquad J_D\otimes(I-J_H),\qquad (I-J_D)\otimes(I-J_H).$$

They are symmetric, idempotent and mutually orthogonal, sum to the identity and have ranks 1, 30, 23 and 690, so

$$\frac1T\sum e^2=\ell^2+\frac1D\sum_d b_d^2+\frac1H\sum_h\bar w_h^2+\frac1T\sum_{d,h}u_{d,h}^2 .$$

$J_D\otimes I_H$ projects onto 24-hour periodic sequences. On the 744-hour window these are spanned by the DFT
frequencies $f$ with $31\mid f$.

A DFT bin is 168-hour periodic only if $7f/31$ is an integer. Because 7 and 31 are coprime, that again means $31\mid f$.
The window therefore has no weekly Fourier subspace separate from the daily harmonics. Weekly effects fall into the
between-day and interaction blocks, and the anchoring layer represents them by day types (as does the carrier, through
its day-type features). Day types are one modelling choice, not a mathematical necessity.

**P3. Energy is the level.** The window's energy error is $744\,\ell(e)$ kWh for loads in kW. The constant direction is
the smallest subspace that carries this energy functional.

- For one window and a positive denominator, a smaller squared level error means a smaller absolute percentage energy
  error.
- Aggregated over windows and units, the two can rank methods differently.

## Replacement

**P4. Exact replacement.** Take fixed estimates $\ell^\ast$ and a centred $b^\ast$. The replacement
$F'=\ell^\ast+b^\ast+w(F)$ changes the MSE by exactly the change in the replaced blocks' losses. Replacements with
fixed estimates are idempotent and commute.

Let $\pi_k$ be the old block shares of the MSE and $\rho_k$ the block RMS ratios. Then

$$\frac{\mathrm{MSE}'}{\mathrm{MSE}}=1+\sum_k\pi_k(\rho_k^2-1).$$

A level-only replacement can therefore reduce the MSE by at most the level share $\pi_\ell$, however accurate the new
level. Replacements are scored from block losses, so they recover no peak and no absolute loss.

The block shares $\pi_k$ differ between base forecasters and between units, so a replacement has to be checked for
each base forecaster. This repository's block shares are in [`results/block_shares.csv`](../results/block_shares.csv).

**P5. Two-source division.** Pooled block losses add. Taking each block from the source with the smaller pooled block
loss therefore gives the pooled optimum among the eight block-wise divisions of two sources.

For a single unit, let $\pi$ be the modified source's share of its own MSE in the replaced blocks, and $r$ the ratio
of the donor's squared error there to the source's. The division's log RMS ratio to the modified source is

$$\tfrac12\log\big[1+\pi(r-1)\big].$$

A gain is bounded below by $\tfrac12\log(1-\pi)$; a loss is unbounded in $r$.

*Consequence.* A fixed division can be the pooled optimum and still lose on the unit-equal estimand. A minority of
units whose historical daily means fail badly can outweigh a majority that gain.

The anchoring layer's per-unit weights and weekly handover respond to this: each unit's own record decides how much
of each source it uses.

**P6. Correction accounting (standard).** Let $e=y-\hat y$ be a reference error and $D$ a correction added to the
forecast. (This $e$ is the truth minus the forecast, the opposite sign to $e=F-y$ in P1; the sign matters for the
cross term below.) Then

$$\Delta\mathrm{MSE}=\mathbb E[D^2]-2\,\mathbb E[eD].$$

- The correction helps if and only if $2\mathbb E[eD]>\mathbb E[D^2]$.
- $\mathbb E[eD]\le0$ means it points the wrong way.
- $0<2\mathbb E[eD]<\mathbb E[D^2]$ means it points the right way but is too large.

At a single window, changing an error $r$ by $\delta$ helps exactly when $\delta(2r+\delta)<0$. The share of errors
with one sign cannot decide this on its own.

## Historical weights

**P7. Completed error support.** Pseudo-origins are spaced 744 h apart, each needs a 1,344-hour context, and there are
at most 12. A complete record of $m$ hours then yields

$$K_{\rm eff}(m)=\min\lbrace12,\lfloor(m-1344)/744\rfloor\rbrace_+$$

completed errors. Marking hours as missing can only lower this count.

The weights need two errors. The data share is therefore $K_{\rm eff}/(K_{\rm eff}+8)$ for $K_{\rm eff}\ge2$, and zero
below that (equal weights):

- 0.20 after 2,832 h (118 days);
- about 0.33, 0.47 and 0.53 after 6, 9 and 12 months;
- 0.60 from 10,272 h.

The annual candidate also needs its lag window at each pseudo-origin. Its two errors therefore need
$8{,}760+2\times744=10{,}248$ h, about 14 months: a year of observations supplies an annual prediction but not its error
record.

These are necessary support lengths, not accuracy guarantees. The counts are not independent sample sizes, because
contexts overlap. This repository's contrasts of ankyra-lite by the length of the unit's record before the origin
(strata of $K_{\rm eff}$ and of years of record; descriptive) are in
[`results/history_length.csv`](../results/history_length.csv).

**P8. Shrinkage bounds.** Let $w=(1-\rho)p+\rho q$, with prior $p$ and data weights $q$ on the same support.

- Every candidate subset $B$ satisfies $(1-\rho)p(B)\le w(B)\le(1-\rho)p(B)+\rho$.
- A combined prediction moves from the prior combination by at most $\rho R_h\,\mathrm{TV}(q,p)$, where $R_h$ is the
  range of the candidate predictions and $\mathrm{TV}(q,p)=\tfrac12\sum_j|q_j-p_j|$ is the total-variation distance
  between the two weight vectors. For a uniform prior on $m$ candidates this is at most $\rho(1-1/m)R_h$.

*Matched support* $k$ means that every candidate in the set has completed errors at the same $k$ pseudo-origins, so
that the prior and the data weights live on the same candidates. For the four non-annual level candidates of the
historical estimator with matched support $k\ge2$, $\rho=k/(k+8)$.
The total weather weight therefore lies in $[(1-\rho)/2,(1+\rho)/2]$: [0.4, 0.6] at $k=2$, [1/3, 2/3] at 4 and
[0.2, 0.8] at 12. The anchoring layer's level adds the model candidate (the carrier's forecast). Its prior weather
mass is then 3/7 when all seven candidates are supported (2/5 before the annual candidates are), and the general
bound applies with that $p(B)$.

These bounds limit movement from the prior, not forecast error.

**P9. Deletion renormalises the mixture.** Suppose only a set $R$ of candidates is available at the real origin, with
prior mass $P$ and data mass $Q$. The renormalised weights are then a mixture with

$$\rho_R=\frac{\rho Q}{(1-\rho)P+\rho Q},$$

and the bounds of P8 must be re-evaluated.

- *Counterexample.* Take the four non-annual candidates in the order recent, long-history, recent weather-adjusted,
  long-history weather-adjusted, so that the third and fourth are the weather candidates. With a uniform prior, $q$
  concentrated on the third, $\rho=0.2$ and $R=\lbrace1,3\rbrace$, $P=1/2$ and $Q=1$, so $\rho_R=1/3$. The weather
  mass becomes $(1-\rho_R)/2+\rho_R=2/3$, above the old bound 0.6.
- *Discontinuity.* The two-error threshold is a jump. In a limiting example, going from one error (equal weights) to
  two moves the weights by an L1 distance of 0.3.

**P10. Least squares versus inverse MSE under shared errors.** When two forecasts of the same load share an error
component, the two weights behave differently:

- the inverse-MSE weight $\mathrm{MSE}_a/(\mathrm{MSE}_a+\mathrm{MSE}_b)$ tends to 1/2 as the shared variance grows;
- the least-squares weight $\langle y-a,\,b-a\rangle/\Vert b-a\Vert^2$ does not depend on the shared component when
  that component is uncorrelated with the difference $b-a$: write $a=y+c+\varepsilon_a$ and $b=y+c+\varepsilon_b$; the
  numerator is $-\langle c+\varepsilon_a,\,\varepsilon_b-\varepsilon_a\rangle$, and its $c$ term vanishes exactly
  under that condition. If the shared component is correlated with the difference, it enters the weight.

The anchoring layer's weekly handover therefore uses the least-squares weight
([Bates and Granger, 1969](#references)), shrunk towards 1/2 with $K_0=2$. Two forecasts of the same unit's load
can share part of their error (for example, when both miss the same unforeseen event); the least-squares weight is
then not pulled towards 1/2.

## Nonnegativity and energy

**P11. Clipping cannot increase any hour's error (standard).** If realised load is nonnegative, then
$|\max(x_h,0)-y_h|\le|x_h-y_h|$ for every hour. Hourly MSE, MAE and every quantile of the window's error distribution
therefore cannot increase.

- The clip raises delivered energy by $\Delta=\sum_{d,h}\max(-F_{d,h},0)$, which is known at the origin.
- It is positively homogeneous, so it does not correct scale errors.
- The guarantee does not hold for net-metered loads, which can be negative.

**P12. Energy, nonnegativity and no-harm are incompatible.** No map can keep the energy readout, return nonnegative
load and guarantee no increase in error, all at once.

Two hours are enough to show it. Take the forecast $p=(-1,3)$, whose mean is 1, and the truth $y=(0,4)$.

- The nonnegative trajectories with mean 1 lie on the line $x_1+x_2=2$.
- That line is tangent at $p$ to the disc $\Vert x-y\Vert^2\le\Vert p-y\Vert^2=2$, so every feasible point is strictly worse than
  $p$.
- The best feasible point, $(0,2)$, has squared error 4. Clipping gives $(0,3)$, with error 1.

The anchoring layer therefore reads energy from the level and projects only the delivered trajectory.

**P13. Day-level projection: a conditional guarantee.** This option projects the 31 daily means onto
$\mathcal A=\lbrace x\ge0:\bar x=\max(\hat\ell,0)\rbrace$ before clipping, and keeps the within-day deviations.

*Condition.* The forecast level is nonnegative, the load is nonnegative, and the level is not under-forecast by more
than the smallest true daily mean: $\ell^\ast-\hat\ell\le\min_d m^\ast_d$.

*Guarantee.* Under the condition, the daily-mean error cannot increase, and therefore neither can the hourly MSE.

*Proof.*

1. The shifted truth $\tilde m=m^\ast+(\hat\ell-\ell^\ast)\mathbf 1$ lies in $\mathcal A$.
2. Projection onto a closed convex set moves no point farther from any member of the set.
3. On $\mathcal A$, the distance to $m^\ast$ equals the distance to $\tilde m$ plus the constant
   $31(\hat\ell-\ell^\ast)^2$.

The guarantee is relative to the unprojected trajectory, not to the clipped one.

The option is provided as `operators.daily_projection`. The anchoring layer delivers the hourly clip, and
ankyra-lite's evaluated forecasts use it.

## Peak

**P14. A mean path underestimates the expected peak (standard).** For exact conditional means, by convexity of the
maximum,

$$\max_{d,h}\mathbb E[y_{d,h}\mid\mathcal F_o]\le\mathbb E\big[\max_{d,h}y_{d,h}\mid\mathcal F_o\big].$$

Fitted forecasters are median-type summaries, not exact conditional means. The inequality therefore motivates a
separate readout; it does not guarantee that any particular correction helps.

**P15. Scalar form and bounds.** Let $\hat L_d$ be the forecast mean of day $d$, $\tau_d$ its day type and $\bar F$
the forecast window mean. Let $a_{j,h}$ be the load of context day $j$ at hour $h$ minus that day's mean,
$\mathcal J(t)$ the four most recent context days of type $t$, and $A_t=\max_{j\in\mathcal J(t)}\max_h a_{j,h}$ the
largest daily excursion among them. Because maxima commute, the hour-wise envelope equals a scalar maximum:

$$U=\max_{d,h}\big(\hat L_d+\max_{j\in\mathcal J(\tau_d)}a_{j,h}\big)=\max_d\,(\hat L_d+A_{\tau_d}).$$

- Only daily peak excursions enter, not peak timing.
- Excursions have zero day mean, so $A_t\ge0$ and $U\ge\max_d\hat L_d\ge\bar F$.
- $\hat P_\kappa=\bar F+\kappa(U-\bar F)\ge U$ for $\kappa\ge1$. The anchoring layer uses $\kappa=1$.

**P16. Peak error decomposition.** Let $M(y)$ be the realised peak of the window, $d^\ast$ the day on which it
occurs, $\hat d$ the selected day and $a_d(y)=\max_h w_{d,h}(y)$. Then

$$U-M(y)=\ell(e)+\big[b_{d^\ast}(F)-b_{d^\ast}(y)\big]+\big[A_{\tau_{d^\ast}}-a_{d^\ast}(y)\big]+\Delta_{\rm sel},\qquad \Delta_{\rm sel}\ge0,$$

and $\hat P_\kappa-M(y)=U-M(y)+(\kappa-1)(U-\bar F)$.

The forecast's level and daily path enter the peak directly, so gains in the anchored level carry into it. The
amplitude term has no universal sign, and the net error has to be measured.

**P17. Finite-support median property.** Assume:

- daily levels are constant within type;
- future daily excursions are drawn independently from each retained empirical sample;
- a type that attains the envelope has $m$ samples and $n$ future days.

If $((m-1)/m)^n<1/2$, the lower median of the future maximum equals the envelope. With four samples, three future days
of that type suffice.

Two of these conditions are shown to be needed by counterexamples:

- *the count condition:* excursions (0, 1, 2, 3) over two equal-level days ($m=4$, $n=2$, $(3/4)^2>1/2$) give a
  median of 2, not 3;
- *constant levels:* three days with levels (0, −100, −100) give a median of 1.

Independence of the draws is part of the working distribution. No counterexample is given for it here, so the
statement does not say what happens without it.

Under continuous iid sampling, the joint maximum of $m$ historical and $n$ future values is historical with probability
$m/(m+n)$. This is an unconditional sampling fact, not coverage for a particular building.

**P18. Asymmetric peak costs (standard).** Let $\bar u$ and $\bar o$ be a method's mean under- and over-forecast of the peak.
Charge $\lambda$ per kW under and 1 per kW over, normalised so that $\lambda=1$ gives the MAE. The cost is

$$\frac{2(\lambda\bar u+\bar o)}{1+\lambda}=2\big[\tau\bar u+(1-\tau)\bar o\big],\qquad \tau=\frac{\lambda}{1+\lambda},$$

which is twice the pinball loss at level $\tau$ ([Koenker and Bassett, 1978](#references)). Each method is
summarised by its pair
$(\bar u,\bar o)$, and the best method for any $\lambda$ lies on the lower envelope of the lines
$\tau\bar u+(1-\tau)\bar o$.

*Evidence for the peak operator (this repository).* The readout applied to ankyra-lite's delivered trajectory is
compared with the maximum of the same trajectory (rows `trajectory maximum`), with the previous window's observed
peak and with the same readout on the baselines' trajectories, on the test windows of every population
([`results/peak_readout.csv`](../results/peak_readout.csv); summary in [`results/SUMMARY.md`](../results/SUMMARY.md)).

## Estimands

**P19. Pooled versus unit-equal summaries.** Let unit $i$ have $n_i$ windows, reference RMS $R_i$ and RMS ratio
$\rho_i$. The two summaries weight units differently:

- the pooled MSE ratio is $\sum_i\omega_i\rho_i^2$, with $\omega_i\propto n_iR_i^2$;
- the unit-equal estimand is $N^{-1}\sum_i\log\rho_i$.

The two have opposite signs when units with large $n_iR_i^2$ have $\rho_i>1$.

Because $\rho_i$ has $R_i$ in its denominator, weights and ratios are mechanically coupled. The sign of a
disagreement therefore does not say which units benefit, and testing a size effect needs a size variable that is not
built from the errors. This is why the result files report four summaries side by side: the unit-equal log ratio
and the pooled MSE ratio (`benchmark_pairwise.csv`), the mean per-unit rank (`benchmark_mean_unit_rank.csv`) and
conventional metrics (`conventional_metrics_late.csv`).

## Evidence bound

**P20. Evidence bound for choosing among candidates.** The level block chooses, in effect, among $K$ candidate
estimates of the same quantity from their errors at $T$ completed pseudo-origins ($K=7$ with the model candidate,
i.e. the carrier, $T\le 6$). Two statements bound what any such rule can do.

- *Null benchmark (identity).* Suppose that, given the candidate errors at the $T$ pseudo-origins, the $K$ candidate
  errors at the target are exchangeable over the candidate labels (their joint distribution does not change when the
  labels are permuted) and have no ties. This is the case in which the pseudo errors carry no information on which
  candidate will be best at the target. Then every rule that picks one candidate from the pseudo errors picks the
  target's best candidate with probability exactly $1/K$, and lands in the best two with probability $2/K$. A
  measured hit rate is therefore read against $1/K$, not against zero.
- *Minimax rate (standard).* With $T$ independent error samples per candidate, the known results on aggregation
  (Nemirovski, 2000; Tsybakov, 2003; Juditsky, Rigollet and Tsybakov, 2008; Rigollet and Tsybakov, 2011) say that no
  rule can guarantee, for every distribution, an excess squared error of smaller order than
  - $\psi(K,T)=\min\lbrace K/T,\sqrt{\log K/T}\rbrace$ over the best *convex combination* of the candidates (convex
    aggregation), or
  - $\log K/T$ over the best *single* candidate (model selection),

  in units of the error variance. The two rates have different comparators. $\psi$ is written here in a simplified
  form: the published rate has a different logarithmic factor when $K$ exceeds $\sqrt T$, and the constants are not
  specified. For $K=7$ and $T=6$ the simplified expressions give $\psi=0.57$ and $\log K/T=0.32$ (for $T=30$: $0.23$
  and $0.06$). These are orders of magnitude, not values of a bound, and they assume independence, which
  pseudo-origins only approximate. They are worst-case orders, not predictions for these data; they indicate that
  with a few months of evidence per candidate strongly shrunk weights are the reasonable choice, which is what the
  weights of P8 do. (The historical candidates use up to 12 pseudo-origins, the carrier candidate at most 6.)

No hit rate of the candidate choice is measured in this repository. The unit's record of history against the carrier
(`anchoring_record`) is reported with every forecast, as in ANKYRA 2.2.1; its deciles on the test windows, against
the realised gain, are in [`results/anchoring_gain_deciles.csv`](../results/anchoring_gain_deciles.csv) (descriptive).

## Exact properties of the carrier network

The carrier is `CopyPointerNetwork` (`build_carrier()`, [`model.py`](../ankyra_lite/carrier/model.py)). Its inputs
are built by `features` ([`features.py`](../ankyra_lite/carrier/features.py)), and its forecasts at the origin and the
pseudo-origins by `carrier_forecasts` ([`inference.py`](../ankyra_lite/carrier/inference.py)). Line numbers refer to
these three files in version 0.1.0 and, where named, to the anchoring layer's
[`core.py`](../ankyra_lite/anchor/core.py); the training code ([`train.py`](../ankyra_lite/carrier/train.py)) is cited
by function name.

Notation: $s>0$ is the scale input, $r$ the raw output of the level head, $\ell=s\,\mathrm{softplus}(r)$ the level,
$p\in\Delta_{31}$ the day shares and $q_d\in\Delta_{24}$ the hourly shares of day $d$, where $\Delta_n$ is the set of
probability vectors of length $n$. The trajectory is (`model.py` 170–173, 335)

$$F_{d,h}=24\cdot(31\,\ell\,p_d)\cdot q_{d,h}=744\,\ell\,p_d\,q_{d,h}.$$

Each property gives a claim, a proof sketch with code references, where it stops, and its check in
[`test_carrier_properties.py`](test_carrier_properties.py). The checks run untrained carriers under several torch
seeds and the same carriers with perturbed weights, so that the copy pointer's gates are non-zero. Statements about
real arithmetic are checked in float64, statements about the delivered carrier forecasts on the package's own float32
path. None of them is an accuracy guarantee.

**C1. Nonnegativity.**

*Claim.* Every hour of the carrier's trajectory is nonnegative, for any weights and any input the network accepts;
in real arithmetic it is strictly positive. For inputs built by `features`, the only exponential that the network
applies to an input value, $\sinh$ of the hourly day tokens in the copy pointer, is at most 8,760 (up to the float16
storage of the token).

*Proof.* The network rejects a scale that is not finite and positive (`model.py` 313–316). $\mathrm{softplus}(r)>0$,
so $\ell>0$ (335); the softmax shares are positive (170–171); $F_{d,h}$ is their product with positive constants
(172–173), and a product of nonnegative floating-point numbers is nonnegative. For the bound, $s\ge\bar y$, the mean
of the observed load in the last 8,760 hours (`features.py` 111–114). An observed hour $h$ in that window has
$y_h\le\sum y=n\,\bar y\le8760\,s$, with $n\le8760$ observed hours. The pointer applies $\sinh$ to the tokens
$\operatorname{arcsinh}(y_h/s)$ of the last 56 days (`model.py` 410–414; the tokens at `features.py` 133), i.e. to
hours inside that window. When no observed load in the window is positive, $s=1$ (115–117) and those tokens are zero.

*Where it stops.* In floating point the shares can underflow to exactly zero, so strict positivity does not carry
over. Finiteness also needs finite weights and no overflow inside the linear layers, which is not bounded here. An
input that overflows the pointer (a day token of 88 in float32) gives a non-finite forecast, and `carrier_forecasts`
raises `FloatingPointError` on any non-finite or negative forecast before it reaches the anchoring layer
(`inference.py` 80–81). The ensemble average (83–84) of nonnegative forecasts is nonnegative. The nonnegativity of
the trajectory that ankyra-lite delivers comes from the anchoring layer's own projection (P11), not from this
property.

*Check.* `NonnegativityTests`: random weights and inputs in float32 and float64 (strictly positive in float64);
extreme logits (exact zeros occur, negative values do not); a record with one large hour 1,000 hours before the
origin and almost no load elsewhere (largest pointer ratio between 8,000 and 8,760 × 1.01); the inference guard on an
overflowing token.

**C2. Energy budget.**

*Claim.* The 744-hour mean of the trajectory is the level, $\frac1{744}\sum_{d,h}F_{d,h}=\ell$; the daily means are
$31\,\ell\,p_d$; the energy output is $744\,\ell$ kWh. The budget outputs are exactly the P1 blocks of the
trajectory: level $\ell$, centred daily path $b_d=31\,\ell\,(p_d-1/31)$ and within-day path
$w_{d,h}=744\,\ell\,p_d\,(q_{d,h}-1/24)$. For an ensemble, the mean of the averaged trajectory is the average of the
members' levels.

*Proof.* $\sum_h q_{d,h}=1$ gives the daily mean $31\,\ell\,p_d$, and $\sum_d p_d=1$ gives the window mean $\ell$
(`model.py` 170–173). The returned daily path and within-day path are these daily means minus $\ell$ and the hourly
values minus their day's mean, i.e. P1's projections, and `energy_kwh` is computed as $744\,\ell$ (174–176). The
ensemble is a plain average (`inference.py` 83–84), and the mean is linear.

*Where it stops.* The identities are exact in real arithmetic and hold to rounding in floating point (the check
allows a relative difference of $10^{-12}$ in float64 and $10^{-5}$ in float32; about $2\times10^{-7}$ is observed in
float32). This is the carrier's level, one candidate of the anchoring layer (P8, P20); ankyra-lite's energy readout
is the anchoring layer's level (P3).

*Check.* `EnergyBudgetTests`.

**C3. The shares are distributions.**

*Claim.* $p\in\Delta_{31}$ and every $q_d\in\Delta_{24}$. The copy pointer's attention $\alpha_d$ of forecast day $d$
over the last 56 days is a probability vector supported on the *complete* days among them (usable, with all 24 hourly
observation flags set), and is identically zero when no day is complete. The attended within-day profile
$\sum_j\alpha_{d,j}\sigma_j$ is then a probability vector over the 24 hours: a convex combination of the complete
days' within-day share profiles $\sigma_j$ (the load divided by 24 times the day's mean when that mean exceeds
$10^{-6}$ of the scale, uniform otherwise).

*Proof.* Softmax (`model.py` 170–171). The pointer gives non-complete days the score $-\infty$ before its softmax and
sets $\alpha$ to zero when no day is complete (412–413, 422–425); $\sum_h y_h/(24\,\bar y)=1$ for $\bar y>10^{-6}$
(414–417).

*Where it stops.* The copied profile enters the hourly logits as a centred log-profile through a learned gate
(429–431), and the copied relative daily mean enters the daily logits in the same way (434–436), so the delivered
shares $p$ and $q_d$ are not themselves mixtures of copied days.

*Check.* `ShareTests`.

**C4. Causality at the issue time.**

*Claim.* The carrier forecast issued at hour $q$ reads load and temperature only at hours before $q$, and day types
only at hours before $q+744$: the calendar of its own forecast month is treated as known. Hence `carrier_forecasts`
at origin $o$ is bitwise unchanged when the load and temperature at hours $\ge o$, and the day types at hours
$\ge o+744$, are replaced by anything (NaN, infinity, negative or huge values); and the pseudo-origin forecast issued
at $q=o-744k$ is bitwise unchanged by any change at hours $\ge q$.

*Proof.* `_take` copies only the requested hour range (`features.py` 43–53). Load and temperature are taken from
$[q-8784,q)$ for the daily tokens and the scale (108, 120) and from the 24 × 744 hours before that for the monthly
tokens (144–145, 148); the low-load diagnostic uses the same pre-origin hours (181). Day types are taken from before
$q$ (124, 159) and from $[q+12,q+744)$ for the forecast days (175). `pseudo_origins` reads the load before the origin
only (`inference.py` 52), and the forecast for $k$ is built at $q=o-744k$ (97). The network is a deterministic
function of its six inputs (evaluation mode, dropout 0; `inference.py` 75).

*Where it stops.* The day types of the forecast month are an input, and changing them changes the forecast. A holiday
announced after the origin therefore enters the forecast (see the calendar limitation in the
[README](../README.md#limitations)).

*Check.* `CausalityTests`.

**C5. The encoder is causal in token order.**

*Claim.* The encoder's memory of daily token $j$ depends only on daily tokens $0,\dots,j$ and their masks, and the
same holds for the monthly tokens. In particular the copy pointer's key for a recent day summarises that day and
earlier days, never later ones.

*Proof.* Every operation of the encoder acts on one token at a time (masking, projection and age features, the signed
resolution embedding, layer norms, the gate and output layers: `model.py` 197–198, 202–205, 283–287, 291), except the
depthwise convolution with kernel 3, which is padded by $2\cdot$dilation on the left only (199–201), so that its output
at $t$ reads $t$, $t-d$ and $t-2d$. A composition of such maps over the dilation schedule (288–290) is causal. The age
features depend on the position only (271–277). The pointer's keys are a linear map of this memory (419–420).

*Where it stops.* The attention read-out (330) and the pointer read the whole memory, by design; all of it lies before
the issue time (C4).

*Check.* `EncoderOrderTests`.

**C6. Masked tokens are ignored.**

*Claim.* The content of a daily or monthly token whose mask is false does not affect any output, even when it is NaN.

*Proof.* Every read of the tokens passes through `torch.where(mask, tokens, 0)` before any arithmetic (`model.py` 283;
in the pointer 412). Masked positions of the memory are set to zero (205, 291) and excluded from the attention by the
key padding mask (330), which gives them weight exactly zero. The input check tests finiteness only on usable tokens
(309–310).

*Where it stops.* The masks themselves are inputs: which days are usable changes the forecast. `features` also sets
unusable tokens to zero (`features.py` 140, 172).

*Check.* `MaskTests`.

**C7. Scale equivariance.**

*Claim.* Let $c>0$, and let the observed load of the last 8,760 hours before the issue time include a positive value
(so that the scale fallback $s=1$ is not used). Then the carrier forecast for the load $c\,y$, with temperature, day
types and category unchanged, is $c$ times the forecast for $y$: exactly in real arithmetic for every $c>0$, and bit
for bit in floating point when $c$ is a power of two (barring overflow and underflow).

*Proof.* *Features.* Means, standard deviations, percentiles and the maximum are positively homogeneous of degree
one, so $s(cy)=c\,s(y)$ (`features.py` 110–114). Every load-derived token is a ratio to $s$: the 24 hourly values and
the daily mean and standard deviation (133–134); the monthly mean, standard deviation, 10th and 90th percentiles,
workday and weekend means and change of the mean (152, 157–162, 170). The other tokens depend only on which hours are
observed or exactly zero, which $c>0$ does not change (observation flags, coverage and masks: 109, 133, 137, 139, 146,
153, 164; zero-load share: 163), or not on the load at all (temperature, calendar, day types, category). So the six
network inputs are unchanged except $s\mapsto cs$. The low-load threshold (181) scales with $c$ but is a diagnostic,
not a network input (`model.py` 45). *Network.* The scale enters the computation only through the level,
$\ell=s\,\mathrm{softplus}(r)$ (335; the input check at 313–316 and the output `scale_kw` at 357 do not compute with
it), and $F=744\,\ell\,p\,q$ (172–173); the pointer's $\sinh(\text{token})$ is the load divided by the scale (414). So
$F(cy)=c\,F(y)$. *Floating point.* Multiplying by $2^k$ is exact and commutes with correctly rounded addition,
subtraction, multiplication, division and square root and with rounding to float32 and float16 (barring overflow and
underflow). Every token is therefore bitwise the same, and $s$, $\ell$, every product in $F$ and the ensemble average
(`inference.py` 83–84) are scaled exactly.

*Where it stops.*

- *Counterexample.* When no observed load in the last 8,760 hours is positive but older load is, $s=1$ for every $c$
  (`features.py` 115–117). The monthly tokens then carry the load unscaled, and the forecast is not $c$ times the
  original.
- For other factors, the rounding of $cy$ to float32 (`features.py` 52) and of the tokens to float16 (183) differs at
  the rounding level (relative differences below $3\times10^{-5}$ in the check, which allows $10^{-3}$).
- The property is the carrier's. The anchoring layer has absolute thresholds (off-state $10^{-6}$ kW, micro-load
  $10^{-3}$ kW; `anchor/core.py` 70–71), so no such statement is made for the trajectory that ankyra-lite delivers.

*Check.* `ScaleTests`: the factors 2, 1/2, 8, $2^{-6}$ and 1024 bit for bit, from the features through the
two-member ensemble; the network's homogeneity in the scale input at fixed tokens (float64; factors 3.7, 0.013 and
250); the factors 3.7 and 0.3 at the rounding level; the zero-scale counterexample.

**C8. No unit identifier.**

*Claim.* The carrier's forecast for a unit is a function of that unit's own load, temperature and day types, its
category, the start timestamp and the issue time. Two units with the same records and the same category get
bit-identical forecasts, whatever their row index and whatever the other rows of the arrays hold.

*Proof.* The unit index selects the row (`features.py` 49; `inference.py` 42–44) and the category (`features.py`
76–81, 129–130), and nothing else. The network's six inputs (`model.py` 45) carry no index, and its parameters are
linear layers, layer norms, convolution kernels, the attention, the hourly dictionary and one scalar bonus for a
matching day type (236–264, 388–393): there is no per-unit table.

*Where it stops.* The category one-hot is an input (`features.py` 131, 179), so the same records in another category
give another forecast. One carrier is trained per population, so the weights depend on that population's training
set.

*Check.* `UnitIdentityTests`.

**C9. The copy pointer.**

*Claim.*

- (a) At initialisation the copy pointer adds nothing: `build_carrier()` under a torch seed gives exactly the outputs
  of the base network (`StructuredLoadNetwork`) built under the same seed.
- (b) For any weights the pointer does not change the level: the carrier's level and energy, and hence the mean of its
  trajectory (C2), are those of its base network with the same base weights. The pointer only redistributes the
  month's energy over days and hours.
- (c) For any weights the pointer is inactive when none of the last 56 days is complete (C3); the output is then
  exactly the base network's.
- (d) It copies only from the unit's own last 56 days.

*Proof.* (a) The base modules are constructed first, in the order of the base network (`model.py` 383, 236–264), and
the pointer modules afterwards (388–391), so a seeded construction gives the same base weights. The gate's weight and
bias are initialised at zero (392–393), so the gate is zero (426), and the pointer adds $0\cdot l$ with a finite $l$
to the logits (431, 436), which leaves them unchanged. (b) The level is computed in the trunk from the global query
(334–335) and passed on unchanged (438–439); the pointer changes only the daily and hourly logits (431, 436). (c)
Without a complete day the gate is multiplied by zero and the copied terms are replaced by zeros (423–426, 431, 436).
(d) The pointer's tokens and keys are the last 56 daily tokens of the unit's own input (410–412, 419–420).

*Where it stops.* Trained gates are not zero, and the pointer does change the daily path and the within-day shape
(the check measures the change). (a) holds for the initialisation, not for a trained checkpoint.

*Check.* `PointerTests`.

**C10. Pseudo-origin out-of-sample guarantee.**

*Claim.* Let the carrier be trained with cut $c$ (an hour index of the record), and let every test origin $o$
satisfy $o\ge c+4464$ (186 days, $6\times744$ hours). Then every pseudo-origin $q=o-744k$, $k\le6$, satisfies $q\ge c$.
So `carrier_forecasts(..., cut=c)` drops nothing (it returns exactly the $k$ that the anchoring layer reads), and the
target month $[q,q+744)$ of every pseudo-origin forecast lies after every training and validation label of the
population, all of which end at or before $c$.

*Proof.* The anchoring layer reads $k=0,\dots,6$ (`anchor/core.py` 66, 292–295), and
$q=o-744k\ge c+4464-6\cdot744=c$. `pseudo_origins` keeps the $k$ with $o-744k\ge c$ and rejects an origin before $c$
(`inference.py` 50–51, 56–57). On the training side, every window of a population comes from `train.make_windows`,
whose origins run up to $c-744$, so that every label ends at or before the cut; `train.split` assigns training
windows that end by the validation boundary and validation windows that end by the cut; and `train.cut_data` blanks
the load and temperature at and after $c$ before any feature is computed. This does not depend on the validation
rule, since every rule only chooses among the windows of `make_windows`. The gap is tight: at $o=c+4463$ the sixth
pseudo-origin is one hour before the cut and is dropped.

*Where it stops.* The guarantee is about time on the population's own record. The inputs of a pseudo-origin forecast
may include hours of the training period, as those of the forecast at the origin do. Borrowed windows of other
populations are other units; the pooling rule makes them end before the target's validation start in the recorded
absolute time ([README](../README.md#training-recipe-r), step 4), and on the original CINELDI store that recorded time is
59 days early ([README](../README.md#data-splits)). With test origins closer than 186 days to the cut, `cut=` drops
the pseudo-origins before the cut, and the anchoring layer has fewer completed errors (P7).

*Check.* `PseudoOriginTests`.

## References

- Bates, J. M., Granger, C. W. J. (1969). The combination of forecasts. *Operational Research Quarterly* 20(4),
  451–468.
- Juditsky, A., Rigollet, P., Tsybakov, A. B. (2008). Learning by mirror averaging. *Annals of Statistics* 36(5),
  2183–2206.
- Koenker, R., Bassett, G. (1978). Regression quantiles. *Econometrica* 46(1), 33–50.
- Nemirovski, A. (2000). Topics in non-parametric statistics. In *Lectures on Probability Theory and Statistics* (Saint-Flour
  1998), Lecture Notes in Mathematics 1738, 85–277. Springer.
- Rigollet, P., Tsybakov, A. B. (2011). Exponential screening and optimal rates of sparse estimation. *Annals of
  Statistics* 39(2), 731–771.
- Tsybakov, A. B. (2003). Optimal rates of aggregation. In *Learning Theory and Kernel Machines* (COLT 2003), Lecture
  Notes in Computer Science 2777, 303–313.
