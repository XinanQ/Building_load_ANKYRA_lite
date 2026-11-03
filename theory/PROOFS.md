# Proofs of the exact properties

**Post-audit qualification.** These proofs distinguish real arithmetic, finite representable implementation paths and empirical comparisons. The [maintenance correction note](../docs/CODE_AUDIT_CORRECTIONS_20261022.md) preserves the original scored-code identities and records the scoped corrections; no result table was regenerated.

The properties below follow from the construction of ankyra-lite's three parts (identities and elementary bounds),
not from fitted relationships; the historical estimates, the anchoring weights, the carrier's weights and the four
constants of the continuation layer are fitted to the data. This page states them with proofs and counterexamples:

- **P1–P20** belong to the anchoring layer, `ankyra_lite.anchor`, a byte-identical copy of the ANKYRA 2.2.1 package.
  They are numbered in the order of its pipeline, and their statements and proofs are those of the ANKYRA 2.2.1
  theory. The data evidence of that theory (measurements from the method's earlier development and from the ANKYRA
  study, made with TimesFM 2.5 as the carrier) is not a result of this repository and is left out; where this
  repository's result files measure a consequence, the statement names the file.
- **C1–C14** belong to the carrier, `ankyra_lite.carrier`
  ([Exact properties of the carrier network](#exact-properties-of-the-carrier-network)). C1–C10 hold for any weights,
  trained or not, under the arithmetic and input assumptions of each claim, except C9(a), which concerns
  initialisation. C11–C14 (new in 0.2.0) concern the loss weighting of recipe Rw's second stage and require their
  stated support and positive-floor conditions; an arbitrary pool need not satisfy them.
- **O1–O9** belong to the continuation layer, `ankyra_lite.continuation` (new in 0.2.0;
  [Exact properties of the continuation layer](#exact-properties-of-the-continuation-layer)). They hold for any
  admitted trajectory under each claim's assumptions. The analytic formulas allow the stated nonnegative
  coefficients and positive decay scales; floating-point finiteness and strict positivity require separate
  representability conditions. They are not guaranteed for arbitrary custom constants.

They make the construction auditable:

- any division of labour between two forecasters can be scored from block losses;
- every weight has a stated support requirement and a bound on how far it can move;
- every readout has a guarantee or a counterexample that marks where the guarantee stops;
- the carrier's forecast is a nonnegative energy budget built only from data before its issue time, and every
  pseudo-origin forecast that the anchoring layer scores is out of sample for the carrier;
- the loss weights of recipe Rw are read from each training window's own context, keep every population's share of
  the draws and reduce to recipe R when they are all equal;
- the continuation layer moves each hour of the anchored trajectory by a factor whose bounds are known at the origin,
  reads no load at or after the origin (of the forecast month only the calendar of its first week), and leaves the
  energy, level and peak readouts unchanged.

The index, with the condition, use, code and check of each property, is in [README.md](README.md). P1–P20 are
implemented in [`operators.py`](operators.py) (or in the anchoring layer's `anchor.blocks`, `anchor.readouts` and
`anchor.metrics`) and checked in [`test_operators.py`](test_operators.py); C1–C10 are properties of the carrier's own
code and are checked in [`test_carrier_properties.py`](test_carrier_properties.py), C11–C14 in
[`test_weighting_properties.py`](test_weighting_properties.py); O1–O9 are properties of the continuation layer's code
and are checked in [`test_continuation_properties.py`](test_continuation_properties.py); counterexamples are included
in all four. Where this repository's result files measure a property's consequence on data, the statement names the
file.

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

*Measured consequence (this repository).* ankyra-lite replaces the carrier's level, daily path and within-day block by
anchored ones. The block RMS ratios of that replacement, as unit-equal contrasts of ankyra-lite against its carrier
alone in each block (level, daily path, within-day; all test windows; descriptive), are the rows
`anchored vs carrier alone` of [`results/contrast_ii_blocks.csv`](../results/contrast_ii_blocks.csv). Figure 7a
([`figures/fig7_operators.png`](../figures/fig7_operators.png)) replaces the carrier's blocks by ankyra-lite's one at a
time on one Cambridge test window, where each bar splits exactly into the three block losses.

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

*Measured consequence (this repository).* [`results/handover_granularity.csv`](../results/handover_granularity.csv)
scores ankyra-lite against fixed divisions of the same two sources on the test windows of the nine populations
(descriptive). The fixed division (the history side's six-candidate level and daily path with the carrier's within-day
block) is resolvably worse than ankyra-lite on all nine; the arm that gives the carrier's daily means no weight is
resolvably worse than the carrier alone on four; a fixed half weight recovers between about half and almost all of
the per-unit weights' gain ([EVALUATION.md](../docs/EVALUATION.md#what-the-handover-contributes)).

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

*Measured consequence (this repository).* This repository does not compare the least-squares weight with an
inverse-MSE weight. [`results/handover_granularity.csv`](../results/handover_granularity.csv) compares the estimated
least-squares weights (four weekly weights, arm `AW`, which is ankyra-lite; one weight per window, arm `AM`) with a
fixed weight of 1/2 (arm `Ah`), the value towards which shared errors pull inverse-MSE weights: `AW` is resolvably
better than `Ah` on two of the nine populations and not separated on the others, and `AW` and `AM` are not separated
on any. The mean estimated weight on the carrier's daily means lies between 0.49 and 0.73 in every forecast week
([`results/lead_weeks.csv`](../results/lead_weeks.csv)).

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

*Measured consequence (this repository).* On the test windows the projection binds on 237 of 5,716 windows (BDG2 1,
GoiEner non-household 142, households 29, Cambridge 9, EWELD 21, Drammen 2, CINELDI 33; none on HEEW and Oslo; column
`windows_projection_binds` of [`results/energy_level_readout.csv`](../results/energy_level_readout.csv)). Only there do
the level readout and the energy of the delivered trajectory differ, by the energy the clip adds (P11). The file gives
the monthly energy contrasts with the level readout; their counts of resolved contrasts equal those with the
delivered trajectory (`energy_error.csv`) on every population (descriptive).

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

*Measured consequence (this repository).* The comparator `same readout on the carrier alone` of
[`results/peak_readout.csv`](../results/peak_readout.csv) applies the readout to the carrier's own forecast at the
origin. The excursions come from the context and are the same for both, so the two readouts differ only through the
forecast daily means. ankyra-lite's readout is resolvably worse on Oslo and Drammen and not separated on the other
seven populations; its daily path is resolvably worse than the carrier's on Drammen and the households
([`results/contrast_ii_blocks.csv`](../results/contrast_ii_blocks.csv)). On these data the anchoring layer's level
gains do not carry measurably into the peak (descriptive).

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

*Evidence for the peak readout (this repository).* The readout is applied to ankyra-lite's nonnegative projected
pre-continuation trajectory F and compared with the maximum of its delivered trajectory F' (rows `trajectory maximum`).
This is a comparison of the released readout choices, not a pure operator comparison on a fixed curve. It is also
compared with the previous window's observed peak and with the same readout on the baselines' trajectories, on the
test windows of every population
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

*Measured consequence (this repository).* [`results/level_candidate_hit_rate.csv`](../results/level_candidate_hit_rate.csv)
measures the hit rate of the level block: how often the candidate with the largest weight (the smallest pseudo-origin
errors) is the one whose window mean at the origin is closest to the realised window mean, read against the chance
rate $1/K$ averaged over the windows. Pooled over the nine populations it is 0.209 for the seven candidates against
0.144 (difference +0.065, 95% interval +0.054 to +0.076, resampling units) and 0.233 against 0.169 for the six
historical candidates (descriptive). The pseudo errors therefore carry some information about the target's best
candidate, far from enough to select it reliably, which is the situation in which the strongly shrunk weights of P8
are the reasonable choice. The unit's record of history against the carrier
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
the trajectory that ankyra-lite delivers comes from the anchoring layer's own projection (P11), kept by the
continuation layer (O1), not from this property.

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
absolute time ([README](../README.md#training-recipe-rw), step 4), and on the original CINELDI store that recorded time is
59 days early ([README](../README.md#data-splits)). With test origins closer than 186 days to the cut, `cut=` drops
the pseudo-origins before the cut, and the anchoring layer has fewer completed errors (P7).

*Check.* `PseudoOriginTests`.

### The loss weighting of recipe Rw (C11–C14)

New in 0.2.0. Recipe Rw is recipe R with one change in its second stage: every training window i of the pool gets a
weight w_i in the loss (`recipe("span_return_weighted", ...)`, i.e. the span return with
`noise_weighting=NoiseWeights()`; [`train.py`](../ankyra_lite/carrier/train.py), cited by function name). Notation,
for a pool window i of population Q with origin o and causal scale s_i (the carrier's scale input):

- x_t = sinh(day token) = y_t / s_i, the load divided by the window's scale;
- σ̂²_i, the mean of (x_t − x_{t−168})² over the hours t of day tokens 345–365 (the last 21 days before o) whose
  partner hour t − 168 is also observed; undefined with fewer than 168 such pairs (`noise_sigma2`);
- p_i, the probability that one draw of the sampler picks window i (`draw_probabilities`);
- f, the p-weighted 25th percentile of the defined σ̂² over the pool (`weighted_quantile`);
- ω_i = max(σ̂²_i, f)^{−1/2}, and ω_i = ω̄, the p-weighted median of the defined raw weights, where σ̂²_i is undefined;
- w_i = c_Q ω_i, with c_Q = Σ_{j∈Q} p_j / Σ_{j∈Q} p_j ω_j (`noise_weights`).

The loss of a batch B is (`weighted_log1p_loss`)

$$L_w=\frac{\sum_{i\in B}w_i\,\ell_i}{\sum_{i\in B}w_i},\qquad \ell_i=\log\big(1+\max(0,\ \mathrm{MSE}_i+0.1\,\mathrm{KL}_i)\big),$$

where ℓ_i is the per-window term of `forecast_loss` (with `compression=True`), unchanged. The checks are in
[`test_weighting_properties.py`](test_weighting_properties.py); they use synthetic training arrays, the package's own
sampler and untrained carriers on the CPU.

**C11. Unit weights are recipe R.**

*Claim.*

- (a) With every weight equal to 1, L_w is bit for bit the loss of recipe R, the batch mean of ℓ_i, on the CPU, and so
  are its gradient and two optimisation steps of `train_step` from the same weights (clipping and AdamW included).
- (b) L_w does not change when every weight is multiplied by the same c > 0 (bit for bit when c is a power of two); in
  particular any constant weight gives recipe R's loss.
- (c) Without weighting (`noise_weighting=None`, the default and every recipe of version 0.1) `train_step` runs the
  code of version 0.1, and `TrainConfig.describe()` has no new key, so the configurations and records of the 0.1
  recipes are unchanged. Rw's configuration differs from R's (`recipe("span_return", ...)`) in `noise_weighting` only.

*Proof.* (a) Both paths compute the same tensor ℓ = log1p(clamp₀(raw)). Then 1 · ℓ_i = ℓ_i exactly, the sum is the same
reduction and Σ_i 1 = |B| exactly; on the CPU PyTorch computes the mean of a float tensor as its sum divided by the
number of elements (the check confirms it bit for bit for batches of 64 and of 5), so the two losses are the same
number. The derivative of L_w with respect to ℓ_i is
w_i / Σw = 1/|B|, the derivative of the mean, so backpropagation, the clipping and the optimiser see the same numbers.
(b) Σ_i (c w_i) ℓ_i / Σ_i c w_i = Σ_i w_i ℓ_i / Σ_i w_i; multiplication by 2^k is exact and commutes with every
rounding. (c) `train_step` takes the unweighted branch when `weights` is None, `describe` leaves the field out when it
is None, and `recipe` builds Rw as the span return plus `noise_weighting=NoiseWeights()`.

*Where it stops.* (a) is a property of the CPU kernels, checked with the installed PyTorch: on a GPU, or with any
reduction whose order differs from the mean's, the two losses agree to rounding only, and GPU training is not bitwise
repeatable in any case. For constant factors that are not powers of two the losses agree to float32 rounding. Equal
weights are not Rw: Rw's weights are not constant (C13), and they change which windows the loss emphasises.

*Check.* `UnitWeightTests`.

**C12. Each population keeps its share of the draws.**

*Claim.* The p_i are the sampler's exact per-draw probabilities: the target's windows share
⌊16,000 · ½⌋ / 16,000 = ½ of the draws, each of its units with equal total probability, and the other half is split
equally across the borrowed populations, unit-balanced within each (all draws are the target's when there is none).
The normalisation gives, for every population Q of the pool,

$$\sum_{i\in Q}p_i\,w_i=\sum_{i\in Q}p_i .$$

So a draw from Q carries an expected weight of 1, the expected total weight of an epoch is its number of draws, and
the weighting moves weight only between windows of the same population: in expectation over the draws, neither the
target's half nor any borrowed population gains or loses weight, whatever its noise level.

*Proof.* `sampler` draws ⌊draws · own_fraction⌋ windows of the target with the unit-balanced probabilities and, for
each other draw, a borrowed population uniformly and then one of its windows with its unit-balanced probabilities;
`draw_probabilities` writes these probabilities down. The normaliser c_Q = Σ_Q p / Σ_Q p ω gives
Σ_Q p c_Q ω = Σ_Q p.

*Where it stops.*

- The balance holds in expectation, not within a batch: a batch's Σw differs from its size, which is why L_w is
  normalised by Σw (C11b).
- It balances weight, not loss: the weights are correlated with the windows' losses, so each population's share of the
  loss can still move.
- With a target share other than ½ the target's share is ⌊draws · own_fraction⌋ / draws; the recipe fixes ½.
- The floor f is pooled over the populations, so a noisier population changes the weights of the other populations,
  not their shares.

*Check.* `SourceBalanceTests` (including a Monte-Carlo comparison of `draw_probabilities` with 10 epochs of the
sampler's draws).

**C13. Bounded weights and the floor.**

*Claim.* Let σ̂²_max be the largest defined σ̂² of the pool, and let f > 0. Then:

- (a) ω_i is nonincreasing in σ̂²_i and equals f^{−1/2}, the largest raw weight, for every window with σ̂²_i ≤ f; these
  windows carry at least a quarter of the draw mass of the windows with a defined σ̂²;
- (b) the neutral raw weight ω̄ of the windows without a noise estimate lies between the smallest and the largest
  defined raw weight;
- (c) every pool window gets a finite positive weight, c_Q (σ̂²_max)^{−1/2} ≤ w_i ≤ c_Q f^{−1/2}, and the windows outside
  the pool get none (NaN);
- (d) within a population the weights differ by at most the factor (σ̂²_max / f)^{1/2}.

*Proof.* max(·, f) and t ↦ t^{−1/2} are monotone. The weighted quantile is the smallest value whose cumulative draw
share reaches ¼, so the windows at or below it carry at least that share. A weighted median of values in
[ω_min, ω_max] lies in that interval. (c) and (d) follow, with c_Q > 0.

*Where it stops.*

- *Counterexample.* The floor can be zero: windows whose last three weeks repeat exactly (σ̂² = 0, for instance a meter
  that repeats one week) holding a quarter or more of the draw mass give f = 0 and an infinite raw weight.
  `noise_weights` refuses such a pool (ValueError).
- The factor in (d) is set by the pool's own noise range; on the released pools the largest weight is
  1.56 (Cambridge) to 6.23 (GoiEner non-household) times the draw-weighted median weight (`weight_max_over_median` of
  each weights record named in `checkpoints/MANIFEST.json`).
- The bounds are not an accuracy statement: down-weighting noisy windows changes what the carrier fits, and whether
  that helps is measured ([EVALUATION.md](../docs/EVALUATION.md)).

*Check.* `WeightBoundTests`.

**C14. The weights read only each window's own recent context.**

*Claim.* σ̂²_i is a function of day tokens 338–365 of window i (its last 28 context days, the hours [o − 672, o)
before its origin o), columns 0–47 (the 24 hourly values arcsinh(y/s) and the 24 observation flags), and of nothing
else. It is the weekly-naive relative mean squared error of the last three weeks,

$$\hat\sigma^2_i=\frac1{|P_i|}\sum_{t\in P_i}\Big(\frac{y_t-y_{t-168}}{s_i}\Big)^2,\qquad P_i=\lbrace t\in[o-504,o):\ y_t\ \text{and}\ y_{t-168}\ \text{observed}\rbrace,$$

defined when |P_i| ≥ 168. f, ω̄ and c_Q are functions of the pool windows' σ̂² and p only. Hence:

- (a) no label hour, no validation or test window and no window outside the pool enters a weight;
- (b) the weights are computed once per pool, before training, use no random stream and are the same for every seed;
- (c) they are bitwise invariant to multiplying a population's load by a power of two (through C7), and two windows
  of one population with the same tokens get the same weight (no unit identifier, C8).

*Proof.* `noise_sigma2` reads `day_tokens[:, 338:366, :48]`; sinh(arcsinh(v)) = v; a flag above ½ marks an observed
hour; the pairs are 7 days apart and need at least 168. `pool_sigma2` reads only the pool's rows, in chunks of
ascending rows (also from a `RowConcat`, never the full rows), and `noise_weights` combines σ̂² with
`draw_probabilities`, which reads the units of the pool rows only. Day token 365 covers the hours [o − 24, o)
(`features`), so everything read lies before the window's origin. No call draws a random number.

*Where it stops.*

- The tokens are stored in float16 for training (C7), so σ̂² is computed from rounded values: against the formula on
  the load the relative difference is about 2 × 10⁻⁴ to 5 × 10⁻⁴ on the synthetic records of the check (which allows
  5 × 10⁻³), and below 2 × 10⁻⁷ with float32 tokens (allowed 10⁻⁵).
- σ̂² is relative to the window's own causal scale s_i (C7), not to its mean over the 28 days.
- A window's weight depends on the other windows of the pool (through f, ω̄ and c_Q): adding or removing a population
  changes every weight.
- A window with fewer than 168 observed pairs gets the neutral weight, whatever its noise.

*Check.* `WeightInputTests`.

## Exact properties of the continuation layer

The continuation layer is `ankyra_lite.continuation` (new in 0.2.0; [`continuation.py`](../ankyra_lite/continuation.py),
cited by function name). `forecast(..., continuation=True)` applies it after the anchoring layer and the state rule, to
the trajectory that version 0.1 delivered. Notation, for a window with origin o:

- y_t, the load before the origin (unobserved hours are NaN);
- F(h), h = 0 … 743, the trajectory before the layer (`trajectory_before_continuation_kw`);
- s = max(mean of the observed y over [o − 672, o), 10⁻³ kW), the scale, and ε = 0.01 s;
- d*, the reference day (O9);
- m_f and m_d, the means of y over the last 3 and the last 24 hours before o;
- n_f = max(mean of F(24d* + 21 … 24d* + 23), 0) and n_d = max(mean of F(24d* … 24d* + 23), 0).

The two anomalies and the correction are (`anomalies`, `delta`, `apply_batch`)

$$x_f=\mathrm{clip}\Big(\ln\frac{m_f+\varepsilon}{n_f+\varepsilon},-1,1\Big),\qquad x_d=\mathrm{clip}\Big(\ln\frac{m_d+\varepsilon}{n_d+\varepsilon},-1,1\Big),$$

$$\Delta(h)=\kappa_1x_f\,e^{-h/\rho_1}+\kappa_2x_d\,e^{-h/\rho_2},\qquad F'(h)=F(h)\,e^{\Delta(h)},$$

with x_f = 0 when one of the last 3 hours is unobserved, x_d = 0 when one of the last 24 is, x_f = 0 when m_f + ε ≤ 0
and x_d = 0 when m_d + ε ≤ 0 (negative loads, where the logarithm is undefined: a rule of the release, see O4 (f)),
Δ evaluated in the order
(κ₁x_f)e^{−h/ρ₁} + (κ₂x_d)e^{−h/ρ₂}, and F' = F on a state window. The constants are κ₁ = 0.479, ρ₁ = 12 h,
κ₂ = 0.330 and ρ₂ = 168 h (`CONSTANTS`), fitted once on development windows and shared by every population
([METHOD.md](../docs/METHOD.md#the-continuation-layer)). The envelope of the correction is

$$E(h)=\kappa_1e^{-h/\rho_1}+\kappa_2e^{-h/\rho_2}.$$

Unless stated otherwise, the analytic properties are in real arithmetic for finite constants κ₁, κ₂ ≥ 0 and
ρ₁, ρ₂ > 0 under the stated input conditions; numerical values use `CONSTANTS`. Floating-point claims additionally
require the representability conditions below. The maintenance implementation preserves the original correction
and `F64 * exp(delta)` evaluation order, restores exact input zeros including signed zero, and raises `ValueError`
for a non-finite non-state correction or final output. It does not clip custom constants, and nonzero outputs may
underflow to zero. This is a bounded API contract, not an arbitrary-constant finiteness theorem. The checks are in [`test_continuation_properties.py`](test_continuation_properties.py): on synthetic
windows, and end to end through `ankyra_lite.forecast` on an artificial building with a stand-in carrier and an
untrained one; bit for bit unless a tolerance is stated. None of them is an accuracy guarantee: what the layer does on
the test windows is measured ([EVALUATION.md](../docs/EVALUATION.md)).

**O1. Nonnegativity, sign and zeros.**

*Claim.* In real arithmetic F' has the sign of F hour by hour and F'(h) = 0 only where F(h) = 0. In floating point,
accepted finite non-state outputs preserve nonnegativity; exact input zeros are explicitly restored, including
signed zero. Nonzero values can underflow to zero. With the frozen constants, |F(h)| at least the smallest normal
number (2.2 × 10⁻³⁰⁸) stays nonzero provided the final output is finite. Off state windows the trajectory before the
layer is nonnegative (the anchoring layer's projection, P11), so accepted delivered outputs are nonnegative too.

*Proof.* |x_f|, |x_d| ≤ 1 and 0 < e^{−h/ρ} ≤ 1, so |Δ(h)| ≤ E(0) and the factor e^{Δ(h)} lies in
[e^{−E(0)}, e^{E(0)}], positive in real arithmetic. A finite representable positive floating-point factor preserves
the sign unless the product underflows; for the frozen constants its minimum is e^{−0.809} = 0.445, so only the
smallest subnormal number can underflow. That numeric bound does not apply to arbitrary custom constants. The
maintenance `apply_batch` rejects a non-finite input trajectory, correction or final output on a non-state window
with `ValueError`; exact input zeros are restored before final-output validation even if the intermediate factor
overflows. State-window bypass semantics are retained.

*Where it stops.*

- *Counterexample.* F(h) = 2^{−1074} kW (5 × 10⁻³²⁴, the smallest subnormal number) with Δ(h) = −0.809 gives
  F'(h) = 0.
- The logarithm is defined for m + ε > 0 only. When m_f or m_d is at or below −ε (a net-metered record can do this)
  that anomaly is set to 0 and flagged (O4 (f)), so the claim still holds; a mean slightly below zero but above −ε is
  used as it is.
- The property is about signs, not errors.

*Check.* `NonnegativityTests`.

**O2. Bounded correction.**

*Claim.* In real arithmetic |Δ(h)| ≤ E(h), E is nonincreasing in h and is strictly decreasing if at least one κ
is positive. With κ₁ = κ₂ = 0 it is identically zero. Wherever F(h) ≠ 0, e^{−E(h)} ≤ F'(h)/F(h) ≤ e^{E(h)}. For the
frozen constants:

| Hour h | E(h) | F'(h) / F(h) lies in (rounded outward) |
|---|---|---|
| 0, the first hour | 0.809 (0.8089999999999999 in floating point) | [0.4453, 2.2457] |
| 23, the last hour of day 1 | 0.35824 | [0.6989, 1.4309] |
| 167, the last hour of day 7 | 0.12213 | [0.8850, 1.1299] |
| 168 (first hour of day 8) and later | 0.121401 at h = 168, less afterwards (≤ 0.1215) | [0.8856, 1.1291] |
| 743, the last hour of day 31 | 0.0039609 (≤ 0.004) | [0.99604, 1.00397] |

The bound is reached when both anomalies are clipped to the same sign. In floating point, |Δ(h)| never exceeds the
envelope evaluated with the same operations, (κ₁ · 1)e^{−h/ρ₁} + (κ₂ · 1)e^{−h/ρ₂} (`envelope`), with equality at
x_f = x_d = ±1, provided the operations are finite. Strict decrease is a real-arithmetic property; rounding and
underflow can introduce plateaus. Ratio bounds do not guarantee a nonzero floating-point result for custom constants.

*Proof.* The clip, the triangle inequality with κ ≥ 0 and e^{−h/ρ} > 0, and the monotonicity of the exponential; E is a
sum of decreasing functions with nonnegative weights, strictly decreasing when a weight is positive. In floating
point, where the intermediate operations are finite, rounding is monotone and symmetric under negation, so |x| ≤ 1 gives |fl(κx)| ≤ κ, then |fl(fl(κx) g)| ≤ fl(κg), and the rounded sum is bounded by the rounded
sum of the bounds.

*Where it stops.*

- The bound limits how far the layer moves an hour away from F, not the error: a correction of the full envelope in
  the wrong direction is possible, for instance when the load changes level right after the origin.
- "At most 0.12 at the end of day 7" holds only after rounding to two decimals: at h = 167 the envelope is 0.12213;
  from h = 168 on it is at most 0.1215 (0.121401 at h = 168, slightly above 0.1214).
- With other constants E(0) = κ₁ + κ₂.

*Check.* `BoundTests`.

**O3. Causality.**

*Claim.* F' at origin o is a function of F, of the load at the hours [o − 672, o) together with which of them are
observed, and of the day types at the hours o − 12 and o + 12 + 24d (d = 0 … 6); of nothing else. Hence F' is bitwise
unchanged when the load at hours before o − 672 or at or after o, or the day types at any other hour, are replaced by
anything (NaN, infinity, negative or huge values). With C4 and the anchoring layer's own isolation, the forecast with
the layer reads no load or temperature at or after the origin.

*Proof.* `window_inputs` copies `load_kw[o − 672:o]`, `day_types[o − 12]` and `day_types[o + 12 + 24d]`; `anomalies`
reads these and F; `delta` reads the constants only; `apply_batch` multiplies. `forecast` passes the pre-origin load
with unobserved hours as NaN (the observed mask applied) and the calendar of the record.

*Where it stops.* The calendar of the first forecast week is an input (through d*, O9): a holiday announced after the
origin enters the layer, as it enters the carrier (C4) and the anchoring layer. The layer reads the first week of F
itself, so whatever information F carries enters F'.

*Check.* `CausalityTests` (including a sanity check that the last observed hour is read).

**O4. Bypass.**

*Claim.*

- (a) On a state window (the anchoring layer reported off-state or micro-load) F' = F bit for bit, with either state
  rule: the seasonal naive, or with `state_rule=None` the carrier's forecast, is delivered without continuation.
- (b) If one of the last 3 hours before the origin is unobserved, x_f = x_d = 0 and F' = F bit for bit
  (`bypass="missing hours"` in the record).
- (c) If one of the last 24 hours, but none of the last 3, is unobserved, x_d = 0 and only the fast term acts.
- (d) An unobserved hour earlier in the 672 hours changes only the scale s, the mean of the observed hours.
- (e) The history-only configuration (`forecast(history, None, ...)`) is never continued (`continuation` is None): it
  was not evaluated with the layer.
- (f) Negative loads. If m_f + ε ≤ 0, x_f = 0; if m_d + ε ≤ 0, x_d = 0. The record flags it (`nonpositive_fast`,
  `nonpositive_day`; `nonpositive_f`, `nonpositive_d` in the batch info), and when both anomalies are set to 0 this
  way (or the day span is unobserved) F' = F bit for bit (`bypass="nonpositive load"`). This is a rule of the release,
  chosen and disclosed by the author: the study's reference implementation raised an error there, no development or
  test window reached the case, and so the rule changes no study output. `RULE` and `DIGEST` still identify the frozen
  layer.

*Proof.* (a) `apply_batch` copies F on state windows, and `forecast` passes `state=state_window`. (b) Both spans
contain the last 3 hours, so x_f = x_d = 0; then Δ = (κ₁ · 0)e^{−h/ρ₁} + (κ₂ · 0)e^{−h/ρ₂} = 0, e⁰ = 1 and F · 1 = F
exactly. (c) and (d) hold by construction. (e) `forecast` returns before the layer when `carriers` is None. (f) By
construction (`anomalies`), with the argument of (b) when both anomalies are 0.

*Where it stops.* The normal carrier-inference path requires the 1,344 hours before the origin completely observed.
The inherited precomputed-mapping off-state exception can accept an incomplete context before that check, but then
bypasses continuation as a state window. The missing-hour cases also apply when `continuation.apply` is called
directly on a record with gaps. (f) can occur through `forecast`, since the record may hold negative loads.

*Check.* `BypassTests`, `ForecastBypassTests`.

**O5. Degeneracy: switching off leaves the carrier-specific pipeline unchanged.**

*Claim.* With κ₁ = κ₂ = 0 (`continuation.ZERO`) F' = F bit for bit on every window with a finite trajectory (and on
every state window), for float32 and float64 trajectories. With `continuation=False` (or None) `forecast` returns the composition of the steps of version 0.1
(carrier forecasts, anchoring layer, state rule, peak readout) bit for bit, with `trajectory_before_continuation_kw`
equal to `trajectory_kw` and `continuation` None; `continuation=ZERO` gives the same trajectory and readouts.

*Proof.* (0 · x)e^{−h/ρ} is a signed zero, the sum of two signed zeros is a zero, e^{±0} = 1 and F · 1 = F; float32
values convert to float64 exactly and back. With the layer off `forecast` skips its last step, and the steps before it
are those of version 0.1.

*Where it stops.* `continuation.ZERO` still computes the anomalies and attaches a record, and it refuses a non-finite
trajectory on a window that is not a state window (O1), as the frozen constants do; `continuation=False` (or None)
computes nothing. (On a window with negative loads, O4 (f), ZERO and the frozen constants no longer raise; the study's
reference implementation did.) "Version 0.1" means its steps: with another carrier (six seeds, or recipe Rw) the layer off gives
that carrier in the 0.1 pipeline, not the 0.1 forecasts. Recovering the published 0.1.1 predictions requires the
same inputs and the original recipe-R three-seed checkpoints; this is a reproduction condition. The existing
`tests/test_forecast_v2.py` reference-version checks instead establish same-carrier, same-input pipeline compatibility
using synthetic histories and an untrained carrier with the same state dictionary in both versions (including
reference version 0.1.1); they can skip
when the reference checkout is unavailable. They do not demonstrate a rerun of the released checkpoints.

*Check.* `DegeneracyTests`.

**O6. Readouts are read before the layer.**

*Claim.* the energy readout (`energy_kwh`) is 744 × `level_kw`; the published curve-energy evaluation uses ΣF. The peak is the existing
`peak_readout` operator applied to F, not necessarily max(F). With the same carrier forecasts and inputs,
`level_kw`, `energy_kwh`, `peak_kw`, `raw_trajectory`, the blocks of the raw trajectory, `projection_energy_increase`
and every field of the anchoring layer are bitwise the same with the layer on and off, and `peak_kw` is the peak
readout of `trajectory_before_continuation_kw`. The interval readout is placed around `trajectory_kw`, i.e. around F'.
The delivered hourly curve's energy generally differs from the energy readout (`energy_kwh`) 744ℓ, where ℓ denotes the level readout, by two
terms known at the origin (either may be zero or they may cancel),

$$\sum_hF'(h)-744\,\ell=\Big[\sum_hF'(h)-\sum_hF(h)\Big]+\Big[\sum_hF(h)-744\,\ell\Big],$$

the first `readouts.continuation_energy_change`, the second the pre-layer curve/readout gap, ordinarily the projection
increase `readouts.projection_energy_increase` (P11), and for
F ≥ 0

$$-\sum_hF(h)\big(1-e^{-E(h)}\big)\le\sum_hF'(h)-\sum_hF(h)\le\sum_hF(h)\big(e^{E(h)}-1\big).$$

For a flat trajectory the bounds are −7.3349% and +9.3035% of Σ F (frozen constants; −7.34% and +9.31% rounded
outward), both reached with clipped anomalies. For any nonnegative F the relative change (Σ F' − Σ F) / Σ F lies in
[e^{−E(0)} − 1, e^{E(0)} − 1], i.e. between −55.5% and +124.6% (rounded outward), approached only when nearly all of
the energy lies in the first hours: the flat-trajectory bounds do not hold for every trajectory.

*Proof.* `forecast` computes the trajectory, the level and the peak as version 0.1 does and then F' from that
trajectory; `readouts` reads `daily_means_kw` and `within_day_kw` from the anchoring layer's result, which the layer
does not touch. The identity is a telescoping sum. The bounds follow from O2 hour by hour; for a flat F they are the
means of e^{±E(h)} − 1 over the 744 hours. In general (Σ F' − Σ F) / Σ F is a mean of e^{Δ(h)} − 1 with weights
F(h) / Σ F, so it lies between the smallest and the largest of e^{±E(h)} − 1, which are those at h = 0 (E decreases).

*Where it stops.*

- The readout rule is a choice, not a consequence: because energy and peak are read before the layer, toggling it on a
  fixed carrier leaves those readouts unchanged. The sum of the delivered hourly curve generally differs from the
  energy readout, but need not on every window. `trajectory_kw` sums to `energy_kwh` plus `projection_energy_increase` and
  `continuation_energy_change`; it equals `energy_kwh` only when the two changes sum to zero.
- On the development windows used to choose the layer, the relative difference between the mean of the delivered
  hourly curve and the energy the evaluation scores (the mean of the trajectory before the layer) had a median of
  0.6%, a 99th percentile of 7% and a maximum of 9%. These figures are measured, not bounded by the flat-trajectory
  bounds: a trajectory with more of its energy in the first days can move further, up to the general bound above. On
  the test windows the difference to the level readout (744 × level) is 0.77% / 7.8% / 310% (median, 99th percentile,
  maximum; `results/continuation_record.csv`, row `all nine populations`); the layer's own energy change there lay
  between −14.8% and +33.4%, inside the general bound, and every larger gap is a window where the nonnegativity
  projection binds (P11), which the flat and general bounds above do not cover.
- Intervals placed around F' differ from those placed around F.

*Check.* `ReadoutTests`.

**O7. Consistency and monotonicity.**

*Claim.*

- (a) A forecast that already continues the last observed hours is left alone: if m_f = n_f and m_d = n_d as
  floating-point numbers, then x_f = x_d = 0 and F' = F bit for bit.
- (b) Δ(h) is nondecreasing in x_f and in x_d; x_f is nondecreasing in m_f and nonincreasing in n_f, and likewise x_d.
  So if both anomalies are nonnegative (the last hours at or above the forecast's reference), F'(h) ≥ F(h) for every
  hour of a nonnegative F, and if both are nonpositive, F'(h) ≤ F(h): the layer then raises or lowers the whole window,
  never one part up and another down.

*Proof.* (a) (m + ε)/(m + ε) = 1 exactly, ln 1 = 0 and clip(0) = 0; then as in O4(b). (b) κ ≥ 0 and e^{−h/ρ} > 0, the
logarithm and the clip are monotone, and e^{Δ} ≥ 1 exactly when Δ ≥ 0.

*Where it stops.* With anomalies of opposite signs (the last 3 hours above and the last day below the forecast, or the
reverse) and distinct decay rates, Δ changes sign at most once; a crossing may be outside the forecast window or
absent for nonnegative lead times. With the frozen constants, x_f = 1 and x_d = −1 gives a crossing after about
4.8 hours, whereas x_f = 0.1 and x_d = −1 stays negative throughout all 744 hours. The ratio of the two exponential
terms is monotone when their rates differ; equal rates give a constant sign or an identically zero correction.
Beyond |ln ratio| = 1 the clip makes the anomaly flat. Monotonicity is in the anomalies, not in accuracy.

- The monotonicity in m_f and m_d holds for m + ε > 0 only. At m + ε ≤ 0 (negative loads, O4 (f)) the anomaly is set
  to 0, so as m falls through −ε it jumps from the clipped value −1 (or above) to 0.

*Check.* `ConsistencyTests`.

**O8. Scale equivariance.**

*Claim.* Let c > 0, and let the scale stay above its floor for both y and c y (s ≥ 10⁻³ kW). Then the layer applied to
the load c y and the trajectory c F gives c F': exactly in real arithmetic, and bit for bit when c is a power of two
(barring overflow and underflow).

*Proof.* s(c y) = c s(y), so ε scales with c; the ratios (c m + c ε)/(c n + c ε) do not depend on c, so x_f, x_d and Δ
are unchanged, and (c F)e^{Δ} = c (F e^{Δ}). Multiplication by 2^k is exact and commutes with every rounded operation
the layer uses (means, maxima, sums, products and quotients).

*Where it stops.*

- *Counterexample.* A load of about 10⁻⁴ kW keeps s at its floor of 10⁻³ kW for c = 1 and for c = 8, so ε does not
  scale and the anomalies change with c.
- Other factors are equivariant to rounding (relative differences below 10⁻¹² in the check).
- The property is the layer's. The anchoring layer has absolute thresholds (C7), so no such statement is made for the
  whole forecaster.

*Check.* `ScaleTests`.

**O9. Reference day and identity.**

*Claim.*

- (a) d* depends on the calendar only: it is the first forecast day d = 0 … 6 whose workday class equals that of the
  last observed day, where day types 0–4 are workdays and 5, 6 and 7 (Saturday, Sunday, holiday) are not; each day's
  type is read at its 12th hour (o − 12 for the last observed day, o + 12 + 24d for forecast day d, as in the peak
  readout and the anchoring layer). d* = 6 when no day of the first forecast week has that class; with a regular
  calendar this happens only when the last observed day is a workday and all five weekdays of the first forecast week
  are holidays.
- (b) The fast anomaly compares the same clock hours: for any origin hour, F(24d* + 21 … 24d* + 23) are the hours of
  the day that y(o − 3 … o − 1) occupied. The day anomaly compares the last 24 observed hours with one origin-aligned
  forecast day of the same class.
- (c) The SHA-256 of the canonical JSON of the constants and the rule (sorted keys, separators "," and ":", ASCII) is
  `DIGEST` = d514e3ec93a745b14950671297bf2f922fd512424a804e292377e1b0c4cc1dce, the identity recorded when the layer
  was frozen. Any change of a constant or of the rule changes it, and every `ContinuationRecord` carries the digest of
  the constants it used.

*Proof.* (a) `reference_day` and `window_inputs`. (b) o + 24d* + 21 + j ≡ o − 3 + j (mod 24) for j = 0, 1, 2. (c)
`digest`.

*Where it stops.* The fallback d* = 6 is that of the layer's design; the study's development code used d* = 0. The two
agree wherever a day of the class exists, which was every development window; 33 test
windows, all on EWELD, used the fallback (`results/continuation_record.csv`). The class separates workdays from the other days only,
so a Saturday is compared with a Sunday or a holiday. For an origin after noon an origin-aligned day spans two
calendar days, as in the peak readout.

*Check.* `ReferenceDayTests` (including the definition written out by hand for several origin hours).

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
