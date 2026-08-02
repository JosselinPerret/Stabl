# A deterministic formulation of Stabl

## Executive conclusion

Stabl is not made deterministic by fixing a seed or by averaging the output of
many seeds. Those approaches only hide Monte Carlo variability. There are two
separate random mechanisms:

1. random half-samples estimate each feature's probability of being selected;
2. random permutations or Model-X knockoffs create null features used by the
   FDP+ threshold.

The mathematically canonical deterministic version of the first mechanism is
the average over **all** admissible subsamples. It is an exact, finite-population
quantity rather than a Monte Carlo estimate. Its direct cost is exponential.
For practical data, it can be evaluated with exact compression/certification
when possible, or approximated reproducibly with a balanced combinatorial
design. These two modes must not be presented as equivalent: only complete
enumeration (or certified compression of it) is exact for an arbitrary base
selector.

The second mechanism cannot simply be replaced by one fixed "clever"
permutation without invalidating the paper's exchangeability argument. If no
probabilistic null model is acceptable, artificial features and FDP+ must be
removed. This document proposes a deterministic, compression-based threshold:
choose among the nested stable supports by minimum description length (MDL),
implemented with an extended information criterion. This changes the guarantee
honestly: the result is a sparse, deletion-stable predictive support, not an
FDR-controlled estimate of the unknown biological support.

Removing artificial features also changes the original features' stability
scores: in Stabl, original and artificial columns compete inside every sparse
fit, not only during threshold calibration. D-Stabl is consequently a
mathematically specified redesign, not a seed-free implementation that is
numerically equivalent to published Stabl.

The recommended design, **D-Stabl**, is therefore:

* exact or balanced-design stability scores over a deterministic family of
  sample deletions;
* no artificial random features;
* an MDL threshold over the distinct stability scores;
* explicit exactness and design-discrepancy diagnostics;
* deterministic tie-breaking and deterministic base learners.

This solves seed dependence. It does not claim that finite observational data
can identify the true biological support without assumptions; no method can
provide that distribution-free guarantee.

## 1. What Stabl is trying to achieve

For observations \(X\in\mathbb R^{n\times p}\), outcomes \(y\), and a sparse
learner with regularization parameter \(\lambda\), Stabl seeks a small set of
features that:

* is repeatedly selected under perturbations of the cohort;
* retains predictive information;
* contains few false discoveries.

The paper defines, for bootstrap/subsample run \(b\),

\[
  Z_{j,\lambda,b}
  =\mathbf 1\{j\text{ is selected by the sparse learner at }\lambda
  \text{ on subsample }b\},
\]

and estimates a selection frequency

\[
  \widehat\pi_{j,\lambda}
  =\frac1B\sum_{b=1}^{B}Z_{j,\lambda,b},\qquad
  \widehat s_j=\max_{\lambda\in\Lambda}\widehat\pi_{j,\lambda}.
\]

Stabl augments the \(p\) original features \(O\) with artificial features
\(A\), then defines

\[
  \widehat{\operatorname{FDP}}^+(t)
  =
  \frac{1+\#\{a\in A:\widehat s_a\ge t\}}
       {\max(1,\#\{j\in O:\widehat s_j\ge t\})}
\]

(with a scaling factor when fewer than \(p\) artificial features are used).
The paper's reliability threshold minimizes this ratio.

In this checkout, the corresponding operations are:

* random index generation: `stabl/stabl.py`, lines 151–204 and 1165–1175;
* fitting and averaging selection masks: lines 737–792 and 1188–1208;
* random artificial features: lines 1356–1418;
* FDP+ computation: lines 1420–1457;
* final maximum-over-\(\lambda\) threshold: lines 1315–1354.

The default is 1,000 half-samples (`n_bootstraps=1000`,
`sample_fraction=0.5`) and random-permutation artificial features. The paper
states the same random half-sample construction in its Methods, and gives
complexity

\[
  O\!\left(BR\,n(p+p')\min\{n,p+p'\}\right)
\]

for \(B\) subsamples, \(R=|\Lambda|\), and \(p'\) artificial features.

The seed-dependence reported in
[upstream issue 15](https://github.com/gregbellan/Stabl/issues/15) is therefore
expected, not an optimizer accident. The suggestion in that discussion to run
100 seeds and select frequently returned features adds a second Monte Carlo
layer. Also, the claim that every truly informative feature is theoretically
selected for every seed does not follow from the paper: its theorem bounds an
FDP ratio under exchangeability; it does not guarantee zero false negatives or
seed-invariant support.

## 2. Exact mathematical target

### 2.1 Admissible subsamples

Let \(m=\lfloor\alpha n\rfloor\). For ordinary, unweighted sampling without
replacement, define

\[
  \Omega_m=\{I\subseteq[n]:|I|=m\},\qquad
  N_m=|\Omega_m|={n\choose m}.
\]

For classification, an admissible family may exclude subsets on which the
learner is undefined (for example, subsets containing only one class). This
must be specified, not handled by repeated random draws:

\[
  \Omega_m^{\mathrm{cls}}
  =\{I\in\Omega_m:\text{every required class occurs in }I\}.
\]

For grouped data, the sampling unit must be a group. If groups are
\(G_1,\ldots,G_q\), define admissible subsets of group indices first and take
the union of their rows. A fixed row count and a fixed group count are generally
incompatible when group sizes differ; the API must say which is constrained.

Weighted bootstrap with replacement has a different finite population:
count vectors \(c\in\mathbb N^n\) with \(\sum_i c_i=m\). It must not be silently
treated as subset sampling. D-Stabl should initially support unweighted
sampling without replacement and grouped sampling; weighted/multiset
enumeration is a separate mode.

### 2.2 Complete stability score

Let \(\mathcal A_\lambda(X_I,y_I)\subseteq[p]\) be the support returned by a
deterministic base learner. Define

\[
  h_{j,\lambda}(I)
  =\mathbf 1\{j\in\mathcal A_\lambda(X_I,y_I)\}.
\]

The exact finite-dataset stability score is

\[
  \pi_{j,\lambda}^{(m)}
  =
  \frac{1}{|\Omega_m|}
  \sum_{I\in\Omega_m}h_{j,\lambda}(I),
  \qquad
  s_j^{(m)}=\max_{\lambda\in\Lambda}\pi_{j,\lambda}^{(m)}.
  \tag{1}
\]

Equation (1) is a complete finite-population U-statistic with a vector-valued,
binary kernel. It is:

* deterministic;
* invariant to row order;
* independent of a seed;
* exactly the expectation targeted by uniform random subsampling, conditional
  on the observed dataset.

The current score is the incomplete Monte Carlo version of (1). For one fixed
\((j,\lambda)\), conditional on the observed data,

\[
  \operatorname{Var}(\widehat\pi_{j,\lambda}\mid X,y)
  =
  \frac{\pi_{j,\lambda}^{(m)}
  (1-\pi_{j,\lambda}^{(m)})}{B}
  \le \frac{1}{4B}
\]

when subsamples are independent. Thus \(B=1000\) still permits a worst-case
conditional standard deviation of about \(0.0158\) for one score, before
maximizing over \(\lambda\), optimizing a threshold, and generating knockoffs.
A seed can change the final support whenever scores lie near a decision
boundary.

### 2.3 Determinism requirements

Equation (1) is deterministic only if all remaining choices are deterministic:

* the base solver uses a deterministic coordinate/update order;
* if the optimization problem has multiple minimizers, the returned minimizer
  is canonical (for example, minimize a strictly convex secondary norm over
  the primary objective's minimizer set);
* convergence tolerances and coefficient-zero thresholds are fixed;
* \(\Lambda\) and preprocessing are fixed;
* ties in coefficient support, score ranking, and threshold choice have
  declared rules;
* parallel reductions use integer counts, not order-dependent floating sums.

For selection indicators, each worker should return bitsets and the reducer
should sum integer counts. Scores are divided by the number of blocks only
after reduction. Row-order invariance additionally requires the preprocessing
and base solver to be row-order invariant; deterministic execution alone does
not imply that property.

## 3. Why exact enumeration is hard

At the paper's \(\alpha=1/2\),

\[
  {n\choose \lfloor n/2\rfloor}
  \sim \frac{2^n}{\sqrt{\pi n/2}}.
\]

For \(n=100\), this is approximately \(10^{29}\) base-learner fits per
regularization value. Exact generic enumeration is therefore not a practical
default.

This is not merely an implementation inconvenience. For an arbitrary black-box
selector, no strict subset of \(\Omega_m\) can determine (1) in general:
two selectors can agree on every evaluated subset and disagree on an
unevaluated subset. Consequently, a claim that \(B\ll {n\choose m}\)
deterministic blocks reproduce the complete score requires assumptions about
the selector or data. Balance alone does not prove it.

There are three honest computational levels.

### Level E: exact enumeration

Enumerate \(\Omega_m\) in combinadic rank order, fit every block, and accumulate
integer support counts. Split contiguous combinadic rank intervals among
workers. This parallelizes without communication except for final integer
reduction and is reproducible across worker counts.

Use this for small \(n\), for grouped datasets with few groups, and as a
reference oracle in tests.

### Level C: certified exact compression

Many subset fits can be compressed only when the base learner supplies a
certificate that their supports are identical.

For squared-loss Lasso, a subset fit depends on

\[
  G_I=X_I^\top X_I=\sum_{i\in I}x_ix_i^\top,\qquad
  c_I=X_I^\top y_I=\sum_{i\in I}x_iy_i.
\]

For candidate active set \(E\) and sign vector \(z_E\), the KKT conditions
(under a fixed objective scaling) are

\[
  \beta_E=G_{I,EE}^{-1}(c_{I,E}-\lambda z_E),\quad
  \operatorname{sign}(\beta_E)=z_E,\quad
  |c_{I,j}-G_{I,jE}\beta_E|\le\lambda\quad(j\notin E).
\]

Traverse a deterministic include/exclude tree of sample subsets. Interval
bounds for the remaining contributions to \(G_I\) and \(c_I\) can sometimes
certify these inequalities for every leaf below a node. Then add the node's
number of leaves to the support counts without fitting each leaf. If a
certificate fails, split the node. This remains exact; its worst case remains
exponential. Logistic, elastic-net, adaptive-Lasso, and sparse-group-Lasso
need their own valid certificates. A heuristic cache is not a certificate.

Warm starts along a deterministic regularization path and hashing identical
\((G_I,c_I)\) values are additional exact optimizations. Continuous omic data
may yield few exact hash collisions, so no large compression factor should be
assumed.

### Level D: deterministic balanced incomplete design

For large problems, choose a fixed block family
\(\mathcal D_B=\{I_1,\ldots,I_B\}\subset\Omega_m\) and compute

\[
  \widetilde\pi_{j,\lambda}
  =\frac1B\sum_{b=1}^B h_{j,\lambda}(I_b).
  \tag{2}
\]

Construct \(\mathcal D_B\) deterministically to minimize first- and second-order
inclusion discrepancy. With

\[
 r_i=\sum_b\mathbf1\{i\in I_b\},\qquad
 r_{i\ell}=\sum_b\mathbf1\{i,\ell\in I_b\},
\]

the uniform-design targets are

\[
 r_i^\star=\frac{Bm}{n},\qquad
 r_{i\ell}^\star=\frac{Bm(m-1)}{n(n-1)}.
\]

A concrete design objective is

\[
 \Phi(\mathcal D_B)=
 \sum_i(r_i-r_i^\star)^2+
 \eta\sum_{i<\ell}(r_{i\ell}-r_{i\ell}^\star)^2,
 \tag{3}
\]

with deterministic lexicographic tie-breaking. For \(m=\lfloor n/2\rfloor\),
blocks should be generated in complementary pairs whenever possible. Class and
group constraints are hard constraints in (3), not post-hoc redraw rules.

To make this constructive, assign each sampling unit a stable identifier and
build a nested design one pair at a time. For even \(n\), at step \(q\), choose

\[
 I_q\in\arg\min_{I\in\Omega_m}
 \Phi\!\left(\mathcal D_{2q-2}\cup\{I,I^c\}\right)
\rho\,\mathbf1\{I\text{ was already used}\},
\]

then append \(I_q,I_q^c\) without changing earlier pairs. For odd \(n\), use
two size-\(\lfloor n/2\rfloor\) disjoint blocks and rotate the omitted unit to
minimize first-order discrepancy. A mixed-integer solver can find the step
minimum for small problems. At larger scale, start from the least-replicated
units and perform deterministic best-improving swaps until no swap decreases
(3), breaking every tie by stable identifier. This coordinate-exchange
construction is deterministic and nested but only locally optimal; report the
achieved discrepancy, not an unproved optimality claim. Without stable sample
identifiers, this approximate design is reproducible for a fixed row order but
is not row-order invariant.

Equation (2) is reproducible and usually covers the cohort more evenly than
independent random draws, but it is an incomplete U-statistic. Report:

* maximum and RMS first-order discrepancy;
* maximum and RMS pairwise discrepancy;
* duplicate block count;
* score and support changes along nested budgets
  \(B_1<B_2<\cdots\).

Do not call (2) exact unless a selector-specific certificate proves equality
with (1). There is no distribution-free error bound for an arbitrary selector
based only on (3).

## 4. Removing random artificial features

### 4.1 What cannot be done validly

A single canonical cyclic shift, reverse ordering, fixed PRNG seed, or
deterministically chosen knockoff draw does not establish that artificial and
unknown null features are exchangeable.

The paper assumes exchangeability of the extended null set:

\[
  (Y,X_S,X^\sigma_{N\cup A})
  \overset d=(Y,X_S,X_{N\cup A})
\]

for every permutation \(\sigma\) of unknown null and artificial features. Its
FDP statements depend on this assumption. The paper also explicitly lists
exchangeability and artificial-feature generation as limitations.

There are only three mathematically honest choices:

1. retain a probabilistic null model and its assumptions;
2. exactly integrate over all of its random outcomes (finite but factorial for
   permutations, generally an intractable integral for Gaussian knockoffs);
3. remove FDP+ and adopt a deterministic objective with a different meaning.

D-Stabl chooses option 3. Option 2 is a useful small-data validation oracle, not
a scalable default.

### 4.2 MDL threshold

The stability scores induce only finitely many distinct candidate supports.
Let \(u_1>\cdots>u_L\) be the distinct values among \(s_1,\ldots,s_p\), and

\[
  S_\ell=\{j:s_j\ge u_\ell\}.
\]

Rather than scanning an arbitrary grid such as `np.arange(0, 1, .01)`, evaluate
exactly these candidate supports. Select the support that minimizes a
description-length criterion.

For Gaussian regression with an unpenalized refit and residual sum of squares
\(\operatorname{RSS}(S)\), use the extended BIC/MDL score

\[
  C(S)=
  n\log\!\left(\frac{\operatorname{RSS}(S)}{n}\right)
  +|S|\log n
  +2\gamma\log {p\choose |S|}.
  \tag{4}
\]

For a GLM, replace the first term by twice the negative maximized
log-likelihood:

\[
  C(S)=-2\ell(\widehat\beta_S;y,X_S)
  +|S|\log n
  +2\gamma\log {p\choose |S|}.
  \tag{5}
\]

The last term is the cost of describing which support among
\({p\choose |S|}\) possibilities was chosen. It is especially relevant when
\(p\gg n\). D-Stabl should expose \(\gamma\) because different consistency
results require different regimes; a default such as \(\gamma=1\) is a policy
choice, not a universal truth.

Candidate supports that cannot be fitted identifiably must be rejected or
evaluated by a separately specified, deterministic predictive code. The empty
support is always a candidate. Choose the minimizer deterministically by:

1. smallest \(C(S)\);
2. then smallest \(|S|\);
3. then lexicographically smallest feature-index tuple.

Equations (4)–(5) turn thresholding into compression: retain a feature only when
its predictive reduction in codelength pays for parameter and support
complexity. No random null features are needed.

This criterion does **not** estimate FDP. The resulting claim is:

> selected features form the shortest sparse predictive description among
> supports encountered on the deterministic stability path.

That statement is precise and testable. Calling the result FDR-controlled would
be false.

## 5. Proposed D-Stabl algorithm

Inputs:

* deterministic base selector \(\mathcal A\);
* fixed regularization grid \(\Lambda\);
* sampling unit (row or group);
* perturbation size \(m\);
* mode `exact`, `certified`, or `balanced`;
* for `balanced`, nested block budgets and fixed discrepancy weight \(\eta\);
* task-appropriate deterministic codelength and \(\gamma\).

Algorithm:

1. Validate data and define the admissible finite family \(\Omega_m\).
2. Build blocks:
   * `exact`: combinadic enumeration;
   * `certified`: deterministic branch-and-bound over the same enumeration;
   * `balanced`: solve (3), including class/group constraints.
3. For every block and \(\lambda\), fit \(\mathcal A_\lambda\) and return a
   support bitset.
4. Sum bitsets as integer counts and compute (1) or (2).
5. Compute \(s_j=\max_\lambda\pi_{j,\lambda}\).
6. Build one candidate support for each distinct \(s_j\), including the empty
   support.
7. Minimize (4), (5), or another explicitly declared deterministic code.
8. Refit the final predictive model on the selected support.
9. Return support, scores, block manifest, exactness status, discrepancy
   diagnostics, solver metadata, and codelength table.

Compact pseudocode:

```text
blocks, status = deterministic_blocks(data, mode, m, B)
counts[j, λ] = 0

parallel for I in blocks:
    for λ in Λ:
        support = deterministic_fit(X[I], y[I], λ)
        emit(bitset(support), λ)

counts = integer_tree_reduce(emitted_bitsets)
π = counts / number_of_blocks
s[j] = max_λ π[j, λ]

candidates = unique_nested_supports(s) ∪ {∅}
S* = argmin_S (code_length(S), |S|, lexicographic_indices(S))
return S*, π, diagnostics, status
```

## 6. Parallel and compression architecture

The workload is embarrassingly parallel by `(block, lambda)` but naïvely
materializing all results is wasteful.

* Represent each support by a packed bitset of \(p\) bits.
* Give each worker a deterministic combinadic rank interval or block-id range.
* Traverse \(\Lambda\) in a fixed order and warm-start only within a block.
* Maintain local integer count arrays; reduce them in a fixed binary tree.
* Checkpoint `(design hash, last block id, integer counts)`.
* Hash preprocessing, block manifests, solver version, tolerance, and feature
  order into the result.
* In certified mode, schedule branch-tree nodes; a certified node contributes
  its leaf multiplicity directly to integer counts.

Approximate balanced mode needs only count arrays of shape \(p\times R\), not
all \(B\times p\times R\) masks. Exact mode may stream combinadic blocks and
therefore does not need to store \({n\choose m}\) index sets.

Parallelism changes runtime, not the mathematical answer, because block
identities and integer addition are independent of scheduling.

## 7. Important code/paper inconsistencies to resolve

These are verified in the current checkout:

1. The paper writes selection as \(s_j\ge t\); `_get_support_mask` and FDP+
   use strict `>` (`stabl/stabl.py`, lines 1342–1343 and 1431–1445).
2. The paper minimizes over all thresholds in \([0,1]\); the code minimizes on
   a user grid, by default `np.arange(0., 1., .01)`, which excludes 1
   (lines 949–950 and 1431–1457).
3. The paper permits arbitrary minimizers; code takes the first grid minimizer.
   A deterministic specification needs an explicit tie rule.
4. `random_state` gives reproducibility for many paths, but it does not remove
   dependence on the chosen seed.
5. Group subsampling uses `GroupShuffleSplit`; its actual row count need not
   equal `n_subsamples` when group sizes differ.

D-Stabl should use a single declared comparison (`>=` is natural with exact
score levels), candidate thresholds equal to observed score levels, and the
MDL tie rules above.

## 8. Guarantees and non-guarantees

### Exact mode guarantees

Conditional on the observed data and deterministic solver:

* exact equality to the uniform average over the declared admissible family;
* row-order invariance if preprocessing, admissibility, and the base solver are
  themselves row-order invariant;
* seed independence;
* reproducibility across worker counts;
* no Monte Carlo error.

### Balanced mode guarantees

* seed independence and reproducibility;
* declared first-/second-order block balance;
* exact averaging over the declared design.

It does not guarantee equality to complete-subset stability for an arbitrary
selector.

### MDL threshold guarantees

* deterministic minimization over all distinct stability-path supports;
* explicit trade-off between fit, parameter count, and support identity.

It does not guarantee FDR control or recovery of the true biological support.

### Fundamental limit

Let \(D=(X,y)\) be any finite observed dataset and let a feature \(j\) have the
observed values in \(D\). There can be two data-generating mechanisms that
assign positive probability to exactly \(D\), one in which \(j\) is genuinely
associated with future outcomes and one in which its observed association is
accidental. Any deterministic algorithm receives the same \(D\) in both cases
and must return the same answer. Therefore no finite-data deterministic
algorithm can identify the true informative set for all distributions.
Randomization does not remove this identifiability limit either.

The valid goal is deterministic algorithmic output with guarantees stated
relative to a declared perturbation family and model assumptions.

## 9. Implementation sequence

1. Introduce a `subsample_design` interface that returns an iterable plus an
   `exactness` record.
2. Implement combinadic exact enumeration and grouped exact enumeration.
3. Add deterministic solver validation and packed-bitset integer aggregation.
4. Implement score-level candidate supports and regression/GLM MDL criteria.
5. Implement complementary balanced designs with discrepancy diagnostics.
6. Add nested-budget convergence reports.
7. Add Lasso KKT certification as an optional accelerator; fall back to exact
   splitting whenever certification fails.
8. Keep legacy stochastic Stabl as an explicitly named compatibility mode,
   rather than silently changing its published semantics.

Minimum validation:

* exact scores equal hand enumeration on small regression and classification
  datasets;
* row permutations leave exact output unchanged;
* worker counts leave every integer count unchanged;
* balanced designs meet reported inclusion counts;
* nested budgets are prefixes of one canonical design;
* score ties and MDL ties follow the declared rules;
* grouped designs never split a group;
* stochastic Stabl and D-Stabl are benchmarked separately, without claiming
  identical targets.

## 10. Decision summary

| Requirement | Proposed mechanism | Qualification |
|---|---|---|
| No seed dependence | complete or fixed-design averaging | exact only for complete/certified mode |
| No random feature runs | remove artificial features | FDP+ is removed too |
| Data-driven threshold | MDL/extended information criterion | selects compressive predictive support, not target FDR |
| Scalability | balanced incomplete design, bitsets, warm paths, parallel blocks | approximation is disclosed |
| Exact acceleration | sufficient-statistic caching and KKT branch certificates | selector-specific; exponential worst case |
| Grouped cohorts | enumerate/design over groups | row fraction may vary |
| Reproducibility | manifests, integer reductions, explicit ties | requires deterministic base solver |

## References

* Hédou et al., “Discovery of sparse, reliable omic biomarkers with Stabl,”
  *Nature Biotechnology* 42, 1581–1593 (2024),
  [doi:10.1038/s41587-023-02033-x](https://doi.org/10.1038/s41587-023-02033-x).
* Meinshausen and Bühlmann, “Stability selection,” *JRSS B* 72, 417–473
  (2010), [doi:10.1111/j.1467-9868.2010.00740.x](https://doi.org/10.1111/j.1467-9868.2010.00740.x).
* Shah and Samworth, “Variable selection with error control: another look at
  stability selection,” *JRSS B* 75, 55–80 (2013),
  [doi:10.1111/j.1467-9868.2011.01034.x](https://doi.org/10.1111/j.1467-9868.2011.01034.x).
* Blom, “Some properties of incomplete U-statistics,” *Biometrika* 63,
  573–580 (1976), [doi:10.1093/biomet/63.3.573](https://doi.org/10.1093/biomet/63.3.573).
* Chen and Chen, “Extended Bayesian information criteria for model selection
  with large model spaces,” *Biometrika* 95, 759–771 (2008),
  [doi:10.1093/biomet/asn034](https://doi.org/10.1093/biomet/asn034).
