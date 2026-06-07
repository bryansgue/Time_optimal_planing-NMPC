# Response to Reviewers

**Manuscript #:** Access-2026-22287
**Title:** Point Mass Planning and NMPC Tracking for Time-Optimal Agile Quadrotor Gate Navigation
**Authors:** Bryan S. Guevara, Alexandre Santos Brandão, José Varela-Aldás

We thank the Associate Editor and both reviewers for their careful reading
and constructive feedback. We are encouraged that both reviewers found the
work technically sound and a contribution to the body of knowledge, and we
have addressed every comment below. For each concern we give **(a)** the
reviewer's concern, **(b)** our response, and **(c)** the action taken.
All changes are marked in yellow in the highlighted PDF. Section and
equation labels refer to the revised manuscript.

In addition to the reviewer-driven changes, during the revision we
performed a full internal audit of the reported data and corrected two
items for accuracy and transparency; these are listed at the end under
*Additional corrections by the authors*.

---

## Reviewer 1

The reviewer states that we "have identified 2 significant research gaps in
the open literature and addressed those gaps with the current study," and
answered *Yes* to contribution to the body of knowledge, technical
soundness, and reference adequacy. We thank the reviewer for this positive
assessment and address the two suggestions for improvement below.

### Comment 1.1 — *"It would be better if the concepts were explained in a more detailed manner."* (Q2, technical soundness)

**(a) Concern.** The planner and controller are described with limited (but
sufficient) technical detail; more explanation would improve the paper.

**(b) Response.** We agree that additional insight strengthens the
formulation without lengthening it unduly. Rather than expand every
derivation, we added explanation at the single point where the formulation
was previously stated without motivation — the structure of the point-mass
cost function — because that term is what links the planner to the quality
of the reference handed to the tracker, and therefore carries the most
conceptual weight for the paper's thesis.

**(c) Action.** Added a sentence after Eq. (the PMM cost,
Section III-C, `eq:pmm_cost`) explaining the role of the three cost terms,
and in particular that the jerk regulariser ($w_j$) is what keeps the
downstream flatness derivatives (attitude quaternion and body rate) well
conditioned, since they are obtained by differentiating the planned
acceleration. This makes explicit how the planner cost couples to the
reference completeness that is the central theme of the paper.

### Comment 1.2 — *"The mathematical formulations could be explained with more insights. e.g., there is no mentioning about the criterion for setting the NMPC parameters such as N, Fc, ..."* (Q3, comprehensiveness)

**(a) Concern.** No criterion is given for selecting the NMPC parameters
(horizon $N$, control rate $f_c$, etc.).

**(b) Response.** We thank the reviewer for pointing this out. The values
were not chosen by a separate tuning study but follow standard practice and
the real-time budget, which we now state explicitly. The control rate
$f_c=100$ Hz is the conventional rate for body-rate agile NMPC in the cited
literature; the horizon $N=50$ ($T_h=0.5$ s) is the largest horizon for
which our implementation keeps the 99th-percentile solve time within the
10 ms control deadline on the test platform (a figure already reported in
the computational-performance results), and the resulting half-second
look-ahead spans approximately one inter-gate transit
($T_f^*/N_g \approx 0.7$ s).

**(c) Action.** Added an explanatory passage in Section V-A
(Experimental Setup), immediately after the parameter listing, giving the
criterion for $f_c$ and $N$/$T_h$ and linking the horizon choice to the
measured real-time budget (Section V-G).

### Reviewer 1 — remaining items

Questions 1 (contribution), 4 (references applicable/sufficient), and 5 (no
inappropriate references) were answered affirmatively and required no
action. No references were suggested for addition or removal.

---

## Reviewer 2

The reviewer found the paper "very well written, well-founded and
referenced," with results that "contribute to the body of knowledge on
quadcopter UAV control," and recommended publication after minor
adjustments. We thank the reviewer and address the single comment below.

### Comment 2.1 — *"The paper would benefit if the authors included a brief discussion of the proposal's contribution to similar works cited in the paper. It would be interesting to highlight the advantages and contributions of the proposal, citing previously published works."*

**(a) Concern.** A brief, explicit discussion positioning the proposal
against similar cited works, highlighting its advantages, would strengthen
the paper.

**(b) Response.** We agree. While the Related Work section surveys the
field, the paper did not consolidate, in one place, how the proposed
pipeline compares with the closest cited approaches. We added such a
discussion using only references already present in the manuscript (no new
citations), contrasting the proposal with the PMM-plus-MPCC pairing of
Foehn et al., the fused MPCC formulations of Romero et al. and Krinner et
al., and the fixed-reference NMPC-vs-DFBC benchmark of Sun et al.

**(c) Action.** Added a paragraph at the end of Section V-H (Discussion)
explicitly positioning the contribution: it retains the fast point-mass
planning of Foehn et al. while feeding the *complete* flat output to the
NMPC; it keeps the planner and tracker modular (unlike the fused MPCC of
Romero et al. / Krinner et al., whose stages cannot be replaced
independently) while recovering the tracking completeness those fused
methods achieve implicitly; and, unlike Sun et al., it *ablates* the
reference structure to isolate which flat components drive gate-crossing
performance. The net positioning — modularity of decomposed approaches with
the reference completeness usually exclusive to fused ones, at no extra
solver cost — is stated explicitly.

### Reviewer 2 — remaining items

All four assessment questions were answered affirmatively and required no
action. No references were suggested for addition or removal.

---

## Additional corrections by the authors

During the revision we audited every quantitative claim against the raw
result files. Tables III and IV, all reported percentages (RMSE and
gate-crossing-distance reductions, crossing rates), peak speeds, and peak
accelerations were confirmed correct and unchanged. Two items were
corrected:

**AC1 — Blow-up (divergence) rate.** The original text stated the
per-controller blow-up rate as "5–15% per mode." On re-checking the raw
trials, the accurate per-controller rates are 0% (NMPC-Att) and 8%
(NMPC-Full) on the figure-8, and 16% for both controllers on the more
demanding vertical loop, leaving 23/25 and 18/25 paired-valid trials
respectively. Section V-D now reports these exact figures and the
paired-exclusion rule transparently.

**AC2 — Figure 2 (evaluation circuits).** The original Figure 2 had been
rendered from an earlier planner export and from the model-in-the-loop
trajectory set, so its underlying numbers did not match the
software-in-the-loop reference and results reported in the tables. The
figure has been regenerated from exactly the data the paper reports: the
point-mass reference of Table III ($T_f^*=5.55$ s / $5.40$ s, peak speed
16.55 / 12.10 m/s) and the software-in-the-loop tracked trajectories. The
figure now shows the same circuits and quantities as the rest of the
manuscript.

---

## Note on authorship

The author list of this revision differs from the original submission. As
required, a *Request for Byline Change* form with a detailed justification
of each author's contribution is submitted separately with this
resubmission.

We believe the revised manuscript fully addresses the reviewers' comments
and thank the reviewers and Associate Editor for helping us improve the
paper.
