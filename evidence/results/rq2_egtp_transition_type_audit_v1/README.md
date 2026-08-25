# EGTP transition-type audit v1

This is a post-hoc, secondary, descriptive audit based on the already-frozen
test protocol. It does not change time-based TP/FP/FN, Boundary F1, method
selection, or k=0.6.

At +/-10 seconds, the validation-selected EGTP had
37 time-matched boundary pairs pooled across the
three training seeds. Conditional on those matches, the from-phase, to-phase,
and exact transition-type correctness rates were
0.595,
0.514, and
0.405, respectively.

These conditional rates must always be reported with Boundary Recall because
missed boundaries are absent from the denominator. They are not primary
performance metrics and do not support a confirmatory claim.
