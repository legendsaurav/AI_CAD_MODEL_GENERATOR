# Primitive recovery module — production implementations
#
# Components:
#   fitting.py     — LM and RANSAC primitive fitting (mathematically rigorous)
#   uncertainty.py — MC Dropout / Ensemble uncertainty estimation
#   generator.py   — Primitive proposal generation
#   estimator.py   — Neural parameter estimation
#   optimizer.py   — Full optimization pipeline (dispatches to fitting.py)
