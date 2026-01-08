import pickle
import arviz as az
from matplotlib import pyplot as plt

trace_file_path = "trace.pkl"

with open(trace_file_path, 'rb') as trace_file:
    trace = pickle.load(trace_file)
# -----------------------------
# 4. ArviZ Visualizations
# -----------------------------
az.summary(trace, round_to=2)
az.summary(trace, kind="stats")
az.plot_trace(trace)
# Posterior parameter distributions
az.plot_posterior(trace, var_names=["alpha", "lf_M"])
plt.show()
# Posterior predictive checks (built-in)