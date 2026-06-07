# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulator import SimpleFAIR


# %% Load observational data
# Berkeley Earth annual GMST anomaly
obs = pd.read_csv("data/berkeley_annual_gmst.csv")
obs_years = obs.year.values
obs_gmst = obs.gmst_anomaly.values       # K, relative to 1850-1900
obs_unc = obs.gmst_uncertainty.values     # 1-sigma uncertainty (K)


# %% Plot Berkeley Earth observations
fig, ax = plt.subplots(figsize=(6, 4))
ax.fill_between(obs_years, obs_gmst - obs_unc, obs_gmst + obs_unc,
                alpha=0.2, color="C3")
ax.plot(obs_years, obs_gmst, color="C3", linewidth=1)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("GMST anomaly (K)\nrel. 1850-1900")
ax.set_title("Berkeley Earth annual GMST anomaly")
plt.tight_layout()
plt.show()


# %% Configuration
# How far into the future to simulate
simulate_up_to = 2100

# Prescribe observed GMST up to this year; free-run after
conditioning_up_to = 2000

# Number of stochastic ensemble members per ECS value
n_ensemble = 20

# ECS values to sweep (K)
ecs_values = [2.0, 3.0, 4.0]


# %% Run stochastic ensembles for each ECS value
# Build the FAIR template once (~1s)
template = SimpleFAIR(end_year=simulate_up_to, stochastic=True)

# Extract the conditioning data from Berkeley Earth
conditioning_years = obs_years[obs_years <= conditioning_up_to]
conditioning_gmst = obs_gmst[obs_years <= conditioning_up_to]

# Run ensembles: for each ECS, run n_ensemble stochastic members
# conditioned on observed GMST up to specified year
ensembles = {}
for ecs in ecs_values:
    runs = []
    for seed in tqdm(range(n_ensemble),
                     desc=f"ECS={ecs:.0f}K",
                     leave=True):
        m = template.copy(seed=seed)
        m.set_ecs(ecs)
        m.prescribe_gmst(conditioning_years, conditioning_gmst)
        m.run()
        t, gmst = m.get_gmst()
        runs.append((t, gmst))
    ensembles[ecs] = runs


# %% Plot: ensemble projections vs observations
colors = {2.0: "C0", 3.0: "C1", 4.0: "C2"}

fig, ax = plt.subplots(figsize=(8, 5))

# Shade the conditioning period (where GMST is prescribed)
ax.axvspan(1850, conditioning_up_to, alpha=0.05, color="gray")
ax.axvline(conditioning_up_to, color="gray", linestyle="--", alpha=0.5)

# Plot ensemble members + mean for each ECS
for ecs, runs in ensembles.items():
    all_gmst = np.array([gmst for _, gmst in runs])
    mean = all_gmst.mean(axis=0)
    t = runs[0][0]
    c = colors[ecs]

    # Individual stochastic members
    for ti, gmst in runs:
        ax.plot(ti, gmst, alpha=0.1, color=c, linewidth=0.8)

    # Ensemble mean
    ax.plot(t, mean, color=c, linewidth=1, label=f"ECS = {ecs:.0f} K")

# Berkeley Earth observations with uncertainty band
obs_mask = obs_years >= 1850
ax.fill_between(obs_years[obs_mask],
                (obs_gmst - obs_unc)[obs_mask],
                (obs_gmst + obs_unc)[obs_mask],
                alpha=0.15, color="C3")
ax.plot(obs_years[obs_mask], obs_gmst[obs_mask],
        color="C3", linewidth=1.5, label="Berkeley Earth")

ax.set_xlim(1850, simulate_up_to)
ax.set_xlabel("Year")
ax.set_ylabel("GMST anomaly (K)\nrel. 1850-1900")
ax.set_title(f"{n_ensemble} members per ECS\n"
             f"conditioned on Berkeley Earth 1850-{conditioning_up_to}")
ax.legend(loc="upper left", fontsize=12, frameon=False)
plt.tight_layout()
plt.show()

# %%
