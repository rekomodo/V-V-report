---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: vv-festim-report-env
  language: python
  name: python3
---

# TMAP7 V&V Val-1da

```{tags} 1D, MES, transient
```

This verification case from TMAP7's V&V report {cite}`ambrosek_verification_2008` consists of a slab of depth $l = 1 \times 10^{-3} \ \mathrm{m}$ with one trap under a strong trapping regime.

The trap reaches 99% concentration at its breakthrough time 
$$
\tau = \frac{l^2 \rho}{2 c_m D}
$$

$\rho$ is the trapping site fraction, \
$c_m (\text{atom} \ \mathrm{m}^{-3})$ is the mobile atom concentration.

+++

## FESTIM Code

```{code-cell} ipython3
:tags: [hide-cell]

# referenced https://github.com/gabriele-ferrero/Titans_TT_codecomparison/blob/main/Festim_models/WeakTrap.py

import festim as F
import numpy as np
import matplotlib.pyplot as plt

D_0 = 1.9e-7
N_A = 6.0221408e23
rho_w = 6.3382e28
rho = 1e-3
E_k = 0.2
E_p = 2.5
T = 1000
D = D_0 * np.exp(-E_k / (F.k_B * T))
S = 2.9e-5 * np.exp(-1 / (F.k_B * T))
sample_depth = 1e-3

c_m = (1e5) ** 0.5 * S * 1.0525e5

model = F.Simulation()
model.mesh = F.MeshFromVertices(vertices=np.linspace(0, sample_depth, num=1001))
model.materials = F.Material(id=1, D_0=D_0, E_D=E_k)
model.T = F.Temperature(value=T)

model.boundary_conditions = [
    F.DirichletBC(surfaces=1, value=c_m * N_A, field=0),
    F.DirichletBC(surfaces=2, value=0, field=0),
]

trap = F.Trap(
    k_0=1.58e7 / N_A,
    E_k=E_k,
    p_0=1e13,
    E_p=E_p,
    density=1e-3 * rho_w,
    materials=model.materials[0],
)

model.traps = [trap]

model.settings = F.Settings(
    absolute_tolerance=1e10, relative_tolerance=1e-10, final_time=1e6  # s
)

model.dt = F.Stepsize(
    initial_value=1e-2,
    dt_min=1e-3,
    stepsize_change_ratio=1.1,
    max_stepsize=lambda t: 250 if t > 1.5e5 else None
)

derived_quantities = F.DerivedQuantities([F.HydrogenFlux(surface=2)])
model.exports = [derived_quantities]

model.initialise()
model.run()
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-input]

# plot computed solution
t = np.array(derived_quantities.t)
computed_solution = derived_quantities.filter(surfaces=2).data
plt.plot(t, np.abs(computed_solution) / 2, label="FESTIM", linewidth=3)

# plot exact solution
tau = sample_depth**2 * rho / (2 * c_m * D) * rho_w / N_A

plt.axvline(tau, color="red", label="exact")

plt.xlabel("Time (s)")
plt.ylabel("Downstream flux (H/m2/s)")

plt.legend()
plt.show()
```

```{code-cell} ipython3

```
