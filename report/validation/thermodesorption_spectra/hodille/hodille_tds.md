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

# Deuterium retention in tungsten

```{tags} 1D, TDS, trapping, transient
```

This validation case is a thermo-desorption spectrum measurement perfomed by Hodille et al. TODO: INSERT CITATION

Deuterium ions at 200 eV were implanted in a 0.5 mm thick sample of high purity tungsten foil (PCW).

The ion beam with an incident flux of $2.5 \times 10^{19} \ \mathrm{D \ m^{-2} \ s^{-1}}$ was turned on for 400 s which corresponds to a fluence of $1.0 \times 10^{22} \ \mathrm{D \ m^{-2}}$

The diffusivity of tungsten in the FESTIM model is as measured by Frauenfelder {cite}`frauenfelder_permeation_1968`.

To reproduce this experiment, three traps are needed: 2 intrinsic traps and 1 extrinsic trap.
The extrinsic trap represents the defects created during the ion implantation.

The time evolution of extrinsic traps density $n_i$ expressed in $\text{m}^{-3}$ is defined as:
\begin{equation}
    \frac{dn_i}{dt} = \varphi_0\:\left[\left(1-\frac{n_i}{n_{a_{max}}}\right)\:\eta_a \:f_a(x)+\left(1-\frac{n_i}{n_{b_{max}}}\right)\:\eta_b \:f_b(x)\right]
\end{equation}

+++

## FESTIM code

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import festim as F
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt



model = F.Simulation()

vertices = np.concatenate(
    [
        np.linspace(0, 1e-9, num=500),
    ]
)

model.mesh = F.MeshFromVertices(vertices)
# Material Setup, only W
tungsten = F.Material(
    id=1,
    D_0=4.1e-07,  # m2/s
    E_D=0.39,  # eV
)

model.materials = tungsten
import sympy as sp

# ### Temperature Settings ### #

storage_temp = 300 # K
exposure1_temp = 600 # K <--------------- experiment variable
exposure2_temp = 600 # K
temperature_ramp = 15 / 60 # K/s
hold_temp = 1323 # K

exposure1_time = 4 * 3600 # s
stop_exposure1 = exposure1_time
stop_storage1 = stop_exposure1 + (exposure1_temp - storage_temp) / temperature_ramp # s

exposure2_time = 19 * 3600  # s
stop_exposure2 = stop_storage1 + exposure2_time
stop_storage2 = stop_exposure2 + (exposure2_temp - storage_temp) / temperature_ramp

stop_tds = stop_storage2 + (hold_temp - storage_temp) / temperature_ramp

temp_function = sp.Piecewise(
    (exposure1_temp, F.t < stop_exposure1),
    (exposure1_temp - temperature_ramp * (F.t - stop_exposure1), F.t < stop_storage1),
    (exposure2_temp, F.t < stop_exposure2),
    (exposure2_temp - temperature_ramp * (F.t - stop_exposure2), F.t < stop_storage2),
    (storage_temp + temperature_ramp * (F.t - stop_storage2), F.t < stop_tds),
    (hold_temp, True),
)
model.T = F.Temperature(value=temp_function)

# ### Source Settings ### #

three_min = 3 * 60 # s
incident_flux = 5.4e18  # D m^2 s^-1, beam strength from paper
ion_flux = sp.Piecewise(
    (incident_flux, F.t < stop_exposure1 + three_min),
    (0, F.t < stop_storage1),
    (incident_flux, F.t < stop_exposure2 + three_min),
    (0, True)
)

source_term = F.ImplantationFlux(
    flux=ion_flux, imp_depth=4.5e-9, width=2.5e-9, volume=1  # H/m2/s  # m  # m
)

model.sources = [source_term]

# TODO: fit f(x)

# ### Trap Settings ### #

w_atom_density = 6.3e28  # atom/m3
w_ion_flux = 9.7e13

# Undamaged material traps
trap_1 = F.Trap(
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k=0.39,
    p_0=1e13,
    E_p=0.85,
    density= 0.01 * w_atom_density,
    materials=tungsten,
)

trap_2 = F.Trap(
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k=0.39,
    p_0=1e13,
    E_p=1.0,
    density= 0.01 * w_atom_density,
    materials=tungsten,
)

# Damage traps

center = 4.5e-9
width = 2.5e-9
distribution = (
    1 / (width * (2 * sp.pi) ** 0.5) * sp.exp(-0.5 * ((F.x - center) / width) ** 2)
)

trap_3 = F.Trap(
    k_0 = 4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k = 0.39,
    p_0 = 1e13,
    E_p = 1.83,
    density = 0.16 * w_atom_density, # <--------------- experiment variable
    materials = tungsten,
)

trap_4 = F.Trap(
    k_0 = 4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k = 0.39,
    p_0 = 1e13,
    E_p = 2.10,
    density = 0.085 * w_atom_density, # <--------------- experiment variable
    materials =tungsten
)

model.traps = [trap_1, trap_2, trap_3, trap_4]


# ### Boundary Conditions ### 
model.boundary_conditions = [F.DirichletBC(surfaces=[1, 2], value=0, field=0)]

min_temp, max_temp = 500, 1200

model.dt = F.Stepsize(
    initial_value=0.3,
    stepsize_change_ratio=1.05,
    max_stepsize=lambda t: 5 if t > stop_storage2 else None,
    dt_min=1e-05,
    milestones=[stop_exposure1, stop_storage1, stop_exposure2, stop_storage2, stop_tds]
)

model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-010,
    final_time = stop_tds + 0.5 * 3600 # time to reach max temp
)

derived_quantities = F.DerivedQuantities(
    [
        F.TotalVolume("solute", volume=1),
        F.TotalVolume("retention", volume=1),
        F.TotalVolume("1", volume=1),
        F.TotalVolume("2", volume=1),
        F.TotalVolume("3", volume=1),
        F.TotalVolume("4", volume=1),
        F.HydrogenFlux(surface=1),
        F.HydrogenFlux(surface=2),
    ],
)

model.exports = [derived_quantities]

model.initialise()
model.run()
```

## Comparison with experimental data

The results produced by FESTIM are in good agreement with the experimental data. The grey areas represent the contribution of each trap to the global TDS spectrum.

```{code-cell} ipython3
:tags: [hide-input]

t = derived_quantities.t
flux_left = derived_quantities.filter(fields="solute", surfaces=1).data
flux_right = derived_quantities.filter(fields="solute", surfaces=2).data
flux_total = -np.array(flux_left) - np.array(flux_right)

t = np.array(t)

# plotting simulation data
plt.plot(t, flux_total, linewidth=3, label="FESTIM")

# plotting trap contributions
""" trap_data = [derived_quantities.filter(fields=f"{i}").data for i in range(1, 5)]
contributions = [-np.diff(trap) / np.diff(t) for trap in trap_data]

colors = [(0.9*(i % 2), 0.2*(i % 4), 0.4*(i % 3)) for i in range(6)]

for i, cont in enumerate(contributions):
    label = f"Trap {i + 1}"
    plt.plot(temp[1:], cont, linestyle="--", color=colors[i], label=label)

    plt.fill_between(temp[1:], 0, cont, facecolor="grey", alpha=0.1) """


# plotting original data
""" experimental_tds = np.genfromtxt("ogorodnikova-original.csv", delimiter=",")
experimental_temp = experimental_tds[:, 0]
experimental_flux = experimental_tds[:, 1]
plt.scatter(experimental_temp, experimental_flux, color="green", label="original", s=16) """

plt.legend()
""" plt.xlim(min_temp, max_temp)
plt.ylim() """
plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Temperature (K)")

plt.show()
```

```{note}
The experimental data was taken from Figure 5 of the original experiment paper {cite}`ogorodnikova_deuterium_2003` using [WebPlotDigitizer](https://automeris.io/)
```
