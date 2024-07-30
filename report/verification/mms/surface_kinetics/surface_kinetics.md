---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: festim-env
  language: python
  name: python3
---

# Surface Kinetics MMS

Author: Vladimir Kulagin

```{tags} 1D, MMS, SurfaceKinetics, transient
```

This MMS case verifies the implementation of the `SurfaceKinetics` boundary condition. We will consider a transient case of hydrogen diffusion on domain $\Omega: x\in[0,1] \cup t\in[0, 5]$ with a homogeneous diffusion coefficient $D$, and a Dirichlet boundary condition on the rear domain side.

+++

The problem is:

\begin{align*}
    &\dfrac{\partial c_\mathrm{m}}{\partial t} = \nabla\cdot\left(D\nabla c_\mathrm{m} \right) + S \quad \textrm{ on } \Omega, \\
    &-D \nabla c_\mathrm{m} \cdot \mathbf{n} = \lambda_{\mathrm{IS}} \dfrac{\partial c_{\mathrm{m}}}{\partial t} + J_{\mathrm{bs}} - J_{\mathrm{sb}} \quad \textrm{ at } x=0, \\
    &c_\mathrm{m} = c_\mathrm{m, 0} \quad \textrm{ at } x=1, \\
    &c_\mathrm{m} = c_\mathrm{m, 0} \quad \textrm{ at } t=0, \\
    &\dfrac{d c_\mathrm{s}}{d t} = J_{\mathrm{bs}} - J_{\mathrm{sb}} + J_{\mathrm{vs}}  \quad \textrm{ at } x=0, \\
    &c_\mathrm{s}= c_\mathrm{s, 0}\quad \textrm{ at } t=0, \\
\end{align*}

with $J_{\mathrm{bs}} = k_{\mathrm{bs}} c_{\mathrm{m}} \lambda_{\mathrm{abs}} \left(1 - \dfrac{c_\mathrm{s}}{n_{\mathrm{surf}}}\right)$, $J_{\mathrm{sb}} = k_{\mathrm{sb}} c_{\mathrm{s}} \left(1 - \dfrac{c_{\mathrm{m}}}{n_\mathrm{IS}}\right)$, $\lambda_{\mathrm{abs}}=n_\mathrm{surf}/n_\mathrm{IS}$.

The manufactured exact solution for mobile concentration is:
\begin{equation*}
c_\mathrm{m, exact}=1+2x^2+x+2t.
\end{equation*}

For this problem, we choose:
\begin{align*}
& k_{\mathrm{bs}}=1/\lambda_{\mathrm{abs}} \\
& k_{\mathrm{sb}}=2/\lambda_{\mathrm{abs}} \\
& n_{\mathrm{IS}} = 20 \\
& n_{\mathrm{surf}} = 5 \\
& D = 5 \\
& \lambda_\mathrm{IS} = 2
\end{align*}

Injecting these parameters and the exact solution for solute H, we obtain:

\begin{align*}
& S = 2(1-2D) \\
& J_{\mathrm{vs}}=2n_\mathrm{surf}\dfrac{2n_\mathrm{IS}+2\lambda_\mathrm{IS}-D}{(2n_\mathrm{IS}-1-2t)^2}+2\lambda_\mathrm{IS}-D \\
& c_\mathrm{s, exact}=n_\mathrm{surf}\dfrac{1+2t+2\lambda_\mathrm{IS}-D}{2n_\mathrm{IS}-1-2t} \\
& c_\mathrm{s,0}=c_\mathrm{s, exact} \\
& c_\mathrm{m,0}=c_\mathrm{m, exact}
\end{align*}

We can then run a FESTIM model with these values and compare the numerical solutions with $c_\mathrm{m, exact}$ and $c_\mathrm{s, exact}$.

```{code-cell} ipython3
# implementation details from
# https://github.com/KulaginVladimir/FESTIM-SurfaceKinetics-Validation/blob/main/MMS/MMS.ipynb

import festim as F
import matplotlib.pyplot as plt
import numpy as np

# Create the FESTIM model
model = F.Simulation()

model.mesh = F.MeshFromVertices(np.linspace(0, 1, 1000))

# Variational formulation
n_IS = 20
n_surf = 5
D = 5
lambda_IS = 2
k_bs = n_IS / n_surf
k_sb = 2 * n_IS / n_surf

solute_source = 2 * (1 - 2 * D)

exact_solution_cm = lambda x, t: 1 + 2 * x**2 + x + 2 * t
exact_solution_cs = (
    lambda t: n_surf * (1 + 2 * t + 2 * lambda_IS - D) / (2 * n_IS - 1 - 2 * t)
)

solute_source = 2 * (1 - 2 * D)

model.sources = [F.Source(solute_source, volume=1, field="solute")]


def J_vs(T, surf_conc, t):
    return (
        2 * n_surf * (2 * n_IS + 2 * lambda_IS - D) / (2 * n_IS - 1 - 2 * t) ** 2
        + 2 * lambda_IS
        - D
    )


model.boundary_conditions = [
    F.DirichletBC(surfaces=[2], value=exact_solution_cm(x=F.x, t=F.t), field="solute"),
    F.SurfaceKinetics(
        k_sb=k_sb,
        k_bs=k_bs,
        lambda_IS=lambda_IS,
        n_surf=n_surf,
        n_IS=n_IS,
        J_vs=J_vs,
        surfaces=1,
        initial_condition=exact_solution_cs(t=0),
        t=F.t,
    ),
]

model.initial_conditions = [
    F.InitialCondition(field="solute", value=exact_solution_cm(x=F.x, t=F.t))
]

model.materials = F.Material(id=1, D_0=D, E_D=0)

model.T = 300  # this is ignored since no parameter is T-dependent

model.settings = F.Settings(
    absolute_tolerance=1e-10, 
    relative_tolerance=1e-10, 
    transient=True,
    final_time=5
)

export_times = [1, 3, 5]
model.dt = F.Stepsize(
    initial_value=5e-3, 
    milestones=export_times
)

derived_quantities = F.DerivedQuantities([F.AdsorbedHydrogen(surface=1)])
model.exports = [
    F.TXTExport("solute", filename="./mobile_conc.txt", times=export_times),
    derived_quantities
]

model.initialise()
model.run()
```

## Comparison with exact solution

```{code-cell} ipython3
def norm(x, c_comp, c_ex):
    return np.sqrt(np.trapz(y=(c_comp - c_ex) ** 2, x=x))


data = np.genfromtxt("mobile_conc.txt", names=True, delimiter=",")
for t in export_times:
    x = data["x"]
    y = data[f"t{t:.2e}s".replace(".", "").replace("+", "")]
    # order y by x
    x, y = zip(*sorted(zip(x, y)))

    (l1,) = plt.plot(
        x,
        exact_solution_cm(np.array(x), t),
        label=f"exact",
    )

    point_cnt = 20
    step = len(x) // point_cnt
    plt.scatter(
        x[::step],
        y[::step],
        label=f"t = {t}s",
        color=l1.get_color(),
        alpha=0.6,
    )

    print(
        f"L2 error for c_m at t = {t}s: {norm(np.array(x), np.array(y), exact_solution_cm(x=np.array(x), t=t))}"
    )

plt.legend(reverse=True)
plt.ylabel("$c_m \\ $")
plt.xlabel("$x \\ (m)$")
plt.show()
```

```{code-cell} ipython3
c_s_computed = derived_quantities[0].data
t = derived_quantities[0].t

print(f"L2 error for c_s: {norm(t, c_s_computed, exact_solution_cs(t=np.array(t)))}")

plt.figure()

# plot computed
point_cnt = 16
step = len(t) // point_cnt
plt.scatter(t[::step], c_s_computed[::step], label="computed", alpha=0.6)

#plot exact
plt.plot(t, exact_solution_cs(np.array(t)), label="exact")

plt.ylabel("$c_s \\ (H \\ m^{-2})$")
plt.xlabel("$t \\ (s)$")
plt.legend()
plt.ylim(bottom=0)
plt.show()
```
