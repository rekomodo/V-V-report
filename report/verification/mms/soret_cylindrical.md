---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: vv-festim-report-env
  language: python
  name: python3
---

# Soret Effect - Cylindrical

```{tags} 2D, MMS, steady state, cylindrical, spherical, soret
```

This MMS case verifies the implementation of the Soret effect in FESTIM on cylindrical meshes.
We will only consider diffusion of hydrogen in a unit square domain $\Omega$ s.t. $r \in [1, 2]$ and $z \in [0, 1]$ at steady state with a homogeneous diffusion coefficient $D$.
Moreover, a Dirichlet boundary condition will be assumed on the boundaries $\partial \Omega $.

The problem is therefore:

$$
\begin{align}
    & - \nabla\cdot\vec{\varphi} = -S  \quad \text{on }  \Omega  \\
    & \vec{\varphi} = -D \ \nabla{c} - D\frac{Q^* c}{k_B T^2} \ \nabla{T} \\
    & c = c_0 \quad \text{on }  \partial \Omega
\end{align}
$$(problem_simple_soret_cylindrical)

The manufactured exact solution for mobile concentration is:

$$
\begin{equation}
    c_\mathrm{exact} = 1 + 3r + 5z
\end{equation}
$$(c_exact_simple_soret_cylindrical)

For this problem, we choose:

$$
\begin{align}
    & T = 300 + 30r^2 + 50z^2 \\
    & Q^* = 4 \\
    & D = 2
\end{align}
$$

Injecting {eq}`c_exact_simple_soret_cylindrical` in {eq}`problem_simple_soret_cylindrical`, we obtain the expressions of $S$ and $c_0$:

$$
\begin{align}
    & S = - D\nabla \cdot \left(\frac{Q^* c_\mathrm{exact}}{k_B T^2} \ \nabla{T} \right) -12 D \\
    & c_0 = c_\mathrm{exact}
\end{align}
$$

We can then run a FESTIM model with these values and compare the numerical solution with $c_\mathrm{exact}$.

+++

## FESTIM code

```{code-cell} ipython3
import festim as F
import sympy as sp
import fenics as f
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Create and mark the mesh
nx = ny = 100
fenics_mesh = f.RectangleMesh(f.Point(1.0, 0), f.Point(2.0, 1.0), nx, ny)

volume_markers = f.MeshFunction("size_t", fenics_mesh, fenics_mesh.topology().dim())
volume_markers.set_all(1)

surface_markers = f.MeshFunction(
    "size_t", fenics_mesh, fenics_mesh.topology().dim() - 1
)

surface_markers.set_all(0)


class Boundary(f.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


boundary = Boundary()
boundary.mark(surface_markers, 1)

# Create the FESTIM model
my_model = F.Simulation()

my_model.mesh = F.Mesh(
    fenics_mesh, volume_markers=volume_markers, surface_markers=surface_markers, type="cylindrical"
)

# Variational formulation
r = F.x
z = F.y

exact_solution = 1 + 3 * r + 5 * z # exact solution

T = 300 + 30*r**2 + 50*z**2

D = 2
Q = 4

def grad_cylindrical(u):
    """Computes the gradient of a function u in cylindrical coordinates.

    Args:
        u (sympy.Expr): a sympy function

    Returns:
        sympy.Matrix: the gradient of u
    """

    return sp.Matrix([sp.diff(u, r), sp.diff(u, z)])


def div_cylindrical(u):
    """Computes the divergence of a vector field u in cylindrical coordinates.

    Args:
        u (sympy.Matrix): a sympy vector field

    Returns:
        sympy.Expr: the divergence of u
    """

    return 1/r * sp.diff(u[0] * r, r) + sp.diff(u[1], z)

diffusion_term = -D * grad_cylindrical(exact_solution)
soret_term = -D * (Q * exact_solution) / (F.k_B * T**2) * grad_cylindrical(T)

mms_source = div_cylindrical( diffusion_term + soret_term )

my_model.sources = [
    F.Source(
        mms_source,
        volume=1,
        field="solute",
    ),
]

my_model.boundary_conditions = [
    F.DirichletBC(surfaces=[1], value=exact_solution, field="solute"),
]

my_model.materials = F.Material(id=1, D_0=D, E_D=0, Q=Q)

my_model.T = F.Temperature(T)

my_model.settings = F.Settings(
    absolute_tolerance=1e-10, relative_tolerance=1e-10, transient=False, soret=True
)

my_model.initialise()
my_model.run()
```

## Comparison with exact solution

```{code-cell} ipython3
:tags: [hide-input]

c_exact = f.Expression(sp.printing.ccode(exact_solution), degree=4)
c_exact = f.project(c_exact, f.FunctionSpace(my_model.mesh.mesh, "CG", 1))

computed_solution = my_model.h_transport_problem.mobile.post_processing_solution
E = f.errornorm(computed_solution, c_exact, "L2")
print(f"L2 error: {E:.2e}")

# plot exact solution and computed solution
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plt.sca(axs[0])
plt.title("Exact solution")
plt.xlabel("r")
plt.ylabel("z")
CS1 = f.plot(c_exact, cmap="inferno")
plt.sca(axs[1])
plt.xlabel("r")
plt.title("Computed solution")
CS2 = f.plot(computed_solution, cmap="inferno")

plt.colorbar(CS2, ax=[axs[0], axs[1]], shrink=0.8)

axs[0].sharey(axs[1])
plt.setp(axs[1].get_yticklabels(), visible=False)

for CS in [CS1, CS2]:
    CS.set_edgecolor("face")


def compute_arc_length(xs, ys):
    """Computes the arc length of x,y points based
    on x and y arrays
    """
    points = np.vstack((xs, ys)).T
    distance = np.linalg.norm(points[1:] - points[:-1], axis=1)
    arc_length = np.insert(np.cumsum(distance), 0, [0.0])
    return arc_length


# define the profiles
profiles = [
    {"start": (1.0, 0.0), "end": (2.0, 1.0)},
    {"start": (1.2, 0.8), "end": (1.7, 0.2)},
    {"start": (1.2, 0.6), "end": (1.8, 0.8)},
]

# plot the profiles on the right subplot
for i, profile in enumerate(profiles):
    start_x, start_y = profile["start"]
    end_x, end_y = profile["end"]
    plt.sca(axs[1])
    (l,) = plt.plot([start_x, end_x], [start_y, end_y])

    plt.sca(axs[2])

    points_x_exact = np.linspace(start_x, end_x, num=30)
    points_y_exact = np.linspace(start_y, end_y, num=30)
    arc_length_exact = compute_arc_length(points_x_exact, points_y_exact)
    u_values = [c_exact(x, y) for x, y in zip(points_x_exact, points_y_exact)]

    points_x = np.linspace(start_x, end_x, num=100)
    points_y = np.linspace(start_y, end_y, num=100)
    arc_lengths = compute_arc_length(points_x, points_y)
    computed_values = [computed_solution(x, y) for x, y in zip(points_x, points_y)]

    (exact_line,) = plt.plot(
        arc_length_exact,
        u_values,
        color=l.get_color(),
        marker="o",
        linestyle="None",
        alpha=0.3,
    )
    (computed_line,) = plt.plot(arc_lengths, computed_values, color=l.get_color())

plt.sca(axs[2])
plt.xlabel("Arc length")
plt.ylabel("Solution")

legend_marker = mpl.lines.Line2D(
    [],
    [],
    color="black",
    marker=exact_line.get_marker(),
    linestyle="None",
    label="Exact",
)

legend_line = mpl.lines.Line2D([], [], color="black", label="Computed")
plt.legend(
    [legend_marker, legend_line], [legend_marker.get_label(), legend_line.get_label()]
)

plt.grid(alpha=0.3)
plt.gca().spines[["right", "top"]].set_visible(False)
plt.show()
```