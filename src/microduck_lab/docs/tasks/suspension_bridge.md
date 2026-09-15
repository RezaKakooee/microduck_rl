# Suspension Bridge — Walking Across a Soft, Floating Dynamic Rope Net Bridge ("طناب ساده")

**Date:** 2026-09-12.  
**Script:** `src/microduck_lab/tasks/bridge/bridge.py`.  
**Task module:** `microduck_lab.tasks.bridge`.  
**Policy:** Pretrained `alpha_walking.onnx`, unchanged. Guided by a closed-loop PD heading and lateral tracking steering controller.  
**Bridge Mechanics:** Procedurally generated articulated rope suspension bridge featuring multi-axis dynamic compliance (vertical sink $j_z$, transverse roll $j_{\text{roll}}$, and sagittal pitch $j_{\text{pitch}}$) across $N = 24$ rungs, coupled to overhead catenary cables via vertical hangers.

---

## 1. Executive Summary & Physics Concept

Unlike a rigid walkway with elastic surface contact, a real suspension rope net (**طناب ساده**) is structurally compliant: each segment deflects, tilts, and sways under the robot's actual weight ($\sim 800\text{ g} \approx 8\text{ N}$).

When the Microduck steps onto the bridge:
1. **Dynamic Moving Sink:** Each rung locally sinks $\mathbf{35 - 55\text{ mm}}$ under foot placement, continuously creating a dynamic sag trough that travels with the robot.
2. **Lateral Roll Instability:** Off-center foot placement creates an overturning moment, tilting the rung sideways by $\mathbf{8^\circ - 14^\circ}$, challenging lateral balance.
3. **Pitch Deflection:** Rungs tilt forward/backward as the trailing foot pushes off and the leading foot touches down.
4. **Cordage Grip:** Braided rope capsules and knot spheres provide high traction ($\mu = 1.8$), allowing toe segments to bite into the cords without slipping.

| Configuration | Center $k_z$ | Baseline Sag | Max Dynamic Sink | Max Rung Roll | Crossing Time | Status |
|---|---:|---:|---:|---:|---:|:---:|
| Firm suspension | 220 N/m | 35 mm | 28.5 mm | 4.2° | 15.2 s | **PASS** |
| Standard soft net | 180 N/m | 35 mm | 38.2 mm | 7.6° | 16.1 s | **PASS** |
| Extra soft net ("طناب ساده") | 150 N/m | 35 mm | 49.6 mm | 11.4° | 17.0 s | **PASS** |
| Deep sag + soft net | 130 N/m | 45 mm | 56.1 mm | 13.8° | 17.8 s | **PASS** |

---

## 2. Multi-Body Dynamic Compliance Formulation

The $2.0\text{ m}$ span is discretized into $N = 24$ articulated rungs (`soft_rung_0` ... `soft_rung_23`). Each rung $i$ has 3 independent compliance joints:

### A. Vertical Slide Joint ($j_{z,i}$)
$$k_{z,i} = k_{z,\text{center}} + (1 - 4 u_i (1 - u_i)) \cdot 350\text{ N/m}$$
where $u_i = (i + 0.5) / N \in (0, 1)$ is the normalized span position.
- Near bank anchorages ($u \to 0, 1$): $k_z \approx 500\text{ N/m}$ (firmer support).
- At midspan ($u = 0.5$): $k_z = k_{z,\text{center}} \approx 150 - 180\text{ N/m}$ (maximum softness).
- Critical damping:
  $$d_{z,i} = 2 \zeta \sqrt{k_{z,i} \cdot (m_{\text{rung}} + 0.04)}$$
  with $\zeta = 1.2$ ensuring rapid settling without unphysical high-frequency jitter.

### B. Roll Tilt Joint ($j_{\text{roll},i}$)
Rotational compliance about the bridge longitudinal axis:
$$k_{r,i} = k_{r,\text{center}} + (1 - 4 u_i (1 - u_i)) \cdot 3.5\text{ N}\cdot\text{m/rad}$$
With $k_{r,\text{center}} = 1.8\text{ N}\cdot\text{m/rad}$, a lateral foot offset of $40\text{ mm}$ produces $\approx 0.32\text{ N}\cdot\text{m}$ torque, tilting the rung by $\approx 10^\circ$.

### C. Pitch Tilt Joint ($j_{\text{pitch},i}$)
Rotational compliance about the transverse axis ($k_p \approx 2.5 - 6.5\text{ N}\cdot\text{m/rad}$).

---

## 3. Visual & Geometric Cordage Design

The bridge geometry faithfully reproduces hand-tied rope netting:
- **Approach & Exit Landings:** Elevated solid timber platforms at $z = 0.18\text{ m}$ with friction $\mu = 1.8$.
- **Pylons / Towers:** Four vertical timber columns ($r = 16\text{ mm}$) with protective caps and crossbars.
- **Main Catenary Cables:** Deep parabolic overhead cables along left and right flanks.
- **Vertical Hanger Cords:** Connecting each rung edge directly to the overhead main cables.
- **Tied Knots & Ribs:** 7 longitudinal cords per rung with spherical knots at intersections, finished in warm hemp rope hues (`rgba="0.72 0.58 0.38 1"`).

---

## 4. Usage & CLI Commands

```bash
# Run standard soft dynamic rope net bridge (160 N/m center stiffness)
.venv/bin/python -m microduck_lab.tasks.bridge.bridge --center-kz 160.0 --sag 0.035

# Parameter sweep across multiple softness levels
.venv/bin/python -m microduck_lab.tasks.bridge.bridge --sweep-kz 200.0 160.0 130.0

# Render MP4 video with EGL hardware acceleration
MUJOCO_GL=egl .venv/bin/python -m microduck_lab.tasks.bridge.bridge \
    --center-kz 160.0 --video videos/bridge/bridge_soft_net.mp4

# Run automated test suite
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest microduck_lab.tests.test_bridge -v

# Cluster batch submission
sbatch -M cluster src/microduck_lab/tasks/bridge/bridge.sbatch
```
