# Suspension bridge ? walking across a loose, soft rope bridge

**Date:** 2026-09-12.  
**Script:** `src/microduck_lab/tasks/bridge/bridge.py`.  
**Task module:** `microduck_lab.tasks.bridge`.  
**Policy:** Pretrained `alpha_walking.onnx`, unchanged. Guided by a closed-loop PD heading and lateral tracking steering controller.  
**Inspiration & Net Geometry:** Adapted directly from the knotted rope obstacle in `src/microduck_lab/tasks/crawl/net.py`, enhanced with soft contact compliance and loose catenary droop.

---

## 1. Executive Summary

The Microduck traverses an elevated 2.0 m rope suspension bridge featuring deep catenary sag and soft compliant knotted rope mesh.

| Configuration | Sag | Net Physics | Distance | Walk Time | Speed | Max Lateral Dev | Status |
|---|---:|:---:|---:|---:|---:|---:|:---:|
| Shallow sag | 20 mm | **Soft** | 1.70 m | 15.3 s | 111 mm/s | 22.5 mm | **PASS** |
| Moderate sag | 30 mm | **Soft** | 1.69 m | 16.3 s | 104 mm/s | 15.8 mm | **PASS** |
| Loose bridge | 40 mm | **Soft** | 1.68 m | 17.8 s | 94 mm/s | 16.9 mm | **PASS** |
| Deep loose sag | 50 mm | **Soft** | 1.64 m | 16.9 s | 97 mm/s | 18.7 mm | **PASS** |

---

## 2. Soft Net Physical Modeling

Instead of rigid contact, all rope capsules and knot spheres are equipped with **MuJoCo soft contact compliance**:
- `solref="0.04 1.0"`: Soft contact restitution time constant with critical damping ($\zeta = 1.0$).
- `solimp="0.6 0.95 0.015 0.5 2"`: Progressive impedance transition across a $15\text{ mm}$ compression zone.
- `margin="0.008"`: An $8\text{ mm}$ soft cushion envelope around every rope segment and knot where the duck's feet sink into the net mesh under body weight.
- `friction="1.3 0.01 0.0002"`: High-traction cordage friction preventing toe slips.

---

## 3. Suspension Bridge Geometry

The bridge is procedurally generated in MuJoCo (`bridge_xml()`):
1. **Approach & Exit Landings**: Solid timber platforms at $z = 0.18$ m, giving stable footing at the start and finish.
2. **Pylons / Towers**: Four vertical timber pillars ($r = 16$ mm) at the banks ($x = -0.25$ m and $x = 1.75$ m), rising to $z = 0.38$ m, topped with protective caps and crossbars.
3. **Main Cables**: Deep catenary parabolic cables along left and right flanks ($y = \pm 0.22$ m).
4. **Vertical Suspenders (Hangers)**: Evenly spaced rope capsules connecting the upper suspension cables to the deck stringers.
5. **Knotted Soft Walking Deck**: A 25 mm grid of cross and longitudinal ropes with knot spheres at intersections and 3D hammock sag:
   $$z(x, y) = H - \text{SAG} \cdot \sin(\pi u) \cdot \sin(\pi v)$$
   creating a deep, loose central trough.

---

## 4. Usage & Reproduction

```bash
# Run standard soft loose bridge (50 mm central sag)
.venv/bin/python -m microduck_lab.tasks.bridge.bridge --sag 0.05 --speed 0.28

# Parameter sweep across multiple sag depths with soft net
.venv/bin/python -m microduck_lab.tasks.bridge.bridge --speed 0.28 --sweep 0.02 0.03 0.04 0.05

# Automated test suite
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest microduck_lab.tests.test_bridge -v

# Cluster batch submission for video rendering and report generation
sbatch -M cluster src/microduck_lab/tasks/bridge/bridge.sbatch
```
