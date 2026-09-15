import time
import mujoco

xml = """
<mujoco>
  <worldbody>
    <body name="r0" pos="0 0 0">
      <joint type="slide" axis="0 0 1"/>
      <geom type="box" size="0.04 0.2 0.01" contype="1" conaffinity="0"/>
    </body>
    <body name="r1" pos="0.08 0 0">
      <joint type="slide" axis="0 0 1"/>
      <geom type="box" size="0.04 0.2 0.01" contype="1" conaffinity="0"/>
    </body>
    <body name="foot" pos="0 0 0.05">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print("Initial ncon (foot in air):", d.ncon) # should be 0!

# Drop foot onto r0
d.qpos[7+2] = 0.015 # foot touching r0
mujoco.mj_forward(m, d)
print("Foot touching r0 ncon:", d.ncon) # should be > 0!
for i in range(d.ncon):
    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom1)
    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2)
    print(f"Contact {i}: {g1} <-> {g2}")
