import numpy as np
p = np.load('local_storage/gaits/crawl_gait.npy')
JOINTS = ('hip_pitch', 'knee', 'hip_roll', 'ankle')
for k, name in enumerate(JOINTS):
    off, amp, ph = p[3*k : 3*k+3]
    print(f"{name:12s}: off={off:+6.2f}, amp={amp:+6.2f}, ph={ph:+6.2f}")
print(f"freq = {p[-3]:.2f}, neck = {p[-2]:.2f}, head = {p[-1]:.2f}")
