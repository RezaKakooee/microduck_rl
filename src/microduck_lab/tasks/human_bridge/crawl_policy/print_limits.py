from microduck_lab.tasks.human_bridge.crawl_policy.story import Story
from microduck_lab.tasks.human_bridge.crawl_policy.world import SERVOS
s = Story(verbose=False)
for i, name in enumerate(SERVOS):
    lo = s.she.lo[i]
    hi = s.she.hi[i]
    print(f"{name:16s}: [{lo:+6.2f}, {hi:+6.2f}]")
