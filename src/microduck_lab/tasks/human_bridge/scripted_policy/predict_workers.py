"""Separate CPU workers for hypothetical predictions. They never render."""
from types import SimpleNamespace
from unittest.mock import patch

import onnxruntime as ort

from microduck_lab.tasks.human_bridge.scripted_policy.world import World, Brain
from microduck_lab.tasks.human_bridge.scripted_policy.scene import LAYOUT as L
from microduck_lab.tasks.human_bridge.scripted_policy.brother import HoldStraight

ENGINE = None


def initialize(settings):
    global ENGINE
    from microduck_lab.tasks.human_bridge.scripted_policy.preview import Preview
    engine = Preview.__new__(Preview)
    engine.cfg = settings
    engine.world = World(L.design())
    # Each worker uses one CPU. The saved policies stay unchanged.
    session = ort.InferenceSession
    def single_thread(*args, **kwargs):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        kwargs['sess_options'] = options
        return session(*args, **kwargs)
    with patch.object(ort, 'InferenceSession', single_thread):
        engine.brain = Brain(engine.world.ducks['she'])
        engine.brain.pol.walking_session = single_thread(settings.policy,
                                                        providers=['CPUExecutionProvider'])
    engine.world.start()
    engine.hold = HoldStraight(engine.world.ducks['he'])
    ENGINE = engine


def evaluate(payload):
    snapshot, policy_state, extra, table, command = payload
    engine = ENGINE
    pol = SimpleNamespace(**policy_state)
    pol.walking_session = engine.brain.pol.walking_session
    pol.standing_session = engine.brain.pol.standing_session
    pol.ort_session = pol.walking_session if pol.current_policy == 'walking' else pol.standing_session
    engine.live = SimpleNamespace(brain=SimpleNamespace(pol=pol),
                                  brother_hold=SimpleNamespace(extra=extra), table=table)
    engine.predictions = []
    score = engine.score(snapshot, command)
    return score, engine.predictions[0]
