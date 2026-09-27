"""Plan a short sequence of corrections, and execute its first part."""
import numpy as np

from microduck_lab.tasks.human_bridge.scripted_policy.preview import Preview


class SequencePreview(Preview):
    def __init__(self, supervisor):
        super().__init__(supervisor)
        self.best = np.tile(self.best, (self.cfg.preview_knots, 1))
        self.plan_time = self.live.duck.world.t
        self.sequences = []

    def current_parameters(self):
        return self.parameters_at(self.best, self.live.duck.world.t-self.plan_time,
                                  self.cfg.preview_horizon)

    def command(self, nominal):
        time = self.live.duck.world.t
        tick = round(time/.02)
        if tick < self.next_plan:
            return self.body_command(self.live.duck, self.current_parameters())
        self.next_plan = tick + max(1, round(self.cfg.preview_interval/.02))
        knots = np.linspace(0., self.cfg.preview_horizon, self.cfg.preview_knots)
        mean = np.array([self.parameters_at(self.best, t+time-self.plan_time,
                                           self.cfg.preview_horizon) for t in knots])
        dim = mean.shape[1]
        low = np.array([-.12, -.2, -.8, -.5, -.3, -.5, 0., 0.][:dim])
        high = np.array([.3, .2, .8, .5, .3, .5, self.cfg.preview_lift_cap, self.cfg.preview_lift_cap][:dim])
        if not self.cfg.preview_feedback:
            low[:2], high[:2] = [-.2, -.12], [.2, .3]
        scale = np.array([.06, .05, .25, .15, .1, .15, .2, .2][:dim])
        std = np.tile(scale, (len(knots), 1))
        base = np.zeros_like(mean)
        base[:, :3] = [self.cfg.speed, 0., 0.] if self.cfg.preview_feedback else nominal
        snapshot = self.snapshot()
        winner, prediction, winning_score = mean.copy(), None, -np.inf
        for _ in range(self.cfg.preview_iterations):
            samples = mean + self.rng.normal(size=(self.cfg.preview_samples, *mean.shape))*std
            # Correlate adjacent corrections. Avoid alternating large targets.
            if len(knots) > 2:
                samples[:, 1:-1] = (.25*samples[:, :-2] + .5*samples[:, 1:-1]
                                      + .25*samples[:, 2:])
            seeds = [mean, base, np.zeros_like(mean), winner]
            if self.cfg.preview_independent_lift:
                for foot in (6, 7):
                    for profile in (np.linspace(self.cfg.preview_lift_cap, 0., len(knots)),
                                    self.cfg.preview_lift_cap*np.sin(np.linspace(0., np.pi, len(knots)))):
                        step = base.copy()
                        step[:, foot] = profile
                        seeds.append(step)
            samples = np.concatenate((samples, seeds))
            samples = np.clip(samples, low, high)
            scores = np.asarray(self.evaluate(snapshot, samples))
            order = np.argsort(scores)
            best = order[-1]
            if scores[best] > winning_score:
                winner = samples[best].copy()
                prediction = self.predictions[best].copy()
                winning_score = float(scores[best])
            elite = samples[order[-max(4, self.cfg.preview_samples//6):]]
            mean = .3*mean + .7*elite.mean(axis=0)
            std = np.maximum(.2*scale, .3*std + .7*elite.std(axis=0))
        self.best, self.plan_time = winner, time
        self.expected = (time+.02, prediction)
        self.scores.append([time, winning_score, *winner[0]])
        self.sequences.append(winner.copy())
        return self.body_command(self.live.duck, self.current_parameters())
