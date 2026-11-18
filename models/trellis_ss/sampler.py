import numpy as np
import torch


class SSFlowEulerSampler:
    """Euler flow sampler with classifier-free guidance limited to a t-interval.

    Same update rule as TRELLIS' FlowEulerGuidanceIntervalSampler: guidance is
    applied only while t is inside ``cfg_interval``; elsewhere the conditional
    prediction is used as is.
    """

    def _pred_v(self, model, x_t, t, cond):
        t_vec = torch.full(
            (x_t.shape[0],), 1000.0 * t, device=x_t.device, dtype=torch.float32
        )
        return model(x_t, t_vec, cond)

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps=25,
        cfg_strength=5.0,
        cfg_interval=(0.5, 1.0),
        rescale_t=3.0,
    ):
        x = noise
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        for i in range(steps):
            t, t_prev = float(t_seq[i]), float(t_seq[i + 1])
            v = self._pred_v(model, x, t, cond)
            if cfg_interval[0] <= t <= cfg_interval[1]:
                v_uncond = self._pred_v(model, x, t, neg_cond)
                v = (1 + cfg_strength) * v - cfg_strength * v_uncond
            x = x - (t - t_prev) * v
        return x
