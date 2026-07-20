import numpy as np
import torch


class VertFlowEulerCfgSampler:
    def _pred_v(self, model, x_t, t, cond, **kwargs):
        t_vec = torch.full(
            (x_t.shape[0],), 1000.0 * t, device=x_t.device, dtype=torch.float32
        )
        return model(x_t, t_vec, cond, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps=12,
        cfg_strength=3.0,
        rescale_t=1.0,
        **kwargs,
    ):
        x = noise
        t_seq = np.linspace(1.0, 0.0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        for i in range(steps):
            t, t_prev = float(t_seq[i]), float(t_seq[i + 1])
            v_cond = self._pred_v(model, x, t, cond, **kwargs)
            v_uncond = self._pred_v(model, x, t, neg_cond, **kwargs)
            v = (1 + cfg_strength) * v_cond - cfg_strength * v_uncond
            x = x - (t - t_prev) * v
        return x


class TopoFlowEulerSampler:
    def _pred_v(self, model, x_t, t, verts, mask, cond, cond_mask):
        t_vec = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.float32)
        return model(x_t, t_vec, verts=verts, mask=mask, cond=cond, cond_mask=cond_mask)

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        verts,
        mask,
        cond=None,
        cond_mask=None,
        steps=50,
    ):
        x = noise
        t_seq = np.linspace(0.0, 1.0, steps + 1)
        for i in range(steps):
            t, t_next = float(t_seq[i]), float(t_seq[i + 1])
            v = self._pred_v(model, x, t, verts, mask, cond, cond_mask)
            x = x + (t_next - t) * v
        return x
