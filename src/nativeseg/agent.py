"""Multiround closed-loop control: multiscale error gate, freeform ROI shift, and verification."""

import math
import random
import torch
from torch.nn import functional as F

from .model import (
    CANVAS,
    CELLS,
    FULL,
    letterbox_params,
    native_grid,
    overlay,
    render,
    to_native,
    to_native_feathered,
)


def cell_box(c):
    if isinstance(c, tuple):
        return c
    return FULL if c == 0 else CELLS[c - 1]


def propose_free_roi(request, errors, best_idx):
    """Compute a continuous freeform bounding box (v0, u0, s) on the 512x512 canvas by shifting
    the base cell center toward contiguous same-scale high-error neighbors."""
    base_v0, base_u0, base_s = cell_box(best_idx)
    base_vc, base_uc = base_v0 + 0.5 * base_s, base_u0 + 0.5 * base_s
    if best_idx <= 4:
        e2 = errors[1:5].float().clamp(min=0).reshape(2, 2)
        centers = torch.tensor([128.0, 384.0], device=errors.device)
        gy, gx = torch.meshgrid(centers, centers, indexing="ij")
        peak = float(e2.max())
        w = torch.where(e2 >= 0.45 * peak, e2**2, torch.zeros_like(e2))
    else:
        e4 = errors[5:21].float().clamp(min=0).reshape(4, 4)
        centers = torch.tensor([64.0, 192.0, 320.0, 448.0], device=errors.device)
        gy, gx = torch.meshgrid(centers, centers, indexing="ij")
        dist_ok = ((gy - base_vc).abs() <= 128.0) & ((gx - base_uc).abs() <= 128.0)
        peak = float(errors[best_idx])
        w = torch.where(dist_ok & (e4 >= 0.40 * peak), e4**2, torch.zeros_like(e4))
    w_sum = float(w.sum())
    if w_sum > 1e-4:
        vc = 0.65 * base_vc + 0.35 * float((w * gy).sum() / w_sum)
        uc = 0.65 * base_uc + 0.35 * float((w * gx).sum() / w_sum)
    else:
        vc, uc = base_vc, base_uc
    s = float(base_s)
    v0 = min(max(round((vc - 0.5 * s) / 4.0) * 4.0, 0.0), float(CANVAS - s))
    u0 = min(max(round((uc - 0.5 * s) / 4.0) * 4.0, 0.0), float(CANVAS - s))
    return (v0, u0, s)


_INSIDE_CACHE = {}


class Request:
    """One (slice, organ) request with its native image, geometry, and current confidence/mask state."""

    def __init__(self, image, text, device, blank_state=False, no_overlay=False):
        t_img = torch.as_tensor(image, device=device).float()
        self.image = t_img[None] if t_img.ndim == 2 else t_img       # (1 or 5, H, W)
        h, w = self.image.shape[-2:]
        self.params = letterbox_params(h, w)
        self.text = text
        self.mask = torch.zeros(h, w, dtype=torch.bool, device=device)
        self.gray = render(self.image, self.params, FULL)            # (1 or 5, 512, 512)
        self.center_gray = (
            self.gray
            if self.gray.shape[0] == 1
            else self.gray[self.gray.shape[0] // 2 : self.gray.shape[0] // 2 + 1]
        )
        self.scale2 = (self.params["resized"][0] / h) ** 2
        key = (h, w, str(device))
        if key not in _INSIDE_CACHE:
            _INSIDE_CACHE[key] = [
                native_grid(self.params, c)[1].to(device) for c in CELLS
            ]
        self.inside = _INSIDE_CACHE[key]
        self.visited, self.done, self.trace = [], False, []
        self.blank_state = blank_state
        self.no_overlay = no_overlay
        self.logit_mem = None
        self.initial_errors = None

    def shown_mask(self):
        return (
            torch.zeros_like(self.mask)
            if (self.blank_state or self.no_overlay)
            else self.mask
        )

    def shown_state_float(self):
        if self.blank_state:
            return torch.zeros_like(self.mask, dtype=torch.float32)
        if self.logit_mem is not None:
            prob = torch.sigmoid(self.logit_mem)
            return prob * self.mask.float() + 0.15 * prob * (~self.mask).float()
        return self.mask.float()

    def full_overlay(self):
        state = render(self.shown_mask().float()[None], self.params, FULL)
        return overlay(self.center_gray, state)

    def cell_views(self, c):
        cell = cell_box(c)
        gray = render(self.image, self.params, cell)
        center_c = (
            gray
            if gray.shape[0] == 1
            else gray[gray.shape[0] // 2 : gray.shape[0] // 2 + 1]
        )
        vis_mask = render(self.shown_mask().float()[None], self.params, cell)
        prior_state = render(self.shown_state_float()[None], self.params, cell)
        return gray, overlay(center_c, vis_mask), prior_state

    def cell_errors(self, truth):
        wrong = self.mask ^ truth
        per_cell = torch.stack([(wrong & m).sum() for m in self.inside]).float()
        return torch.cat((wrong.sum().float()[None], per_cell))


def decide_targets(requests, truths):
    return torch.stack(
        [torch.log1p(r.cell_errors(t) * r.scale2) for r, t in zip(requests, truths)]
    )


def apply_zoom(request, c, cell_logits, feathered=False, logit_memory=False):
    box = cell_box(c)
    if feathered or logit_memory:
        native, inside, weight = to_native_feathered(
            cell_logits.float(), request.params, box, taper=0.06 if feathered else 0.0
        )
        if logit_memory and request.logit_mem is not None:
            request.logit_mem = torch.where(
                inside,
                (1.0 - weight) * request.logit_mem + weight * native,
                request.logit_mem,
            )
            request.mask = request.logit_mem > 0
        else:
            cur_logits = 2.0 * request.mask.float() - 1.0
            blended = (1.0 - weight) * cur_logits + weight * torch.tanh(native)
            request.mask = torch.where(inside, blended > 0, request.mask)
    else:
        native, inside = to_native(cell_logits.float(), request.params, box)
        request.mask = torch.where(inside, native > 0, request.mask)


def entropy_scores(logits_native, request):
    p = logits_native.sigmoid().clamp(1e-6, 1 - 1e-6)
    h = -(p * p.log() + (1 - p) * (1 - p).log())
    return torch.stack([(h * m).sum() for m in request.inside])


@torch.no_grad()
def run_agent(
    model,
    requests,
    policy,
    max_steps,
    tau=0.05,
    min_area=800,
    seed=0,
    presence_threshold=0.5,
    initial_masks=None,
    verify=False,
    absent_exit=None,
    record=False,
    truths=None,
    revise="zoom",
    free_roi=False,
    logit_memory=False,
):
    """Run a batch of organ-slice requests through NativeSeg's multiround loop."""
    rng = random.Random(seed)
    device = requests[0].image.device
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    need_initial = (
        initial_masks is None
        or policy == "uncertainty"
        or absent_exit is not None
        or logit_memory
    )
    if need_initial:
        with autocast:
            logits, presence = model.initial(
                [r.text for r in requests], torch.stack([r.gray for r in requests])
            )
    entropy = []
    for i, r in enumerate(requests):
        if need_initial:
            native, _ = to_native(logits[i].float(), r.params, FULL)
            keep = float(presence[i].float().sigmoid()) >= presence_threshold
            r.initial_raw = native > 0
            r.presence = float(presence[i].float().sigmoid())
            entropy.append(
                entropy_scores(native, r)
                if keep
                else torch.zeros(len(CELLS), device=native.device)
            )
            r.mask = (native > 0) & keep
            if logit_memory:
                r.logit_mem = torch.where(
                    r.mask, native.clamp(min=0.05), native.clamp(max=-0.05)
                )
        if initial_masks is not None:
            r.mask = torch.as_tensor(initial_masks[i], device=device).bool()
            r.initial_raw = r.mask.clone()
            if not need_initial:
                r.presence = 1.0
            if logit_memory:
                r.logit_mem = 2.5 * (2.0 * r.mask.float() - 1.0)
        r.initial_mask = r.mask.clone()
        r.pending = None
        r.reverted = []
        r.decide_calls = 0
        r.states = []
        if absent_exit is not None and not bool(r.mask.any()) and r.presence < absent_exit:
            r.done = True

    def assess(ids):
        for i in ids:
            requests[i].decide_calls += 1
        with autocast:
            pred = model.decide_logits(
                [requests[i].text for i in ids],
                torch.stack([requests[i].full_overlay() for i in ids]),
            ).float()
        return torch.expm1(pred.clamp(min=0))

    def check(i, errors):
        r = requests[i]
        if r.initial_errors is None:
            r.initial_errors = errors.clone()
        if not (verify and r.pending):
            r.pending = None
            return errors
        c, before, old_mask, old_errors, old_mem = r.pending
        r.pending = None
        global_cap = float(old_errors[0])
        if logit_memory and r.initial_errors is not None:
            global_cap = min(global_cap, float(r.initial_errors[0]))
        if float(errors[c]) >= before or float(errors[0]) > global_cap:
            r.mask = old_mask
            r.logit_mem = old_mem
            r.reverted.append(c)
            return old_errors
        return errors

    if record and policy != "learned":
        for r in requests:
            r.states.append(
                (0, r.mask.clone(), float("nan"), float(r.mask.sum()) * r.scale2)
            )
    for step in range(max_steps):
        active = [i for i, r in enumerate(requests) if not r.done]
        if not active:
            break
        choice = {}
        choice_box = {}
        if policy == "learned":
            errors_all = assess(active)
            for j, i in enumerate(active):
                r = requests[i]
                errors = check(i, errors_all[j])
                cells = errors[1:].clone()
                for c in r.visited:
                    if isinstance(c, int) and c > 0:
                        cells[c - 1] = -1
                best = int(cells.argmax())
                limit = tau * max(float(r.mask.sum()) * r.scale2, min_area)
                if record:
                    r.states.append(
                        (
                            step,
                            r.mask.clone(),
                            float(cells[best]),
                            float(r.mask.sum()) * r.scale2,
                        )
                    )
                r.trace.append(
                    {
                        "step": step,
                        "predicted_total": float(errors[0]),
                        "predicted_best": float(cells[best]),
                        "limit": limit,
                    }
                )
                if float(cells[best]) < limit:
                    r.done = True
                else:
                    choice[i] = best + 1
                    choice_box[i] = (
                        propose_free_roi(r, errors, best + 1)
                        if free_roi
                        else (best + 1)
                    )
                    if verify:
                        r.pending = (
                            best + 1,
                            float(errors[best + 1]),
                            r.mask.clone(),
                            errors,
                            None if r.logit_mem is None else r.logit_mem.clone(),
                        )
        else:
            for i in active:
                r = requests[i]
                if policy == "uncertainty":
                    scores = entropy[i].clone()
                    for c in r.visited:
                        if isinstance(c, int) and c > 0:
                            scores[c - 1] = -1
                    choice[i] = int(scores.argmax()) + 1
                elif policy == "random":
                    choice[i] = rng.choice(
                        [c for c in range(1, len(CELLS) + 1) if c not in r.visited]
                    )
                elif policy == "all2x2":
                    choice[i] = step + 1 if step < 4 else None
                elif policy == "all4x4":
                    choice[i] = step + 5 if step < 16 else None
                elif policy == "reread":
                    choice[i] = 0
                elif policy == "oracle":
                    errors = r.cell_errors(truths[i])[5:].clone()
                    for c in r.visited:
                        if isinstance(c, int) and c > 4:
                            errors[c - 5] = -1
                    choice[i] = (
                        int(errors.argmax()) + 5 if float(errors.max()) > 0 else None
                    )
                else:
                    raise ValueError(policy)
                if choice[i] is None:
                    r.done = True
                    del choice[i]
                else:
                    choice_box[i] = choice[i]
        if not choice:
            continue
        ids = list(choice)
        views = [requests[i].cell_views(choice_box[i]) for i in ids]
        with autocast:
            if revise == "initial":
                cell_logits, _ = model.initial(
                    [requests[i].text for i in ids], torch.stack([v[0] for v in views])
                )
            else:
                cell_logits = model.zoom(
                    [requests[i].text for i in ids],
                    torch.stack([requests[i].full_overlay() for i in ids]),
                    torch.stack([v[0] for v in views]),
                    torch.stack([v[1] for v in views]),
                    torch.stack([v[2] for v in views]),
                )
        for j, i in enumerate(ids):
            apply_zoom(
                requests[i],
                choice_box[i],
                cell_logits[j],
                feathered=free_roi,
                logit_memory=logit_memory,
            )
            requests[i].visited.append(choice[i])
            if record and policy != "learned":
                r = requests[i]
                r.states.append(
                    (
                        step + 1,
                        r.mask.clone(),
                        float("nan"),
                        float(r.mask.sum()) * r.scale2,
                    )
                )
    pending = [i for i, r in enumerate(requests) if r.pending]
    if pending:
        final = assess(pending)
        for j, i in enumerate(pending):
            check(i, final[j])
            requests[i].final_assessments = 1
    if record and policy == "learned":
        for r in requests:
            if len(r.visited) == max_steps:
                r.states.append(
                    (
                        max_steps,
                        r.mask.clone(),
                        float("inf"),
                        float(r.mask.sum()) * r.scale2,
                    )
                )
    for r in requests:
        r.done = True
    return requests


def _ellipse(h, w, cy, cx, ry, rx, device):
    yy = torch.arange(h, device=device).float()[:, None]
    xx = torch.arange(w, device=device).float()[None, :]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def perturb(truth, rng):
    """Synthetic specialist-like errors on a mask for Stage B and LRN† curriculum training."""
    h, w = truth.shape
    device = truth.device
    m = truth.clone()
    if not m.any():
        if rng.random() < 0.5:
            cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
            m |= _ellipse(h, w, cy, cx, rng.uniform(2, 12), rng.uniform(2, 12), device)
        return m
    if rng.random() < 0.05:
        return torch.zeros_like(m)
    ys, xs = torch.nonzero(m, as_tuple=True)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    for op in rng.sample(["dilate", "erode", "cut", "blob", "shift"], k=rng.choice((1, 1, 2))):
        f = m.float()[None, None]
        k = rng.randint(1, 3)
        if op == "dilate":
            m = F.max_pool2d(f, 2 * k + 1, 1, k)[0, 0] > 0
        elif op == "erode":
            m = -F.max_pool2d(-f, 2 * k + 1, 1, k)[0, 0] > 0
        elif op == "cut":
            angle = rng.uniform(0, 2 * math.pi)
            cy, cx = rng.uniform(y0, y1 + 1), rng.uniform(x0, x1 + 1)
            yy = torch.arange(h, device=device).float()[:, None] - cy
            xx = torch.arange(w, device=device).float()[None, :] - cx
            m = m & ((yy * math.sin(angle) + xx * math.cos(angle)) < 0)
        elif op == "blob":
            cy = rng.uniform(y0 - 15, y1 + 15)
            cx = rng.uniform(x0 - 15, x1 + 15)
            m = m | _ellipse(
                h, w, cy, cx, rng.uniform(2, 10), rng.uniform(2, 10), device
            )
        elif op == "shift":
            m = torch.roll(m, (rng.randint(-4, 4), rng.randint(-4, 4)), (0, 1))
    return m
