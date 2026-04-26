"""
Unit test for teamwork.attn._masked_attention_impl.

Validates the dense (B, T) flex_attention path (with scatter/gather, presence
masking, attn_keep gating, and rotary embedding) against a per-batch reference
implementation that gathers present components and runs vanilla SDPA.

Run from the repo root:

    uv run python scripts/test_attn.py
"""

import sys
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb

from teamwork.attn import _masked_attention_impl


def reference_impl(
    img_q, img_k, img_v,
    txt_q, txt_k, txt_v,
    image_rotary_emb,
    batch_indices,
    teammate_indices,
    present,
    attn_keep,
):
    B = present.shape[0]
    Tc, L_img = img_q.shape[0], img_q.shape[1]
    L_text = txt_q.shape[1]

    img_out = torch.zeros_like(img_q)
    txt_out = torch.zeros_like(txt_q)

    bi = batch_indices.tolist()
    ti = teammate_indices.tolist()

    for b in range(B):
        comp_for_t: dict[int, int] = {}
        for c in range(Tc):
            if bi[c] == b:
                comp_for_t[ti[c]] = c
        present_teammates = sorted(comp_for_t.keys())
        if not present_teammates:
            continue

        q_parts = [txt_q[b]]
        k_parts = [txt_k[b]]
        v_parts = [txt_v[b]]
        for t in present_teammates:
            c = comp_for_t[t]
            q_parts.append(img_q[c])
            k_parts.append(img_k[c])
            v_parts.append(img_v[c])
        q = torch.cat(q_parts, dim=0)
        k = torch.cat(k_parts, dim=0)
        v = torch.cat(v_parts, dim=0)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos_parts = [cos[:L_text]]
            sin_parts = [sin[:L_text]]
            for t in present_teammates:
                start = L_text + t * L_img
                cos_parts.append(cos[start : start + L_img])
                sin_parts.append(sin[start : start + L_img])
            ref_cos = torch.cat(cos_parts, dim=0)
            ref_sin = torch.cat(sin_parts, dim=0)
            q = apply_rotary_emb(q.unsqueeze(0), (ref_cos, ref_sin), sequence_dim=1)[0]
            k = apply_rotary_emb(k.unsqueeze(0), (ref_cos, ref_sin), sequence_dim=1)[0]

        n = len(present_teammates)
        S = L_text + n * L_img
        mask = torch.ones(S, S, dtype=torch.bool, device=q.device)
        if attn_keep is not None:
            for i, t_i in enumerate(present_teammates):
                for j, t_j in enumerate(present_teammates):
                    if not bool(attn_keep[t_i, t_j].item()):
                        i_start = L_text + i * L_img
                        j_start = L_text + j * L_img
                        mask[
                            i_start : i_start + L_img,
                            j_start : j_start + L_img,
                        ] = False

        q_bhsf = q.transpose(0, 1).unsqueeze(0)
        k_bhsf = k.transpose(0, 1).unsqueeze(0)
        v_bhsf = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q_bhsf, k_bhsf, v_bhsf,
            attn_mask=mask.unsqueeze(0).unsqueeze(0),
        )
        out = out.squeeze(0).transpose(0, 1)

        txt_out[b] = out[:L_text]
        for i, t in enumerate(present_teammates):
            c = comp_for_t[t]
            img_out[c] = out[L_text + i * L_img : L_text + (i + 1) * L_img]

    return img_out, txt_out


def run_case(name, *, with_rotary, with_attn_keep, jagged):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float32

    B, T = 2, 3
    L_text, L_img = 128, 128
    H, Fd = 4, 64

    if jagged:
        # batch 0: teammates {0, 1}; batch 1: teammates {0, 1, 2}
        component_specs = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    else:
        # both batches have all teammates
        component_specs = [(b, t) for b in range(B) for t in range(T)]
    Tc = len(component_specs)

    batch_indices = torch.tensor([b for b, _ in component_specs],
                                 dtype=torch.int64, device=device)
    teammate_indices = torch.tensor([t for _, t in component_specs],
                                    dtype=torch.int64, device=device)
    present = torch.zeros(B, T, dtype=torch.bool, device=device)
    present[batch_indices, teammate_indices] = True

    img_q = torch.randn(Tc, L_img, H, Fd, device=device, dtype=dtype)
    img_k = torch.randn(Tc, L_img, H, Fd, device=device, dtype=dtype)
    img_v = torch.randn(Tc, L_img, H, Fd, device=device, dtype=dtype)
    txt_q = torch.randn(B, L_text, H, Fd, device=device, dtype=dtype)
    txt_k = torch.randn(B, L_text, H, Fd, device=device, dtype=dtype)
    txt_v = torch.randn(B, L_text, H, Fd, device=device, dtype=dtype)

    if with_rotary:
        S_total = L_text + T * L_img
        cos = torch.randn(S_total, Fd, device=device, dtype=dtype)
        sin = torch.randn(S_total, Fd, device=device, dtype=dtype)
        rotary = (cos, sin)
    else:
        rotary = None

    if with_attn_keep:
        attn_keep = torch.ones(T, T, dtype=torch.bool, device=device)
        attn_keep[0, 2] = False
        attn_keep[2, 0] = False
    else:
        attn_keep = None

    img_out, txt_out = _masked_attention_impl(
        img_q, img_k, img_v,
        txt_q, txt_k, txt_v,
        rotary,
        batch_indices, teammate_indices,
        present, attn_keep,
    )
    ref_img, ref_txt = reference_impl(
        img_q, img_k, img_v,
        txt_q, txt_k, txt_v,
        rotary,
        batch_indices, teammate_indices,
        present, attn_keep,
    )

    img_diff = (img_out - ref_img).abs().max().item()
    txt_diff = (txt_out - ref_txt).abs().max().item()
    tol = 5e-4
    status = "PASS" if (img_diff < tol and txt_diff < tol) else "FAIL"
    print(f"[{status}] {name}: img max-abs-diff={img_diff:.2e}, "
          f"txt max-abs-diff={txt_diff:.2e}")
    return img_diff < tol and txt_diff < tol


def main():
    if not torch.cuda.is_available():
        print("flex_attention requires CUDA; skipping.")
        sys.exit(0)

    cases = [
        ("dense    , no-rotary, no-keep", False, False, False),
        ("dense    , rotary   , no-keep", True,  False, False),
        ("dense    , no-rotary, keep   ", False, True,  False),
        ("dense    , rotary   , keep   ", True,  True,  False),
        ("jagged   , no-rotary, no-keep", False, False, True),
        ("jagged   , rotary   , no-keep", True,  False, True),
        ("jagged   , no-rotary, keep   ", False, True,  True),
        ("jagged   , rotary   , keep   ", True,  True,  True),
    ]
    all_pass = True
    for name, rotary, keep, jagged in cases:
        ok = run_case(name, with_rotary=rotary, with_attn_keep=keep, jagged=jagged)
        all_pass = all_pass and ok
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
