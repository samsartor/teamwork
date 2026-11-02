"""
Unit tests for the sequence-axis teamwork core (`teamwork.sequence_linear`, `teamwork.sequence_mask`)
against dense references:

- `to_sequence`/`from_sequence` round trip (exact);
- `SequenceTeamworkLinear` and `SequenceTeamworkModulation` vs the batch-axis `BatchTeamworkLinear`
  on the same per-teammate tokens (fp32), jagged presence, communication on/off, bias;
- the flex `BlockMask` vs a dense boolean reference for both mask rules, and the Qwen-Image 2.1
  `(edited.out, source.in)` ti2i mask vs diffusers' own block-causal mask.

Run from the repo root:

    uv run python scripts/test_sequence_core.py
"""

import sys
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention, _ordered_to_dense

from teamwork.config import TeamworkConfig
from teamwork.data import Components
from teamwork.batch_linear import BatchSelection, BatchTeamworkLinear
from teamwork.sequence_linear import (
    SequenceLayout,
    SequenceSelection,
    SequenceTeamworkLinear,
    SequenceTeamworkModulation,
    from_sequence,
    to_sequence,
)
from teamwork.sequence_mask import sequence_block_mask, sequence_dense_mask, sequence_mask_mod

FP32_TOL = 1e-5
# The fused kernel, which honors the BlockMask's block sparsity (the eager fallback only warns).
COMPILED_FLEX = torch.compile(flex_attention, dynamic=False)
RESULTS: list[bool] = []


def report(name: str, diff: float, tol: float) -> None:
    ok = diff <= tol
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max-abs-diff={diff:.2e} (tol {tol:.0e})")


def report_exact(name: str, mismatches: int) -> None:
    ok = mismatches == 0
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: mismatches={mismatches}")


def max_abs(a: Tensor, b: Tensor) -> float:
    assert a.shape == b.shape, (a.shape, b.shape)
    return (a - b).abs().max().item() if a.numel() > 0 else 0.0


# --------------------------------------------------------------------------- layouts & components

ROSTER = ["edited.out", "source.in", "mask.in"]
OUTPUTS = tuple(name.endswith(".out") for name in ROSTER)


def joint_layout(rule="joint") -> SequenceLayout:
    layout = SequenceLayout.joint(text_length=7, image_length=12, outputs=OUTPUTS, rule=rule)
    assert layout.uniform_start == 7
    return layout


def qwen_like_layout(rule="block_causal") -> SequenceLayout:
    # Images before and after text, slots out of stream order, uneven gaps.
    layout = SequenceLayout.from_pieces(
        [("text", 5), ("image", 1), ("text", 9), ("image", 2), ("image", 0), ("text", 3)],
        image_length=12,
        outputs=OUTPUTS,
        rule=rule,
    )
    assert layout.uniform_start is None
    assert layout.image_offsets == (38, 5, 26), layout.image_offsets
    return layout


def jagged_components(device: torch.device | str = "cpu") -> Components:
    # b0: all three, b1: edited + mask, b2: source only. Shuffled component order.
    pairs = [(1, 2), (0, 1), (2, 1), (0, 0), (1, 0), (0, 2)]  # (batch, teammate)
    batch = torch.tensor([b for b, _ in pairs], device=device)
    team = torch.tensor([t for _, t in pairs], device=device)
    outputs = [i for i, (_, t) in enumerate(pairs) if OUTPUTS[t]]
    inputs = [i for i, (_, t) in enumerate(pairs) if not OUTPUTS[t]]
    return Components(
        teammate_indices=team,
        batch_indices=batch,
        input_subindices=torch.tensor(inputs, device=device),
        output_subindices=torch.tensor(outputs, device=device),
        num_teammates=3,
        batch_size=3,
    )


def full_components(B: int, T: int) -> Components:
    team = torch.arange(T).repeat(B)
    batch = torch.arange(B).repeat_interleave(T)
    return Components(
        teammate_indices=team,
        batch_indices=batch,
        input_subindices=torch.tensor([], dtype=torch.int64),
        output_subindices=torch.arange(B * T),
        num_teammates=T,
        batch_size=B,
    )


def batch_selection(components: Components) -> BatchSelection:
    """The batch-axis selection equivalent to `components` (the dense LoRA reference)."""
    return BatchSelection.from_components(components)


def single_teammate_components(B: int, T: int, teammate: int) -> Components:
    return Components(
        teammate_indices=torch.full((B,), teammate),
        batch_indices=torch.arange(B),
        input_subindices=torch.tensor([], dtype=torch.int64),
        output_subindices=torch.arange(B),
        num_teammates=T,
        batch_size=B,
    )


# --------------------------------------------------------------------------- to/from_sequence


def test_round_trip(layout_name: str, layout: SequenceLayout) -> None:
    torch.manual_seed(0)
    comps = jagged_components()
    D = 5
    count = comps.teammate_indices.shape[0]
    img = torch.randn(count, layout.image_length, D)
    txt = torch.randn(comps.batch_size, layout.text_length, D)
    stream = to_sequence(img, txt, comps, layout)
    assert stream.shape == (comps.batch_size, layout.seq_len, D)
    img2, txt2 = from_sequence(stream, comps, layout)
    mismatches = int((img2 != img).sum()) + int((txt2 != txt).sum())

    # Absent slots hold exact zeros; present slots hold their component at the static offset.
    present = SequenceSelection.from_components(comps, layout, attn_keep=None, text_teammate=None).present
    L = layout.image_length
    for b in range(comps.batch_size):
        for t, o in enumerate(layout.image_offsets):
            block = stream[b, o : o + L]
            if bool(present[b, t]):
                (c,) = ((comps.batch_indices == b) & (comps.teammate_indices == t)).nonzero()[:, 0].tolist()
                mismatches += int((block != img[c]).sum())
            else:
                mismatches += int((block != 0).sum())
    report_exact(f"round trip, {layout_name}", mismatches)


# --------------------------------------------------------------------------- LoRA linear


def make_cfg(communicate: bool, use_bias: bool) -> TeamworkConfig:
    return TeamworkConfig(
        teammates=ROSTER,
        lora_rank=4,
        base_model="test",
        title="test",
        lora_communication=communicate,
        use_bias=use_bias,
    )


def randomize(adapter: nn.Module) -> None:
    with torch.no_grad():
        for p in adapter.parameters():
            p.normal_(0.0, 0.3)


AdapterFactory = Callable[[nn.Linear, TeamworkConfig], nn.Module]


def build_pair(factory: AdapterFactory, cfg: TeamworkConfig, d_in: int, d_out: int):
    base = nn.Linear(d_in, d_out, bias=True)
    seq = factory(base, cfg)
    randomize(seq.adapter)
    ref = BatchTeamworkLinear(base, cfg)
    ref.adapter = seq.adapter  # shared parameters
    for p in base.parameters():
        p.requires_grad_(False)
    return base, seq, ref


def test_linear(layout_name: str, layout: SequenceLayout, communicate: bool, use_bias: bool, text_teammate: int | None):
    torch.manual_seed(1)
    cfg = make_cfg(communicate, use_bias)
    d_in, d_out = 24, 20
    base, seq, ref = build_pair(SequenceTeamworkLinear, cfg, d_in, d_out)
    assert isinstance(seq, SequenceTeamworkLinear)
    comps = jagged_components()
    B, T = comps.batch_size, comps.num_teammates

    # Garbage (not zeros) in absent slots: it must not leak through communication.
    x = torch.randn(B, layout.seq_len, d_in, requires_grad=True)
    seq.selection = SequenceSelection.from_components(comps, layout, attn_keep=None, text_teammate=text_teammate)
    out = seq(x)
    seq.selection = None

    img_x, txt_x = from_sequence(x, comps, layout)
    ref.selection = batch_selection(comps)
    ref_img = ref(img_x)
    ref.selection = None
    out_img, out_txt = from_sequence(out, comps, layout)

    tag = f"{layout_name}, comm={'on ' if communicate else 'off'}, bias={'on ' if use_bias else 'off'}, text={text_teammate}"
    report(f"linear image tokens  {tag}", max_abs(out_img, ref_img), FP32_TOL)

    if text_teammate is None:
        ref_txt = base(txt_x)
    else:
        ref.selection = batch_selection(single_teammate_components(B, T, text_teammate))
        ref_txt = ref(txt_x)
        ref.selection = None
    report(f"linear text tokens   {tag}", max_abs(out_txt, ref_txt), FP32_TOL)

    # Absent slots receive exactly the base output.
    present = SequenceSelection.from_components(comps, layout, attn_keep=None, text_teammate=None).present
    base_out = base(x)
    absent_diff = 0.0
    L = layout.image_length
    for b, t in (~present).nonzero().tolist():
        o = layout.image_offsets[t]
        absent_diff = max(absent_diff, max_abs(out[b, o : o + L], base_out[b, o : o + L]))
    report(f"linear absent = base {tag}", absent_diff, FP32_TOL)

    # Gradients through the image tokens (in-place view / slice adds) vs the reference.
    g = torch.randn_like(ref_img)
    params = [p for p in seq.adapter.parameters()]
    seq_grads = torch.autograd.grad((out_img * g).sum(), [x, *params], allow_unused=True)
    ref_grads = torch.autograd.grad((ref_img * g).sum(), [x, *params], allow_unused=True)
    grad_diff = 0.0
    for a, b in zip(seq_grads, ref_grads):
        if a is None and b is None:
            continue
        a = torch.zeros_like(b) if a is None else a
        b = torch.zeros_like(a) if b is None else b
        grad_diff = max(grad_diff, max_abs(a, b))
    report(f"linear grads         {tag}", grad_diff, FP32_TOL)


def test_modulation(communicate: bool, use_bias: bool, text_teammate: int | None):
    torch.manual_seed(2)
    cfg = make_cfg(communicate, use_bias)
    d_in, d_out = 16, 36
    base, seq, ref = build_pair(SequenceTeamworkModulation, cfg, d_in, d_out)
    assert isinstance(seq, SequenceTeamworkModulation)
    comps = jagged_components()
    B, T = comps.batch_size, comps.num_teammates
    layout = joint_layout()

    team_temb = torch.randn(B, T, d_in)
    text_temb = torch.randn(B, d_in)
    seq.selection = SequenceSelection.from_components(comps, layout, attn_keep=None, text_teammate=text_teammate)
    rows = seq.rows(text_temb, team_temb)
    try:
        seq(text_temb)
        raise AssertionError("plain forward must refuse to run with a selection")
    except AssertionError as e:
        assert "rows()" in str(e), e
    seq.selection = None

    ref.selection = batch_selection(comps)
    ref_rows = ref(team_temb[comps.batch_indices, comps.teammate_indices])
    ref.selection = None
    tag = f"comm={'on ' if communicate else 'off'}, bias={'on ' if use_bias else 'off'}, text={text_teammate}"
    seq_rows = rows.teammates[comps.batch_indices, comps.teammate_indices]
    report(f"modulation teammate rows {tag}", max_abs(seq_rows, ref_rows), FP32_TOL)

    absent = ~SequenceSelection.from_components(comps, layout, attn_keep=None, text_teammate=None).present
    report(f"modulation absent = base {tag}", max_abs(rows.teammates[absent], base(team_temb[absent])), FP32_TOL)

    if text_teammate is None:
        ref_text = base(text_temb)
    else:
        ref.selection = batch_selection(single_teammate_components(B, T, text_teammate))
        ref_text = ref(text_temb)
        ref.selection = None
    report(f"modulation text row      {tag}", max_abs(rows.text, ref_text), FP32_TOL)


# --------------------------------------------------------------------------- masks


def grid_mask(mask_mod, B: int, S: int, device) -> Tensor:
    b = torch.arange(B, device=device)[:, None, None]
    h = torch.zeros((), dtype=torch.int64, device=device)
    q = torch.arange(S, device=device)[None, :, None]
    kv = torch.arange(S, device=device)[None, None, :]
    return mask_mod(b, h, q, kv).expand(B, S, S).cpu()


def block_mask_effective(bm: BlockMask, B: int, S: int) -> Tensor:
    """What flex_attention actually computes with `bm`: full blocks all-True, partial via mask_mod."""
    qbs, kbs = bm.BLOCK_SIZE
    partial = _ordered_to_dense(bm.kv_num_blocks, bm.kv_indices)[:, 0].bool()  # (B, nq, nk)
    if bm.full_kv_num_blocks is not None:
        assert bm.full_kv_indices is not None
        full = _ordered_to_dense(bm.full_kv_num_blocks, bm.full_kv_indices)[:, 0].bool()
    else:
        full = torch.zeros_like(partial)
    expand = lambda m: m.repeat_interleave(qbs, 1).repeat_interleave(kbs, 2)[:, :S, :S].cpu()  # noqa: E731
    mod = grid_mask(bm.mask_mod, B, S, partial.device)
    return expand(full) | (expand(partial) & mod)


def check_flex_vs_sdpa(name: str, selection: SequenceSelection, dense: Tensor) -> None:
    if not torch.cuda.is_available():
        print(f"[SKIP] {name}: flex_attention needs CUDA")
        return
    device = torch.device("cuda")
    sel = SequenceSelection(
        layout=selection.layout,
        present=selection.present.to(device),
        attn_keep=None if selection.attn_keep is None else selection.attn_keep.to(device),
        text_teammate=selection.text_teammate,
        key_valid=None if selection.key_valid is None else selection.key_valid.to(device),
    )
    B, S, H, Dh = sel.batch_size, sel.layout.seq_len, 2, 16
    torch.manual_seed(3)
    q, k, v = (torch.randn(B, H, S, Dh, device=device) for _ in range(3))
    out = COMPILED_FLEX(q, k, v, block_mask=sequence_block_mask(sel))
    assert isinstance(out, Tensor)
    dense_d = dense.to(device)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_d[:, None])
    rows = dense_d.any(-1)[:, None, :, None].expand_as(out)  # fully masked rows: flex gives 0, SDPA NaN
    diff = max(max_abs(out[rows], ref[rows]), out[~rows].abs().max().item() if bool((~rows).any()) else 0.0)
    report(f"flex vs SDPA(dense)  {name}", diff, 1e-5)


def test_mask(name: str, layout: SequenceLayout, present: Tensor, attn_keep: Tensor | None, key_valid: Tensor | None = None):
    sel = SequenceSelection(layout=layout, present=present, attn_keep=attn_keep, text_teammate=None, key_valid=key_valid)
    B, S = sel.batch_size, layout.seq_len
    dense = sequence_dense_mask(sel)
    mod = grid_mask(sequence_mask_mod(sel), B, S, "cpu")
    eff = block_mask_effective(sequence_block_mask(sel), B, S)
    report_exact(f"mask_mod vs dense    {name}", int((mod != dense).sum()))
    report_exact(f"BlockMask vs dense   {name}", int((eff != dense).sum()))
    check_flex_vs_sdpa(name, sel, dense)


def test_masks():
    T = 3
    jagged = torch.tensor([[True, True, True], [True, False, True], [False, True, False]])
    keep = torch.tensor([[True, False, True], [True, True, False], [False, True, True]])
    # Big enough for several 128-blocks, and not a multiple of 128.
    big_joint = SequenceLayout.joint(text_length=50, image_length=80, outputs=OUTPUTS)
    big_qwen = SequenceLayout.from_pieces(
        [("text", 30), ("image", 1), ("text", 45), ("image", 2), ("image", 0), ("text", 11)],
        image_length=80,
        outputs=OUTPUTS,
        rule="block_causal",
    )
    # Two outputs apart, and a non-VLM input placed after an output.
    roster4 = (True, True, False, False)  # edited.out, matte.out, source.in, extra.in
    big_qwen4 = SequenceLayout.from_pieces(
        [("text", 30), ("image", 2), ("text", 25), ("image", 0), ("image", 3), ("image", 1)],
        image_length=64,
        outputs=roster4,
        rule="block_causal",
    )
    full = torch.ones(3, T, dtype=torch.bool)
    key_valid = torch.ones(3, big_qwen.seq_len, dtype=torch.bool)
    key_valid[0, 100:110] = False
    key_valid[2, 120:125] = False
    for lname, layout in [("joint", big_joint), ("qwen ", big_qwen)]:
        test_mask(f"{lname} full", layout, full, None)
        test_mask(f"{lname} jagged", layout, jagged, None)
        test_mask(f"{lname} jagged+keep", layout, jagged, keep)
    test_mask("qwen  jagged+keep+key_valid", big_qwen, jagged, keep, key_valid)
    keep4 = torch.ones(4, 4, dtype=torch.bool)
    keep4[0, 1] = keep4[3, 2] = keep4[2, 0] = False
    jagged4 = torch.tensor([[True, True, True, True], [True, False, True, True]])
    test_mask("qwen4 jagged+keep (out-before-in)", big_qwen4, jagged4, keep4)

    # Sanity: joint + full presence + no keep is unmasked attention.
    sel = SequenceSelection(layout=big_joint, present=full, attn_keep=None, text_teammate=None)
    report_exact("joint full = all-True", int((~sequence_dense_mask(sel)).sum()))


def test_qwen_matches_diffusers():
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21Transformer2DModel,
        build_qwenimage21_block_causal_mask,
    )

    # Base ti2i: `…user\n<image1><vs>[SRC]<ve>{prompt}<|im_end|>\n<|im_start|>assistant\n<pad…>[TARGET]`.
    header, prompt, pad = 9, 37, 6
    h, w = 8, 10
    L = h * w
    B = 2
    S = header + L + prompt + pad + L
    image_pad_mask = torch.zeros(S, dtype=torch.bool)
    image_pad_mask[header : header + L] = True
    image_pad_mask[S - L :] = True
    image_ids, target_mask = QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, [(1, h, w), (1, h, w)])
    key_valid = torch.ones(B, S, dtype=torch.bool)
    key_valid[0, S - L - pad : S - L] = False  # b0 right-padded by `pad`
    key_valid[1, S - L - 2 : S - L] = False  # b1 by 2
    theirs = build_qwenimage21_block_causal_mask(image_ids, key_valid, B, torch.device("cpu"))
    theirs_grid = grid_mask(theirs.mask_mod, B, S, "cpu")
    theirs_eff = block_mask_effective(theirs, B, theirs.seq_lengths[0])[:, :S, :S]
    assert bool((theirs_grid == theirs_eff).all())

    # Roster (edited.out, source.in), vlm_inputs = [source.in].
    layout = SequenceLayout.from_pieces(
        [("text", header), ("image", 1), ("text", prompt + pad), ("image", 0)],
        image_length=L,
        outputs=(True, False),
        rule="block_causal",
    )
    assert (layout.slot_of_token("cpu") == torch.where(image_ids == 0, 1, torch.where(image_ids == 1, 0, -1))).all()
    assert (target_mask == (layout.slot_of_token("cpu") == 0)).all()
    sel = SequenceSelection(
        layout=layout,
        present=torch.ones(B, 2, dtype=torch.bool),
        attn_keep=None,
        text_teammate=None,
        key_valid=key_valid,
    )
    ours_dense = sequence_dense_mask(sel)
    ours_mod = grid_mask(sequence_mask_mod(sel), B, S, "cpu")
    report_exact("qwen ti2i: dense vs diffusers", int((ours_dense != theirs_grid).sum()))
    report_exact("qwen ti2i: mask_mod vs diffusers", int((ours_mod != theirs_grid).sum()))


def main():
    for name, layout in [("uniform", joint_layout()), ("non-uniform", qwen_like_layout())]:
        test_round_trip(name, layout)
    for name, layout in [("uniform    ", joint_layout()), ("non-uniform", qwen_like_layout())]:
        for communicate in (True, False):
            for use_bias in (True, False):
                for text_teammate in (None, 0, 2):
                    test_linear(name, layout, communicate, use_bias, text_teammate)
    for communicate in (True, False):
        for use_bias in (True, False):
            for text_teammate in (None, 1):
                test_modulation(communicate, use_bias, text_teammate)
    test_masks()
    test_qwen_matches_diffusers()
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
