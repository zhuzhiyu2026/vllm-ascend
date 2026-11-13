# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative-decoding DFX dump helper (v1 model-runner diagnosis aid).

Always-on while the standard DEBUG logging level is enabled (same switch as
the existing ``[spec/dfx]`` probes: ``VLLM_LOGGING_LEVEL=DEBUG`` turns every
dump on; INFO leaves them off, paying one integer compare per call site).
No trigger state machine, no record budget: every spec-decode step logs the
full chain, so a hang is directly visible as frozen ``A1`` lines.

Dump points and the decision tree they implement (one run with a real
request localizes the failure layer):

  A1 propose_entry    seed source (prev-step sampled tokens) + req_ids
                      + CPU-side metadata (num_computed tokens, sequence
                      tails, CPU-side scheduled drafts, previous verify
                      output) for host/device accounting drift analysis
  A2 seed_inputs      seed written into the drafter + num_rejected + seq_lens
  E  draft_logits     draft logits finite? + per-position argmax
  B  dspark_out       seed + proposed tokens as seen by the proposer
  C  verify_entry     drafts as seen by the verifier + target argmax + pos0_match
  D  verify_exit      sampler output rows + bonus + all_placeholder flag
  F  fallthrough      one-shot warning if dspark runs the sequential loop
                      (parallel_drafting=False) -- structural bug for DSpark.

Reading the result (temporal signature: the reported failure is "healthy at
startup, collapsing later", which rules out structural misconfig and points
at step-count/context-length accumulation):
  0. Find the transition step: the first step where C pos0_match goes
     all-False and D output_rows turn all-placeholder. Everything that
     changed at that boundary is the suspect set.
  1. F present (from step 1)         -> parallel_drafting misconfig, done.
  2. E non-finite / constant argmax  -> draft forward poisoned (attn meta / KV prefill).
  3. B invalid ids (0 / repeated)    -> markov loop / draft buffer bug.
  4. B != C for the same step        -> runner<->scheduler metadata transport bug.
  5. C pos0_match always False, B sane
       and drafts look one-step-behind target argmax -> seed/alignment shift.
  6. C pos0_match mostly True but D shows accepted=0 -> rejection sampler/kernel bug.
  7. Collapse correlates with seq_lens growth (A2) -> SWA/DSA sparse-index
     or draft-KV prepopulation issue (dspark_swa_indices / DSA device metadata).
  8. A1 cpu_num_computed / cpu_seq_tail frozen or drifting vs the engine
     progress -> host-side accounting corruption (the "never recovers" class).
"""

import logging

import torch

_STEP = 0


def next_step() -> int:
    """Advance the per-process spec-decode step counter (call once per engine step)."""
    global _STEP
    _STEP += 1
    return _STEP


def cur_step() -> int:
    return _STEP


def active(logger: logging.Logger) -> bool:
    """True while the standard DEBUG gate is open."""
    return logger.isEnabledFor(logging.DEBUG)


def _fmt(value, max_rows: int = 4):
    if value is None:
        return "None"
    if isinstance(value, torch.Tensor):
        t = value.detach()
        if t.dim() == 0:
            return t.item()
        return t[:max_rows].tolist()
    if isinstance(value, (list, tuple)):
        return list(value)[:max_rows]
    return value


def dump(logger: logging.Logger, tag: str, **fields) -> None:
    """Log one DFX record via logger.debug. Never raises: diagnostics must not take down the engine."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        parts = [f"step={_STEP}", tag]
        for key, val in fields.items():
            parts.append(f"{key}={_fmt(val)}")
        logger.debug("[spec/dfx2] %s", " ".join(str(p) for p in parts))
    except Exception as e:  # noqa: BLE001
        logger.debug("[spec/dfx2] step=%d %s dump failed: %s", _STEP, tag, e)
