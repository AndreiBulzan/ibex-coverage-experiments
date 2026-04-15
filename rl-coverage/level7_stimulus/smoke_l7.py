"""Sanity-check the Level 7 pipeline end-to-end.

Runs a short program that includes the three new ops (AUIPC, ECALL, EBREAK)
so we can confirm:
  (a) Vtop doesn't crash,
  (b) the exception path activates — controller module toggle rises,
  (c) compressed_decoder stays hot (RVC still works),
  (d) data bus toggles more than before (memory prepop works).
"""

import time
from pathlib import Path

from codec_l7 import Op
from env_l7 import run_program

ACTIONS = []
# A burst of each new op with diverse fields
for ib in range(5):
    ACTIONS.append((int(Op.AUIPC), 7, 0, 0, ib))
ACTIONS.append((int(Op.ECALL), 0, 0, 0, 0))
ACTIONS.append((int(Op.EBREAK), 0, 0, 0, 0))
# Some loads from diverse addresses to exercise the memory prepop
for off in range(5):
    ACTIONS.append((int(Op.LW), 10, 5, 0, off))    # LW x10, off(x5)
    ACTIONS.append((int(Op.LB), 11, 5, 0, off))    # LB x11, off(x5)
# Some RVC to confirm those still work
for ib in range(5):
    ACTIONS.append((int(Op.C_ADDI), 8, 0, 0, ib))
    ACTIONS.append((int(Op.C_ADD),  8, 0, 9, ib))
# Repeat to 256 total
ACTIONS = (ACTIONS * (256 // len(ACTIONS) + 1))[:256]

print(f"Running {len(ACTIONS)} actions...")
t0 = time.time()
summary = run_program(ACTIONS)
dt = time.time() - t0
assert summary is not None, "Vtop failed"

for kind in ("toggle", "branch", "line"):
    c, t = summary.by_kind[kind]
    print(f"  {kind:7s}: {c}/{t} = {summary.kind_pct(kind):.2f}%")
print(f"  wall: {dt:.1f}s")

print("\nPer-module toggle (exception-path modules highlighted):")
interesting = ["ibex_compressed_decoder", "ibex_controller", "ibex_cs_registers",
               "ibex_decoder", "ibex_alu", "ibex_load_store_unit",
               "ibex_multdiv_fast", "ibex_csr", "ibex_id_stage"]
for page, (c, t) in sorted(summary.by_page.items()):
    if not page.startswith("v_toggle/"): continue
    mod = page[len("v_toggle/"):].split("__")[0]
    if mod in interesting:
        pct = 100 * c / t if t else 0
        print(f"  {mod:<32s} {c:>4}/{t:<4} = {pct:5.1f}%")
