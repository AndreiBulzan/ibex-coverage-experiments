# Ibex Coverage Experiments

Hands-on verification experiments on the lowRISC Ibex RISC-V core,
demonstrating the stimulus-simulate-measure loop that an AI coverage agent
would run. Everything here uses free, open-source tools and real chip RTL.

## What is in this repo

### `decoder/` -- Combinational decoder (isolated)

The Ibex instruction decoder is a purely combinational block (~1800 lines of
SystemVerilog). We extracted it from the Ibex repo and wrapped it so we can
feed it 32-bit instruction words from Python and instantly read what operation
it decoded. No clock, no pipeline, no memory -- just input-output.

Two tests live here:

- **`test_decoder.py`** runs three phases to illustrate the coverage loop:
  1. Hand-crafted instructions (a human picks ADD, SUB, LW, etc.)
  2. Pure random 32-bit values (baseline -- most are illegal)
  3. Encoding-aware random (understands RISC-V bit fields, much more effective)

- **`test_max_coverage.py`** does a systematic sweep: every operation times
  every register (0..31) across every port. Result: **2041 / 2107 bins
  covered**. The remaining ~65 bins are ISA-unreachable (e.g., RISC-V has no
  SUBI instruction, so the subtract-immediate cross-coverage bins can never
  fire).

The coverage model matches the one from the LLM4DV paper (ZixiBenZhang/ml4dv),
which defines 2107 bins across three types:
  - Type 1: which ALU/memory operations were seen (26 bins)
  - Type 2: which register ports were exercised (96 bins)
  - Type 3: cross-coverage, operation x register (1985 bins)

### `cpu/` -- Full Ibex CPU with RTL toggle coverage

The full Ibex RISC-V CPU (~15K lines of SystemVerilog, minimal configuration).
We load machine-code programs into a simulated instruction memory, let the CPU
execute them, and observe which instructions retire through the RVFI (RISC-V
Formal Interface).

- **`test_cpu_coverage.py`** builds a single program that targets all 196
  LLM4DV CPU benchmark bins. These 196 bins cover six categories:

  | Type | What it measures | Bins |
  |------|-----------------|------|
  | SEEN | Each of the 14 operations executed at least once | 14 |
  | ZERO_DST | R-type or JAL with destination register = x0 | 11 |
  | ZERO_SRC | R-type or S-type with a source register = x0 | 13 |
  | SAME_SRC | R-type or S-type with rs1 == rs2 | 13 |
  | BR | JAL forward + JAL backward | 2 |
  | RAW_HAZARD | Read-after-write data dependency between consecutive instructions | 143 |

  **Result: 196/196 = 100%** (the LLM4DV paper's best published result using
  Claude 3.5 Sonnet was 5.61%).

- **`instruction_monitor.py`** is the coverage monitor, adapted from LLM4DV
  with a bug fix. The original code cleared `last_insn` on every idle cycle
  (when `rvfi_valid == 0`). In a pipelined CPU there are always idle cycles
  between instruction retirements, so the original monitor could never see two
  consecutive instructions -- making RAW hazard detection impossible. Our fix:
  only update `last_insn` when a new instruction actually retires.

- **Verilator RTL toggle coverage** (`--coverage` flag in Makefile) measured
  how much of the actual hardware was exercised:

  **985 / 5746 toggle points = 17%**

  Per-module breakdown (selected):

  | Module | Covered | Total | Rate |
  |--------|---------|-------|------|
  | ibex_alu.sv | 47 | 119 | 39% |
  | ibex_controller.sv | 87 | 298 | 29% |
  | ibex_cs_registers.sv | 79 | 422 | 18% |
  | ibex_decoder.sv | 68 | 530 | 12% |
  | ibex_compressed_decoder.sv | 9 | 107 | 8% |
  | ibex_multdiv_fast.sv | 38 | 169 | 22% |

  The 17% is expected: our test only uses ~14 instruction types out of 50+.
  Major uncovered areas include branches (BEQ/BNE/...), loads (LB/LH/LW),
  LUI, AUIPC, CSR operations, MUL/DIV, compressed instructions, exceptions,
  and interrupts. A natural next step is writing tests that exercise these
  paths to push toggle coverage significantly higher.

### CPU configuration

The cocotb wrapper (`cocotb_ibex.sv`) instantiates Ibex in a minimal
configuration: no PMP, no ICache, no debug triggers, no security hardening,
2-stage pipeline, no branch predictor. All interrupts are tied to zero.
These parameters can be changed to exercise more of the design.

For comparison, lowRISC's own verification of Ibex targets the "opentitan"
configuration (PMP with 16 regions, ICache, debug, security, 3-stage
pipeline) and achieves 88.7% branch coverage / 90% functional coverage
across ~50K-100K coverage bins using 1530 regression tests and commercial
simulators.


## How to set up

### Prerequisites

Tested on WSL2 Ubuntu with Anaconda Python 3.13.

```
# Verilator (simulator) and compression library
sudo apt-get install verilator zlib1g-dev

# cocotb (Python-to-simulator bridge)
pip install 'cocotb>=1.8,<2.0'
```

Verilator 5.x is required. Check with `verilator --version`.


## How to run

### Decoder demo

```bash
cd decoder/

# Build (first time, or after editing .sv files)
make

# Run the 3-phase demo
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu MODULE=test_decoder ./sim_build/Vtop

# Run the systematic max-coverage sweep
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu MODULE=test_max_coverage ./sim_build/Vtop
```

### Full CPU test

```bash
cd cpu/

# Build (first time, or after editing .sv files)
make

# Run the 196-bin coverage test
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu MODULE=test_cpu_coverage ./sim_build/Vtop

# Analyze RTL toggle coverage (after running the test)
verilator_coverage --annotate /tmp/ibex_cov_annotate coverage.dat
```

The `--annotate` command writes per-file annotated source to
`/tmp/ibex_cov_annotate/`. Lines with `%000000` were never toggled.

### Build troubleshooting

If the build fails at the `verilator_includer` step (Anaconda's libstdc++
conflicting with system libraries), you can manually concatenate the generated
C++ files. From inside `sim_build/`:

```bash
cat Vtop.cpp Vtop___024root__DepSet_*.cpp Vtop__Dpi.cpp \
    Vtop__Trace__0.cpp Vtop___024root__Slow.cpp \
    Vtop_ibex_pkg__Slow.cpp Vtop_ibex_pkg__DepSet_*.cpp \
    Vtop__Syms.cpp Vtop__Trace__0__Slow.cpp \
    Vtop__TraceDecls__0__Slow.cpp > Vtop__ALL.cpp
make -f Vtop.mk
```

You may also need to set `LD_LIBRARY_PATH` to include both
`/usr/lib/x86_64-linux-gnu` (system libstdc++) and your Python installation's
`lib/` directory (for `libpython3.x.so`).


## Where to go from here

The infrastructure is ready. The loop is: generate instructions, load them into
the simulated CPU, run, measure coverage. An AI agent slots in at the "generate
instructions" step.

Concrete next steps for pushing the 17% toggle coverage higher:

1. **More instruction types** -- add I-type ALU (ADDI, XORI, ...), loads
   (LB/LH/LW/LBU/LHU), branches (BEQ/BNE/BLT/BGE/...), LUI, AUIPC, JALR,
   CSR operations, MUL/DIV, and compressed (16-bit) instructions.

2. **Control flow** -- take branches both ways (taken and not-taken), trigger
   pipeline flushes, exercise the branch predictor (if enabled).

3. **Exceptions and interrupts** -- send ECALL/EBREAK, feed illegal
   instructions, drive `irq_timer_i` / `irq_external_i` from cocotb, set up
   mtvec and an interrupt handler in the test program.

4. **Richer configuration** -- enable PMP, ICache, debug triggers, writeback
   stage in `cocotb_ibex.sv` to expose more RTL paths.

The annotated coverage output (`verilator_coverage --annotate`) shows exactly
which RTL lines are uncovered, making it straightforward to target gaps.


## Attribution

- Ibex RTL: [lowRISC/ibex](https://github.com/lowRISC/ibex), Apache-2.0
- Coverage model and benchmark bins: [ZixiBenZhang/ml4dv](https://github.com/ZixiBenZhang/ml4dv) (LLM4DV), Apache-2.0
- `instructions.py`, `shared_types.py`: from LLM4DV, Apache-2.0 (imports adjusted)
- `instruction_monitor.py`: from LLM4DV, Apache-2.0, with bug fix (idle-cycle `last_insn` clearing removed)
- `test_decoder.py`, `test_max_coverage.py`, `test_cpu_coverage.py`: original work
