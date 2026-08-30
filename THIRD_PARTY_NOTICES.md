# Third-party notices

## EchoPrime

<https://github.com/echonet/EchoPrime> — © 2024 Cedars-Sinai Medical Center,
released under the **EchoPrime Academic Software License** (academic and
nonprofit research use only; commercial use requires a separate license from
Cedars-Sinai).

EchoSonar uses EchoPrime as a **frozen** clip encoder and view classifier. What
that means concretely:

| Item | How EchoSonar handles it |
|---|---|
| `echo_prime_encoder.pt`, `view_classifier.pt` | **Not redistributed.** Download them from the EchoPrime release and pass `--weights-dir`. |
| `assets/MIL_weights.csv` | **Not redistributed.** It's EchoPrime's file; grab it from your EchoPrime checkout (see README) — `--mil-weights` and the auto-search look for it there. |
| `crop_and_scale` | Reproduced verbatim in `echosonar/third_party/echoprime.py`, with attribution. |
| `mask_outside_ultrasound` | Re-implemented in `echosonar/io/dicom.py` (same algorithm, working in RGB instead of YUV). A derivative work. |
| Normalisation constants, view names, tensor shapes | Restated in `echosonar/constants.py`. |

### Why the code is copied rather than imported

`EchoPrime/utils/utils.py` opens `assets/per_section.json` at module import
time, so importing it requires the process working directory to be the
EchoPrime checkout. That global side effect is unsafe in library code and breaks
in `spawn`-based multiprocessing workers, which is why the two small pure
functions above are carried here instead of imported.

`crop_and_scale` in particular is reproduced rather than rewritten on purpose:
the crop offsets, the 10% zoom and the cubic interpolation kernel all feed
straight into the embeddings, and a re-implementation that differs by a pixel
would produce features that look fine but do not match the released ones.

### If you need a permissively licensed release

Anyone running this pipeline must accept the EchoPrime license anyway to obtain
the weights and `MIL_weights.csv`. If you nonetheless want EchoSonar's own code
under a permissive license, `echosonar/third_party/` is the only piece to carve
out: keep it under the EchoPrime terms (or drop it and require a
`--echoprime-dir`), and license the rest as you choose.
