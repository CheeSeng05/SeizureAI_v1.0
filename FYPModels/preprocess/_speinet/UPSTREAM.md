# SPEINet source attribution

- Repository: <https://github.com/yangt1013/SPEINet>
- Revision: `3aaa3f1e761e8047f50e6fe235b9ea61389dfbf9`
- Paper: [Video Deblurring by Sharpness Prior Detection and Edge Information](https://arxiv.org/abs/2501.12246), Yang Tian et al., 2025.
- License: MIT; the upstream copyright notice is retained in `LICENSE`.
- Original file hashes are recorded in `source.json`.

`speinet.py`, `swinir.py`, `recons_video_ori.py`, `SearchTransfer.py`, `block.py`
and the required Richardson–Lucy functions in `rcl.py` come from the upstream
`model` directory. `sharpness.py` contains the six autofocus feature functions
from `inference_SPEINet.py`. `detector.json` records the coefficients, intercept,
feature order and source hash of `LD_detector/pickle/LogisticRegression_0.5_11.pkl`.
The adapter evaluates its linear decision function without loading a pickle.

## Source adaptations

- Package-relative imports replace the upstream top-level `model` imports.
- SwinIR imports the same layers through the current `timm.layers` namespace.
- Unused plotting, training and utility imports are omitted. The Richardson–Lucy
  module contains only the two functions called by SPEINet.
- No network layers or attention computations were changed. The adapter sets
  the network device explicitly and strictly loads its state dictionary.

## Raw-video adapter policies

The adapter in the parent directory uses the upstream architecture settings:
three temporal neighbours, two reference slots, RGB in `[0,1]`, 32 features,
three residual blocks, Swin window size 5, embedding width 256, six stages of
depth 6, eight heads per stage, MLP ratio 2 and a `1conv` residual connection.

Raw video uses a bounded context window: nearest detected sharp frames before
and after the center within seven source frames. A sole reference fills both
slots; no reference fills both slots with zeros. This is a documented adaptation
of the reference script's sequence/endpoint selection. The network gates on
slot 3 and consumes slot 4 for sharp-feature transfer. Endpoints are reflected;
a single-frame clip repeats its only frame.

Spatial inference uses overlapping tiles, padding to multiples of 20 and
blending overlaps with positive Hann weights. Padding is removed from the
output. These choices preserve original dimensions and bound attention memory;
they differ from the reference script's full-frame crop/recursive-chop path.
Outputs are therefore not claimed to reproduce the paper's benchmark results.

The local cached checkpoint restored a full sample-video frame, but produced
non-finite values on other test inputs with this source revision; see the parent
`README.md` for validation status. The adapter raises an error on invalid output.
