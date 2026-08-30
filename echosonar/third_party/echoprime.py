"""Code reproduced from EchoPrime.

    Source:  https://github.com/echonet/EchoPrime  (utils/utils.py)
    © 2024 Cedars-Sinai Medical Center
    Licensed under the EchoPrime Academic Software License (academic /
    nonprofit research use only). See THIRD_PARTY_NOTICES.md.

Only the pieces required for *numerical equivalence* with the released
EchoPrime weights live here, kept byte-for-byte rather than re-implemented: any
difference in the crop, the zoom factor, or the interpolation kernel changes the
embeddings silently. Everything else in EchoSonar is our own code.

We reproduce rather than import because ``EchoPrime/utils/utils.py`` reads
``assets/per_section.json`` at module import time, so it can only be imported
with the process working directory set to the EchoPrime checkout -- a global
side effect that is unsafe in library code and in multiprocessing workers.
"""

import cv2

__all__ = ["crop_and_scale"]


def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    """Centre-crop to the target aspect ratio, zoom in 10%, then resize."""
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)
    return img
