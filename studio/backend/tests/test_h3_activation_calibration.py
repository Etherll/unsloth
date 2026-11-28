# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""H3 phase need covers every measured peak without over-charging (each 0.48 GB extra streams one block)."""

from __future__ import annotations

import pytest

from core.inference import video_minimax_h3_residency as R

# (width, height, frames): (denoise peak reserved GB, VAE decode peak reserved GB), the worst fresh process.
MEASURED = {
    (864, 480, 124): (2.161, 6.619),
    (960, 544, 124): (2.537, 7.181),
    (864, 480, 243): (3.425, 7.531),
    (1344, 768, 124): (4.743, 8.799),
    (1344, 768, 192): (7.316, 11.347),
}


def _denoise(w, h, f):
    vol = w * h * f / 1e6
    return R.H3_STREAM_WINDOW_GB + R.H3_ACTIVATION_GB_PER_MPIXEL_FRAME * vol * (
        1 + R.H3_FRAGMENTATION_FRACTION
    )


@pytest.mark.parametrize("shape", sorted(MEASURED))
def test_the_phase_need_covers_every_measured_phase(shape):
    denoise, decode = MEASURED[shape]
    need = R.h3_phase_need_gb(*shape, te_streamed_gb = 3.0)
    assert need >= denoise + R.H3_PHASE_OVERHEAD_GB
    assert need >= decode + R.H3_PHASE_OVERHEAD_GB


@pytest.mark.parametrize("shape", sorted(MEASURED))
def test_the_denoise_charge_stays_near_the_measured_peak(shape):
    measured = MEASURED[shape][0]
    charged = _denoise(*shape) - R.H3_STREAM_WINDOW_GB
    assert measured <= charged <= 2.0 * measured


def test_a_default_5s_1344x768_clip_leaves_most_of_a_32gb_card_to_resident_blocks():
    assert R.h3_phase_need_gb(1344, 768, 192, te_streamed_gb = 3.0) < 15.0
