from random import choices

import torch
from torch import Tensor
from typing import Optional
from torch_time_stretch import time_stretch, get_fast_stretches

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.object_dict import ObjectDict


class TimeStretch(BaseWaveformTransform):
    """
    Time stretch the signal without changing the pitch.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        min_rate: float = 0.8,
        max_rate: float = 1.25,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: int = None,
    ):
        """
        :param min_rate: Minimum rate of change of total duration of the signal. A rate below 1 means the audio is slowed down. (default 0.8)
        :param max_rate: Maximum rate of change of total duration of the signal. A rate greater than 1 means the audio is sped up. (default 1.25)
        :param n_fft:
        :param hop_length:
        :param mode: ``per_example``, ``per_channel``, or ``per_batch``. Default ``per_example``.
        :param p:
        :param p_mode:
        :param sample_rate:
        """
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
        )

        if min_rate > max_rate:
            raise ValueError("min_rate must be smaller than max_rate")

        if not sample_rate:
            raise ValueError("sample_rate is invalid")

        self._fast_stretches = get_fast_stretches(
            sample_rate,
            lambda x: x >= min_rate and x <= max_rate and x != 1,
        )
        self.n_fft = n_fft if n_fft is not None else sample_rate // 64
        self.hop_length = hop_length if hop_length is not None else self.n_fft // 32

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """
        :param samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        batch_size, num_channels, num_samples = samples.shape

        self.transform_parameters["rate"] = choices(self._fast_stretches, k=batch_size)

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        """
        :param samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        batch_size, num_channels, num_samples = samples.shape

        for i in range(batch_size):
            time_stretched_samples = time_stretch(
                samples[i][None],
                self.transform_parameters["rate"][i],
                sample_rate,
            )[0]

            # Apply zero padding if the time stretched audio is not long enough to fill the
            # whole space, or crop the time stretched audio if it ended up too long.
            padded_samples = torch.zeros(size=samples[i].shape, dtype=samples.dtype)
            window = time_stretched_samples[..., : samples[i].shape[-1]]
            actual_window_length = window.shape[
                -1
            ]  # may be smaller than samples.shape[-1]
            padded_samples[..., :actual_window_length] = window
            time_stretched_samples = padded_samples

            samples[i, ...] = time_stretched_samples

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
