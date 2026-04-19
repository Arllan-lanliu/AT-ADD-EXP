import torch
import torch.nn as nn


class ResNet18ForAudio(nn.Module):
    """
    Mel-spectrogram + ResNet-18 baseline CM.

    Computes a log Mel-spectrogram on-device with torchaudio (GPU-compatible),
    passes it through a ResNet-18 backbone (1-channel input), and outputs
    ``(hidden, logits)`` matching the interface expected by the rest of the
    training / inference pipeline.

    Requires ``torchvision`` to be installed::

        pip install torchvision

    Args:
        n_mels (int):      Number of Mel filter banks (default 80).
        n_fft (int):       FFT size (default 512).
        hop_length (int):  Hop size in samples (default 160, i.e. 10 ms at 16 kHz).
        sample_rate (int): Audio sample rate (default 16 000 Hz).
        num_classes (int): Number of output classes (default 2: real / fake).
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        sample_rate: int = 16_000,
        num_classes: int = 2,
    ):
        super().__init__()
        try:
            import torchaudio.transforms as T
            import torchvision.models as tv
        except ImportError as e:
            raise ImportError(
                "ResNet18ForAudio requires torchaudio and torchvision. "
                "Install them with:  pip install torchaudio torchvision"
            ) from e

        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

        backbone = tv.resnet18(weights=None)
        # Replace first conv to accept 1-channel mel-spectrogram
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        hidden_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()

        self.encoder = backbone
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, audio: torch.Tensor):
        """
        Args:
            audio: ``(B, T)`` raw waveform tensor on the model's device.

        Returns:
            Tuple ``(hidden, logits)`` where hidden is ``(B, 512)`` and
            logits is ``(B, num_classes)``.
        """
        x = self.mel(audio)            # (B, n_mels, T')
        x = self.amplitude_to_db(x)    # (B, n_mels, T')
        x = x.unsqueeze(1)             # (B, 1, n_mels, T')
        hidden = self.encoder(x)       # (B, 512)
        logits = self.classifier(hidden)
        return hidden, logits

