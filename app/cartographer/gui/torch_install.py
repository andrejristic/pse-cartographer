"""
Torch-support detection + guided install (C4).

Dissecting a patient (L2-L4 / cartography) needs PyTorch + transformers. On a torch-less
machine the Patient tab shows the exact copy-paste commands and offers a button that runs
the install into THIS app's own interpreter environment and streams progress. Torch is
deliberately not baked into the .deb (huge, CPU/GPU-specific) — this is the opt-in path.
"""
from __future__ import annotations

import sys

from PySide6.QtCore import QObject, QProcess, Signal

# CPU wheel by default (the target machine has no CUDA GPU). If a CUDA GPU is present the
# operator can drop the --index-url line to get the default (GPU) build.
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def install_commands(cpu: bool = True) -> list:
    """The exact commands to enable dissection, for display and for the runner."""
    py = sys.executable
    torch_cmd = [py, "-m", "pip", "install", "torch"]
    if cpu:
        torch_cmd += ["--index-url", CPU_INDEX]
    extras = [py, "-m", "pip", "install", "transformers", "accelerate",
              "sentencepiece", "protobuf"]
    return [torch_cmd, extras]


def commands_text(cpu: bool = True) -> str:
    return "\n".join(" ".join(c) for c in install_commands(cpu))


class TorchInstaller(QObject):
    """Runs the pip installs sequentially via QProcess, streaming output."""
    line = Signal(str)
    finished = Signal(bool)

    def __init__(self, cpu: bool = True, parent=None):
        super().__init__(parent)
        self.queue = install_commands(cpu)
        self.proc = None

    def start(self):
        self._next()

    def _next(self):
        if not self.queue:
            self.finished.emit(torch_available())
            return
        cmd = self.queue.pop(0)
        self.line.emit("$ " + " ".join(cmd))
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(lambda *_: self._next())
        self.proc.start(cmd[0], cmd[1:])

    def _read(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "ignore")
        for ln in data.splitlines():
            self.line.emit(ln)
