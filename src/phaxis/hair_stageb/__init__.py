"""RHAxiscc Stage B inference runtime embedded in PHAxis."""

from .serialization import make_detection_payload

__all__ = ["StageBEnsemble", "make_detection_payload"]


def __getattr__(name: str):
    if name == "StageBEnsemble":
        from .runtime import StageBEnsemble

        return StageBEnsemble
    raise AttributeError(name)

