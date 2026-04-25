"""Training layer: CLI and orchestration."""

__all__ = ["train_model"]


def __getattr__(name):
    # Lazy import avoids the runpy warning when the module is executed as
    # `python -m cardio_ml.training.train`.
    if name == "train_model":
        from cardio_ml.training.train import train_model

        return train_model
    raise AttributeError(name)
