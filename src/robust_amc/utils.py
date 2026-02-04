import torch


def get_device(device: str = "auto") -> str:
    """Get the best available device."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA acceleration is not available")
        return "cuda"

    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise
        return "mps"

    elif device == "cpu":
        return "cpu"

    else:
        raise RuntimeError(f"Unsupported device: {device}")
