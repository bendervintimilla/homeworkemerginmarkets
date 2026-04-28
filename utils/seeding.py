import os
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed everything we can reach so experiments are reproducible.

    TensorFlow / PyTorch are imported lazily to keep this dependency-free for
    the tabular HWs (HW1-HW3) which don't need deep learning frameworks.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf  # noqa: WPS433
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch  # noqa: WPS433
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
