import os

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from bertalign.utils import yield_overlaps

class Encoder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        max_seq = int(os.environ.get("HVB_EMBED_MAX_SEQ", "256"))
        self.model.max_seq_length = max_seq
        if os.environ.get("HVB_EMBED_FP16", "") and torch.cuda.is_available():
            self.model = self.model.half()
        self._pool = None

    def _multi_gpu_devices(self):
        if not torch.cuda.is_available():
            return []
        n = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n)] if n > 1 else []

    def transform(self, sents, num_overlaps):
        overlaps = [line for line in yield_overlaps(sents, num_overlaps)]

        devices = self._multi_gpu_devices()
        if devices:
            if self._pool is None:
                self._pool = self.model.start_multi_process_pool(target_devices=devices)
            batch = int(os.environ.get("HVB_EMBED_BATCH", "64"))
            sent_vecs = self.model.encode_multi_process(
                overlaps, self._pool, batch_size=batch
            )
        else:
            batch = int(os.environ.get("HVB_EMBED_BATCH", "32"))
            sent_vecs = self.model.encode(overlaps, batch_size=batch, show_progress_bar=True)

        sent_vecs = np.ascontiguousarray(sent_vecs, dtype=np.float32)
        embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
        sent_vecs.resize(num_overlaps, len(sents), embedding_dim)

        len_vecs = [len(line.encode("utf-8")) for line in overlaps]
        len_vecs = np.array(len_vecs)
        len_vecs.resize(num_overlaps, len(sents))

        return sent_vecs, len_vecs

    def close(self):
        if self._pool is not None:
            SentenceTransformer.stop_multi_process_pool(self._pool)
            self._pool = None
