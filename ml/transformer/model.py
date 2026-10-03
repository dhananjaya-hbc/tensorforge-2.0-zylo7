"""Model B: a multilingual encoder fine-tuned with four heads.

- primary   (softmax over 11 categories)
- secondary (11 independent sigmoids: "is this issue also present?")
- urgent    (sigmoid)
- language  (softmax over 6 language tags) - auxiliary, training only. The API never sends the
  language, but predicting it during training pushes the encoder to separate scripts and
  transliteration styles (Sinhala vs Singlish vs Tamil vs Tanglish).
"""

import torch
from torch import nn
from transformers import AutoModel

from ml.labels import CATEGORIES, LANGUAGES


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)


class MultiTaskClassifier(nn.Module):
    def __init__(self, encoder_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.primary = nn.Linear(h, len(CATEGORIES))
        self.secondary = nn.Linear(h, len(CATEGORIES))
        self.urgent = nn.Linear(h, 1)
        self.language = nn.Linear(h, len(LANGUAGES))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        z = self.dropout(mean_pool(out.last_hidden_state, attention_mask))
        return {
            "primary": self.primary(z),
            "secondary": self.secondary(z),
            "urgent": self.urgent(z).squeeze(-1),
            "language": self.language(z),
        }


class InferenceWrapper(nn.Module):
    """What gets exported to ONNX: probabilities only, no language head."""

    def __init__(self, model: MultiTaskClassifier):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        o = self.model(input_ids, attention_mask)
        return torch.softmax(o["primary"], -1), torch.sigmoid(o["secondary"]), torch.sigmoid(o["urgent"])
