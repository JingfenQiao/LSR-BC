import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

class SpladeSparseEncoder(nn.Module):
    """
    SPLADE-style sparse encoder:
      logits (B,L,V) -> masked with attention and special tokens -> relu -> max over L -> log1p => (B,V)

    Output is a dense vocab-sized vector; dot product corresponds to sparse scoring.
    """

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.activation = F.relu
        self.norm = torch.log1p
    
    def tokenize(self, sentences: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
        encoding = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        special_tokens_mask = encoding["special_tokens_mask"]

        return input_ids, attention_mask, special_tokens_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, special_tokens_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # (B, L, V)

        # mask padding and special tokens
        mask = attention_mask.unsqueeze(-1) * (1 - special_tokens_mask.unsqueeze(-1))  # (B, L, 1)
        logits = logits * mask.float()  # set masked to 0

        activated = self.norm(self.activation(logits))  # (B, L, V)
        sparse_rep = torch.max(activated, dim=1).values  # (B, V)
        return sparse_rep

    def save_pretrained(self, save_directory: str) -> None:
        self.model.save_pretrained(save_directory)

def freeze_module(m: nn.Module) -> None:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
