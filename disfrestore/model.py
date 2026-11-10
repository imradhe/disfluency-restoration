"""Transformer tagger over the interleaved word / gap-phone sequence (see :mod:`disfrestore.encoding`)."""
from __future__ import annotations

import torch
from torch import nn

from .encoding import FMAX, IGNORE, LABELS, PHONE_TOKENS


class Tagger(nn.Module):
    """Pretrained encoder (BERT, ELECTRA, RoBERTa, DeBERTa ...) + kind embedding + alignment-feature MLP + linear head."""

    def __init__(self, encoder_name: str, tokenizer, dropout: float = 0.1, pretrained: bool = True, config=None):
        super().__init__()
        from transformers import AutoModel

        self.encoder_name = encoder_name
        self.enc = AutoModel.from_pretrained(encoder_name) if pretrained else AutoModel.from_config(config)
        self.enc.resize_token_embeddings(len(tokenizer), mean_resizing=pretrained)
        if pretrained:
            self._init_phone_embeddings(tokenizer)
        E = self.enc.get_input_embeddings().embedding_dim
        H = self.enc.config.hidden_size
        self.kind = nn.Embedding(4, E)
        self.feat = nn.Sequential(nn.Linear(FMAX, E), nn.GELU(), nn.Linear(E, E))
        nn.init.zeros_(self.feat[2].weight)  # start as a plain pretrained encoder, learn to use the evidence
        nn.init.zeros_(self.feat[2].bias)
        nn.init.normal_(self.kind.weight, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(H, len(LABELS))

    @torch.no_grad()
    def _init_phone_embeddings(self, tokenizer):
        """New phone tokens start as the mean embedding of the word-pieces of the phone's ARPAbet name."""
        emb = self.enc.get_input_embeddings().weight
        for tokn in PHONE_TOKENS:
            name = tokn[4:-1]
            pieces = tokenizer(name, add_special_tokens=False)["input_ids"]
            tid = tokenizer.convert_tokens_to_ids(tokn)
            if pieces and tid < emb.shape[0]:
                emb[tid] = emb[pieces].mean(0)

    def forward(self, ids, attn, kind, feats):
        e = self.enc.get_input_embeddings()(ids) + self.kind(kind) + self.feat(feats)
        h = self.enc(inputs_embeds=e, attention_mask=attn).last_hidden_state
        return self.head(self.drop(h))

    def loss(self, batch, class_weight: torch.Tensor | None = None):
        logits = self(batch["ids"], batch["attn"], batch["kind"], batch["feats"])
        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1),
                                           weight=class_weight, ignore_index=IGNORE), logits

    # -- self-contained checkpoints: weights + encoder config + tokenizer (no network needed to load) --------------
    def save(self, directory, tokenizer, extra: dict | None = None) -> None:
        from pathlib import Path

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.enc.config.save_pretrained(d)
        tokenizer.save_pretrained(d)
        torch.save({"state": self.state_dict(), "encoder": self.encoder_name, "labels": LABELS, "extra": extra or {}},
                   d / "tagger.pt")

    @classmethod
    def load(cls, directory, map_location="cpu"):
        from pathlib import Path

        from transformers import AutoConfig, AutoTokenizer

        d = Path(directory)
        tok = AutoTokenizer.from_pretrained(d)
        blob = torch.load(d / "tagger.pt", map_location=map_location, weights_only=False)
        assert blob["labels"] == LABELS, "label set of the checkpoint differs from this code version"
        model = cls(blob["encoder"], tok, pretrained=False, config=AutoConfig.from_pretrained(d))
        model.load_state_dict(blob["state"])
        return model.eval(), tok, blob.get("extra", {})
