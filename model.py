import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """
    Simple two-tower (bi-encoder) model with user and item embeddings.

    - Embeddings are normalized before similarity so dot-product ~= cosine similarity.
    - Normalization stabilizes training and makes inner product comparable across vectors.
    """

    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward_user(self, user_idx: torch.LongTensor) -> torch.Tensor:
        x = self.user_emb(user_idx)
        x = F.normalize(x, p=2, dim=-1)  # normalize so dot-product ~ cosine
        return x

    def forward_item(self, item_idx: torch.LongTensor) -> torch.Tensor:
        x = self.item_emb(item_idx)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def get_all_item_embeddings(self) -> torch.Tensor:
        # Return normalized item embeddings (num_items, emb_dim)
        emb = self.item_emb.weight
        emb = F.normalize(emb, p=2, dim=-1)
        return emb.detach().cpu()

    def get_user_embedding(self, user_idx: torch.LongTensor) -> torch.Tensor:
        return self.forward_user(user_idx)

    def get_item_embedding(self, item_idx: torch.LongTensor) -> torch.Tensor:
        return self.forward_item(item_idx)
