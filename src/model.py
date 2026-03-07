import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """
    Two-tower (bi-encoder) model with:
    - User tower: user embedding → 3-layer MLP with residual + LayerNorm → L2-normalized output
    - Item tower: concat(item_emb, aisle_emb, dept_emb) → 3-layer MLP with residual + LayerNorm → L2-normalized output

    Improvements over the baseline 2-layer model:
    - Deeper 3-layer MLPs increase representational capacity
    - Residual connections prevent gradient degradation and preserve embedding identity
    - LayerNorm stabilises training with hard negatives and mixed-precision
    - GELU activation (smoother than ReLU) consistently outperforms in retrieval models
    """

    AISLE_EMB_DIM = 32
    DEPT_EMB_DIM = 16

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_aisles: int,
        num_depts: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # --- User tower (3-layer with residual) ---
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, emb_dim),
        )

        # --- Item tower (3-layer with residual) ---
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.aisle_emb = nn.Embedding(num_aisles, self.AISLE_EMB_DIM)
        self.dept_emb = nn.Embedding(num_depts, self.DEPT_EMB_DIM)

        item_input_dim = emb_dim + self.AISLE_EMB_DIM + self.DEPT_EMB_DIM
        self.item_tower = nn.Sequential(
            nn.Linear(item_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, emb_dim),
        )
        # Linear projection for item-side residual (input_dim ≠ emb_dim)
        self.item_residual_proj = nn.Linear(item_input_dim, emb_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.aisle_emb.weight, std=0.01)
        nn.init.normal_(self.dept_emb.weight, std=0.01)
        for m in list(self.user_tower) + list(self.item_tower):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.item_residual_proj.weight, nonlinearity='relu')

    def forward(self, *args, **kwargs):
        # Two-tower models don't have a single unified forward pass — the towers
        # must be called independently so item embeddings can be pre-computed and
        # cached in a FAISS index at serving time. Calling model(x) directly is
        # intentionally unsupported to make this separation explicit.
        raise NotImplementedError(
            "TwoTowerModel does not implement forward(). "
            "Use forward_user(user_idx) or forward_item(item_idx, aisle_idx, dept_idx) directly. "
            "At training time, compute logits manually: (user_emb @ item_emb.T) / temperature. "
            "At inference time, use get_user_embedding() and get_all_item_embeddings() with a FAISS index."
        )

    def forward_user(self, user_idx: torch.LongTensor) -> torch.Tensor:
        x = self.user_emb(user_idx)
        x = x + self.user_tower(x)          # residual: identity + learned transform
        return F.normalize(x, p=2, dim=-1)

    def forward_item(
        self,
        item_idx: torch.LongTensor,
        aisle_idx: torch.LongTensor,
        dept_idx: torch.LongTensor,
    ) -> torch.Tensor:
        x_cat = torch.cat([
            self.item_emb(item_idx),
            self.aisle_emb(aisle_idx),
            self.dept_emb(dept_idx),
        ], dim=-1)
        # Residual with linear projection (input_dim → emb_dim)
        x = self.item_residual_proj(x_cat) + self.item_tower(x_cat)
        return F.normalize(x, p=2, dim=-1)

    def get_user_embedding(self, user_idx: torch.LongTensor) -> torch.Tensor:
        return self.forward_user(user_idx)

    def get_item_embedding(
        self,
        item_idx: torch.LongTensor,
        aisle_idx: torch.LongTensor,
        dept_idx: torch.LongTensor,
    ) -> torch.Tensor:
        return self.forward_item(item_idx, aisle_idx, dept_idx)

    def get_all_item_embeddings(
        self,
        item_aisle: torch.LongTensor,
        item_dept: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Return normalized embeddings for all items. Shape: (num_items, emb_dim).
        Always runs on the model's device and returns a CPU tensor.
        item_aisle / item_dept may be on any device; they are moved internally.
        """
        device = next(self.parameters()).device
        item_idx = torch.arange(self.item_emb.num_embeddings, device=device)
        return self.forward_item(
            item_idx,
            item_aisle.to(device),
            item_dept.to(device),
        ).detach().cpu()