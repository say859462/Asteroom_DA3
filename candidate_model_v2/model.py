from __future__ import annotations

import math

import torch
import torch.nn as nn


class QueryEvidenceModel(nn.Module):
    """Connectivity head over frozen joint-view visual regions."""

    is_intrinsically_symmetric = False

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_feature_layers: int = 4,
        num_views: int = 6,
        region_grid: int = 32,
        num_queries: int = 8,
        transformer_depth: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        query_pool_temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_feature_layers = int(num_feature_layers)
        self.num_views = int(num_views)
        self.region_grid = int(region_grid)
        self.num_regions = self.region_grid**2
        self.num_queries = int(num_queries)
        self.query_pool_temperature = float(query_pool_temperature)

        if self.input_dim % 2 != 0:
            raise ValueError("input_dim must contain equal local/global halves")
        if self.hidden_dim % self.num_feature_layers != 0:
            raise ValueError("hidden_dim must be divisible by num_feature_layers")
        if self.hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if self.query_pool_temperature <= 0:
            raise ValueError("query_pool_temperature must be positive")

        half_dim = self.input_dim // 2
        layer_dim = self.hidden_dim // self.num_feature_layers
        self.local_norms = nn.ModuleList(
            nn.LayerNorm(half_dim) for _ in range(self.num_feature_layers)
        )
        self.global_norms = nn.ModuleList(
            nn.LayerNorm(half_dim) for _ in range(self.num_feature_layers)
        )
        self.layer_projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.input_dim, layer_dim),
                nn.LayerNorm(layer_dim),
            )
            for _ in range(self.num_feature_layers)
        )
        self.region_norm = nn.LayerNorm(self.hidden_dim)
        self.position_projection = nn.Sequential(
            nn.Linear(8, self.hidden_dim),
            nn.GELU(),
        )
        self.token_norm = nn.LayerNorm(self.hidden_dim)

        self.evidence_queries = nn.Parameter(
            torch.empty(1, self.num_queries, self.hidden_dim)
        )
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_state_norm = nn.LayerNorm(self.hidden_dim)
        self.context_norm_a = nn.LayerNorm(self.hidden_dim)
        self.context_norm_b = nn.LayerNorm(self.hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)

        pair_dim = self.hidden_dim * 3
        self.evidence_projection = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.aggregator = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_depth,
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer(
            "position_features",
            self._build_position_features(),
            persistent=False,
        )
        nn.init.trunc_normal_(self.evidence_queries, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _build_position_features(self) -> torch.Tensor:
        denominator = max(self.region_grid - 1, 1)
        features: list[list[float]] = []
        for view_index in range(self.num_views):
            theta = view_index * (2.0 * math.pi / self.num_views)
            for region_index in range(self.num_regions):
                x = (region_index % self.region_grid) / denominator
                y = (region_index // self.region_grid) / denominator
                features.append(
                    [
                        math.sin(theta),
                        math.cos(theta),
                        x,
                        y,
                        math.sin(math.pi * x),
                        math.cos(math.pi * x),
                        math.sin(math.pi * y),
                        math.cos(math.pi * y),
                    ]
                )
        return torch.tensor(features, dtype=torch.float32)

    def _validate_regions(self, value: torch.Tensor, name: str) -> None:
        expected = (
            self.num_views,
            self.num_feature_layers,
            self.num_regions,
            self.input_dim,
        )
        if value.ndim != 5 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"Expected {name} [B,{expected[0]},{expected[1]},"
                f"{expected[2]},{expected[3]}], got {tuple(value.shape)}"
            )

    def _tokenize_regions(self, regions: torch.Tensor) -> torch.Tensor:
        half_dim = self.input_dim // 2
        projected_layers: list[torch.Tensor] = []
        for layer_index in range(self.num_feature_layers):
            layer = regions[:, :, layer_index]
            local = self.local_norms[layer_index](layer[..., :half_dim])
            global_context = self.global_norms[layer_index](layer[..., half_dim:])
            balanced = torch.cat((local, global_context), dim=-1)
            projected_layers.append(self.layer_projections[layer_index](balanced))
        tokens = self.region_norm(torch.cat(projected_layers, dim=-1))
        tokens = tokens.reshape(
            tokens.shape[0],
            self.num_views * self.num_regions,
            self.hidden_dim,
        )
        position = self.position_projection(
            self.position_features.to(device=tokens.device, dtype=tokens.dtype)
        )
        return self.token_norm(tokens + position.unsqueeze(0))

    def _pool_query_view_scores(
        self,
        attention_a: torch.Tensor,
        attention_b: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = attention_a.shape[0]
        view_attention_a = attention_a.reshape(
            batch_size, self.num_queries, self.num_views, self.num_regions
        ).sum(dim=-1)
        view_attention_b = attention_b.reshape(
            batch_size, self.num_queries, self.num_views, self.num_regions
        ).sum(dim=-1)
        query_joint = view_attention_a[:, :, :, None] * view_attention_b[:, :, None, :]
        log_joint = query_joint.float().clamp_min(1e-8).log()
        return torch.logsumexp(
            log_joint / self.query_pool_temperature,
            dim=1,
        ) * self.query_pool_temperature

    def forward_symmetric_with_debug(
        self,
        regions_a: torch.Tensor,
        regions_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_ab, debug_ab = self.forward_with_debug(regions_a, regions_b)
        logits_ba, debug_ba = self.forward_with_debug(regions_b, regions_a)
        debug = dict(debug_ab)
        debug["view_pair_scores"] = 0.5 * (
            debug_ab["view_pair_scores"]
            + debug_ba["view_pair_scores"].transpose(1, 2)
        )
        return 0.5 * (logits_ab + logits_ba), debug

    def forward(self, regions_a: torch.Tensor, regions_b: torch.Tensor) -> torch.Tensor:
        return self.forward_with_debug(regions_a, regions_b)[0]


class CandidateModelV2(QueryEvidenceModel):
    architecture_name = (
        "DA3_DINOv2_Joint12_MultiLayer_BoundedIdentityQueryEvidence_VisualOnly"
    )

    def __init__(self, *, query_identity_scale: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= query_identity_scale <= 1.0:
            raise ValueError("query_identity_scale must be in [0, 1]")
        self.query_identity_scale = float(query_identity_scale)
        self.evidence_identity_norm = nn.LayerNorm(self.hidden_dim)

    def forward_with_debug(
        self,
        regions_a: torch.Tensor,
        regions_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_regions(regions_a, "regions_a")
        self._validate_regions(regions_b, "regions_b")
        if regions_a.shape[0] != regions_b.shape[0]:
            raise ValueError("regions_a and regions_b must have the same batch size")

        tokens_a = self._tokenize_regions(regions_a)
        tokens_b = self._tokenize_regions(regions_b)
        batch_size = tokens_a.shape[0]
        queries = self.evidence_queries.expand(batch_size, -1, -1).to(
            dtype=tokens_a.dtype
        )
        context_a, attention_a = self.cross_attention(
            self.query_norm(queries),
            tokens_a,
            tokens_a,
            need_weights=True,
            average_attn_weights=False,
        )
        query_state = self.query_state_norm(
            queries + self.attention_dropout(context_a)
        )
        context_b, attention_b = self.cross_attention(
            self.query_norm(query_state),
            tokens_b,
            tokens_b,
            need_weights=True,
            average_attn_weights=False,
        )

        evidence_a = self.context_norm_a(context_a)
        evidence_b = self.context_norm_b(context_b)
        pair_features = torch.cat(
            (
                evidence_a + evidence_b,
                (evidence_a - evidence_b).abs(),
                evidence_a * evidence_b,
            ),
            dim=-1,
        )
        image_evidence = self.evidence_projection(pair_features)
        query_identity = self.query_norm(queries)
        evidence = self.evidence_identity_norm(
            image_evidence + self.query_identity_scale * query_identity
        )
        cls = self.cls_token.expand(batch_size, -1, -1).to(dtype=evidence.dtype)
        encoded = self.aggregator(torch.cat((cls, evidence), dim=1))
        logits = self.classifier(encoded[:, 0]).squeeze(-1)

        attention_a = attention_a.mean(dim=1)
        attention_b = attention_b.mean(dim=1)
        view_pair_scores = self._pool_query_view_scores(attention_a, attention_b)
        return logits, {
            "attention_a": attention_a,
            "attention_b": attention_b,
            "view_pair_scores": view_pair_scores,
            "image_evidence_tokens": image_evidence,
            "evidence_tokens": evidence,
        }
