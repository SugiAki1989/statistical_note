from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class ColumnConfig:
    race_id: str = "RaceId"
    date: str = "Date"
    lane: str = "Teiban"
    rank: str = "ReverseRanking"  # 1着=6, ..., 6着=1 を想定
    start: str = "StartTime_3races"
    sort_keys: tuple[str, ...] = ("Date", "Venue", "Round", "Teiban")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    early_stopping_patience: int = 20
    grad_clip_norm: float = 1.0

    tau: float = 1.0
    top3_margin_weight: float = 0.20

    lane_vocab: int = 7
    lane_emb_dim: int = 16
    hidden: int = 64
    heads: int = 4
    edge_dim: int = 6
    dropout: float = 0.10

    num_nodes_per_race: int = 6
    test_days: int = 30
    valid_days: int = 30
    fill_start_value: float = 0.25

    use_standardization: bool = True


# ============================================================
# Utility
# ============================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_columns(config_path: str | Path) -> list[str]:
    with open(config_path) as f:
        return list(yaml.safe_load(f)["features"])


# ============================================================
# Model
# ============================================================
class ImprovedRaceGATv2(nn.Module):
    """
    GNN単体で改善するための版。

    改善ポイント:
    1. 入力MLPを追加してノード表現を安定化
    2. GATv2を3層化
    3. 各層に residual を追加
    4. LayerNorm / Dropout を追加
    5. 最終 head の前にもう1段 MLP を追加
    """

    def __init__(
        self,
        cont_dim: int,
        lane_vocab: int,
        lane_emb_dim: int,
        hidden: int,
        heads: int,
        edge_dim: int,
        dropout: float,
    ):
        super().__init__()

        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")

        self.dropout = dropout
        self.lane_emb = nn.Embedding(lane_vocab, lane_emb_dim)

        in_dim = cont_dim + lane_emb_dim

        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

        self.gat1 = GATv2Conv(
            hidden,
            hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden)

        self.gat2 = GATv2Conv(
            hidden,
            hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden)

        self.gat3 = GATv2Conv(
            hidden,
            hidden // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _block(
        self,
        x: torch.Tensor,
        conv: GATv2Conv,
        norm: nn.LayerNorm,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x = conv(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = norm(x + residual)
        return x

    def forward(
        self,
        x_cont: torch.Tensor,
        lane: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        lane_vec = self.lane_emb(lane)
        x = torch.cat([x_cont, lane_vec], dim=1)
        x = self.input_mlp(x)

        x = self._block(x, self.gat1, self.norm1, edge_index, edge_attr)
        x = self._block(x, self.gat2, self.norm2, edge_index, edge_attr)
        x = self._block(x, self.gat3, self.norm3, edge_index, edge_attr)

        return self.head(x).squeeze(-1)


# ============================================================
# Loss / Metrics
# ============================================================
def listnet_kl_loss_per_graph(
    pred_score: torch.Tensor,
    true_relevance: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    # ReverseRanking は 1着=6, ..., 6着=1 の想定。
    # そのまま relevance として使う。
    p_true = F.softmax(true_relevance.float() / tau, dim=0)
    log_p_true = torch.log(p_true + 1e-12)
    log_p_pred = F.log_softmax(pred_score / tau, dim=0)
    return torch.sum(p_true * (log_p_true - log_p_pred))


def top3_margin_loss_per_graph(
    pred_score: torch.Tensor,
    true_relevance: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    3連単寄りの補助 loss。
    真の上位3艇のスコアが、下位3艇より高くなるよう促す。
    """
    top_idx = torch.argsort(true_relevance, descending=True)[:3]
    bot_idx = torch.argsort(true_relevance, descending=True)[3:]

    top_mean = pred_score[top_idx].mean()
    bot_mean = pred_score[bot_idx].mean()

    return F.relu(margin - (top_mean - bot_mean))


def combined_loss_batch(
    pred_score: torch.Tensor,
    true_relevance: torch.Tensor,
    batch_vec: torch.Tensor,
    tau: float,
    top3_margin_weight: float,
) -> torch.Tensor:
    num_graphs = int(batch_vec.max().item()) + 1
    total = pred_score.new_tensor(0.0)

    for g in range(num_graphs):
        idx = (batch_vec == g).nonzero(as_tuple=False).view(-1)
        ps = pred_score[idx]
        tr = true_relevance[idx]

        loss_listnet = listnet_kl_loss_per_graph(ps, tr, tau=tau)
        loss_top3 = top3_margin_loss_per_graph(ps, tr)
        total = total + loss_listnet + top3_margin_weight * loss_top3

    return total / max(num_graphs, 1)


@torch.no_grad()
def ndcg_at_k_per_graph(
    pred_score: torch.Tensor,
    true_relevance: torch.Tensor,
    k: int,
) -> float:
    n = pred_score.size(0)
    k = min(k, n)

    pred_order = torch.argsort(pred_score, descending=True)
    ideal_order = torch.argsort(true_relevance, descending=True)

    rel_pred = true_relevance[pred_order][:k].float()
    rel_ideal = true_relevance[ideal_order][:k].float()

    positions = torch.arange(1, k + 1, dtype=torch.float, device=pred_score.device)
    discounts = torch.log2(positions + 1.0)

    dcg = torch.sum((2.0**rel_pred - 1.0) / discounts)
    idcg = torch.sum((2.0**rel_ideal - 1.0) / discounts)

    if float(idcg.item()) <= 0.0:
        return 0.0
    return float((dcg / idcg).item())


@torch.no_grad()
def evaluate_ndcg(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    k: int,
) -> float:
    model.eval()
    total = 0.0
    races = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x_cont, batch.lane, batch.edge_index, batch.edge_attr)
        graph_ids = batch.batch
        num_graphs = int(graph_ids.max().item()) + 1

        for g in range(num_graphs):
            idx = (graph_ids == g).nonzero(as_tuple=False).view(-1)
            total += ndcg_at_k_per_graph(pred[idx], batch.y_rank[idx], k=k)
            races += 1

    return total / max(races, 1)


@torch.no_grad()
def evaluate_trifecta_hit_rate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0
    hits = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x_cont, batch.lane, batch.edge_index, batch.edge_attr)
        graph_ids = batch.batch
        num_graphs = int(graph_ids.max().item()) + 1

        for g in range(num_graphs):
            idx = (graph_ids == g).nonzero(as_tuple=False).view(-1)
            pred_order = torch.argsort(pred[idx], descending=True)
            true_order = torch.argsort(batch.y_rank[idx], descending=True)
            if torch.all(pred_order[:3] == true_order[:3]):
                hits += 1
            total += 1

    return hits / max(total, 1)


# ============================================================
# Graph building
# ============================================================
def make_complete_graph_edge_index(n: int) -> torch.Tensor:
    edges: list[list[int]] = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append([i, j])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_edge_attr(
    lane: torch.Tensor,
    start: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    edge_attr を 2次元 -> 6次元に強化。

    0: |lane_j - lane_i|
    1: start_j - start_i
    2: 1(jが内側) = 1(lane_j < lane_i)
    3: 1(jが外側) = 1(lane_j > lane_i)
    4: |start_j - start_i|
    5: 1(STが相手の方が速い) = 1(start_j < start_i)
    """
    src = edge_index[0]
    dst = edge_index[1]

    lane_src = lane[src].float()
    lane_dst = lane[dst].float()
    start_src = start[src].float()
    start_dst = start[dst].float()

    lane_dist = (lane_dst - lane_src).abs()
    start_diff = start_dst - start_src
    is_inner = (lane_dst < lane_src).float()
    is_outer = (lane_dst > lane_src).float()
    abs_start_diff = (start_dst - start_src).abs()
    is_faster_start = (start_dst < start_src).float()

    return torch.stack(
        [
            lane_dist,
            start_diff,
            is_inner,
            is_outer,
            abs_start_diff,
            is_faster_start,
        ],
        dim=1,
    )


def build_one_race_graph(
    race_df: pl.DataFrame,
    feature_cols: list[str],
    cols: ColumnConfig,
    n_nodes: int,
) -> Data:
    if race_df.height != n_nodes:
        raise ValueError(f"1 race must have {n_nodes} rows, got {race_df.height}")

    race_df = race_df.sort(cols.lane)

    x_cont = torch.tensor(race_df.select(feature_cols).to_numpy(), dtype=torch.float)
    lane = torch.tensor(race_df.get_column(cols.lane).to_list(), dtype=torch.long)
    y_rank = torch.tensor(race_df.get_column(cols.rank).to_list(), dtype=torch.long)
    start = torch.tensor(race_df.get_column(cols.start).to_list(), dtype=torch.float)

    edge_index = make_complete_graph_edge_index(n_nodes)
    edge_attr = build_edge_attr(lane, start, edge_index)

    data = Data(
        x_cont=x_cont,
        lane=lane,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_rank=y_rank,
    )
    data.num_nodes = n_nodes
    return data


def iter_group_frames(df: pl.DataFrame, group_col: str) -> Iterable[pl.DataFrame]:
    for grouped in df.group_by(group_col, maintain_order=True):
        if isinstance(grouped, tuple):
            yield grouped[1]
        else:
            yield grouped


def build_graph_list(
    df: pl.DataFrame,
    feature_cols: list[str],
    cols: ColumnConfig,
    n_nodes: int,
) -> list[Data]:
    return [
        build_one_race_graph(race_df, feature_cols, cols, n_nodes)
        for race_df in iter_group_frames(df, cols.race_id)
    ]


# ============================================================
# Data prep
# ============================================================
def load_features(
    csv_path: str | Path, cols: ColumnConfig, fill_value: float
) -> pl.DataFrame:
    df = pl.read_csv(csv_path, try_parse_dates=True).sort(list(cols.sort_keys))
    return df.with_columns(pl.col(cols.start).fill_null(fill_value).alias(cols.start))


def build_work_df(
    raw_df: pl.DataFrame,
    all_feature_cols: list[str],
    cols: ColumnConfig,
) -> tuple[pl.DataFrame, list[str]]:
    struct_cols = {cols.race_id, cols.date, cols.lane, cols.rank}
    feature_cols_model = [c for c in all_feature_cols if c not in struct_cols]

    required_cols = [
        cols.race_id,
        cols.date,
        cols.lane,
        cols.rank,
        cols.start,
    ] + feature_cols_model
    required_cols = list(dict.fromkeys(required_cols))

    return raw_df.select(required_cols), feature_cols_model


def remove_invalid_rows_and_incomplete_races(
    df: pl.DataFrame,
    cols: ColumnConfig,
    feature_cols: list[str],
    n_nodes: int,
) -> pl.DataFrame:
    required_nonnull_cols = [cols.lane, cols.rank, cols.start] + feature_cols
    work_df = df.drop_nulls(required_nonnull_cols)

    float_cols = [
        c
        for c in [cols.start] + feature_cols
        if work_df.schema[c] in (pl.Float32, pl.Float64)
    ]
    for col_name in float_cols:
        work_df = work_df.filter(~pl.col(col_name).is_nan())

    valid_race_ids = (
        work_df.group_by(cols.race_id)
        .agg(pl.len().alias("n_rows"))
        .filter(pl.col("n_rows") == n_nodes)
        .get_column(cols.race_id)
    )
    return work_df.filter(pl.col(cols.race_id).is_in(valid_race_ids))


def split_by_date(
    df: pl.DataFrame,
    cols: ColumnConfig,
    cfg: TrainConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    test_start = df.select(
        pl.col(cols.date).max() - pl.duration(days=cfg.test_days)
    ).to_series()[0]
    valid_start = test_start - dt.timedelta(days=cfg.valid_days)

    test_df = df.filter(pl.col(cols.date) >= test_start)
    valid_df = df.filter(
        (pl.col(cols.date) >= valid_start) & (pl.col(cols.date) < test_start)
    )
    train_df = df.filter(pl.col(cols.date) < valid_start)
    return train_df, valid_df, test_df


def compute_standardization_stats(
    df: pl.DataFrame, feature_cols: list[str]
) -> dict[str, float]:
    stats = df.select(
        [pl.col(c).mean().alias(f"{c}__mean") for c in feature_cols]
        + [pl.col(c).std().alias(f"{c}__std") for c in feature_cols]
    )
    return stats.to_dicts()[0]


def standardize_df(
    df: pl.DataFrame,
    feature_cols: list[str],
    stats: dict[str, float],
) -> pl.DataFrame:
    exprs = []
    for c in feature_cols:
        mean = stats[f"{c}__mean"]
        std = stats[f"{c}__std"]
        if std is None or std == 0:
            exprs.append(pl.col(c))
        else:
            exprs.append(((pl.col(c) - mean) / std).alias(c))

    other_cols = [c for c in df.columns if c not in feature_cols]
    return df.select(other_cols + exprs)


def make_dataloaders(
    work_df: pl.DataFrame,
    feature_cols_model: list[str],
    cols: ColumnConfig,
    cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, list[Data], list[Data], list[Data]]:
    cleaned_df = remove_invalid_rows_and_incomplete_races(
        df=work_df,
        cols=cols,
        feature_cols=feature_cols_model,
        n_nodes=cfg.num_nodes_per_race,
    )

    train_df, valid_df, test_df = split_by_date(cleaned_df, cols, cfg)

    if cfg.use_standardization:
        stats = compute_standardization_stats(train_df, feature_cols_model)
        train_df = standardize_df(train_df, feature_cols_model, stats)
        valid_df = standardize_df(valid_df, feature_cols_model, stats)
        test_df = standardize_df(test_df, feature_cols_model, stats)

    train_graphs = build_graph_list(
        train_df, feature_cols_model, cols, cfg.num_nodes_per_race
    )
    valid_graphs = build_graph_list(
        valid_df, feature_cols_model, cols, cfg.num_nodes_per_race
    )
    test_graphs = build_graph_list(
        test_df, feature_cols_model, cols, cfg.num_nodes_per_race
    )

    train_loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=cfg.batch_size, shuffle=False)

    return (
        train_loader,
        valid_loader,
        test_loader,
        train_graphs,
        valid_graphs,
        test_graphs,
    )


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x_cont, batch.lane, batch.edge_index, batch.edge_attr)
        loss = combined_loss_batch(
            pred_score=pred,
            true_relevance=batch.y_rank,
            batch_vec=batch.batch,
            tau=cfg.tau,
            top3_margin_weight=cfg.top3_margin_weight,
        )

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1

    return total_loss / max(batches, 1)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
) -> tuple[nn.Module, float]:
    best_valid_ndcg3 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg)
        valid_ndcg3 = evaluate_ndcg(model, valid_loader, device, k=3)
        valid_trifecta = evaluate_trifecta_hit_rate(model, valid_loader, device)

        improved = valid_ndcg3 > best_valid_ndcg3
        if improved:
            best_valid_ndcg3 = valid_ndcg3
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            mark = " *best" if improved else ""
            print(
                f"epoch={epoch:03d} "
                f"loss={train_loss:.4f} "
                f"valid_ndcg3={valid_ndcg3:.3f} "
                f"valid_trifecta={valid_trifecta:.3f}{mark}"
            )

        if patience_counter >= cfg.early_stopping_patience:
            print("Early stopping triggered")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_valid_ndcg3


@torch.no_grad()
def print_test_predictions(
    model: nn.Module,
    graphs: list[Data],
    device: torch.device,
    max_races: int = 20,
) -> None:
    model.eval()
    print("\n=== Per-race predictions (test) ===")

    for i, graph in enumerate(graphs[:max_races]):
        graph = graph.to(device)
        pred = model(graph.x_cont, graph.lane, graph.edge_index, graph.edge_attr)

        pred_order_idx = torch.argsort(pred, descending=True)
        true_order_idx = torch.argsort(graph.y_rank, descending=True)

        pred_order = (pred_order_idx + 1).tolist()
        true_order = (true_order_idx + 1).tolist()
        hit = torch.all(pred_order_idx[:3] == true_order_idx[:3]).item()

        print(
            f"Race {i}: true_order={true_order}  pred_order={pred_order}  trifecta_hit={hit}"
        )


# ============================================================
# Main
# ============================================================
def main() -> None:
    cols = ColumnConfig()
    cfg = TrainConfig()
    seed_everything(cfg.seed)

    all_feature_cols = load_feature_columns("./config.yaml")
    raw_df = load_features("./features.csv", cols, cfg.fill_start_value)
    work_df, feature_cols_model = build_work_df(raw_df, all_feature_cols, cols)

    (
        train_loader,
        valid_loader,
        test_loader,
        train_graphs,
        valid_graphs,
        test_graphs,
    ) = make_dataloaders(work_df, feature_cols_model, cols, cfg)

    if len(train_graphs) == 0:
        raise ValueError("train_graphs is empty after preprocessing")

    cont_dim = train_graphs[0].x_cont.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImprovedRaceGATv2(
        cont_dim=cont_dim,
        lane_vocab=cfg.lane_vocab,
        lane_emb_dim=cfg.lane_emb_dim,
        hidden=cfg.hidden,
        heads=cfg.heads,
        edge_dim=cfg.edge_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    model, best_valid_ndcg3 = fit_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
    )

    test_ndcg3 = evaluate_ndcg(model, test_loader, device, k=3)
    test_trifecta = evaluate_trifecta_hit_rate(model, test_loader, device)

    print("\n=== TEST RESULT ===")
    print(f"best_valid_ndcg3: {best_valid_ndcg3:.3f}")
    print(f"test_ndcg3:       {test_ndcg3:.3f}")
    print(f"test_trifecta:    {test_trifecta:.3f}")

    print_test_predictions(model, test_graphs, device=device, max_races=20)


if __name__ == "__main__":
    main()
