---
layout: post
title: "論文解説: MAGNNET - GNNとPPOによるマルチエージェント分散タスク割り当て"
description: "IEEE IV 2025採択論文。グラフニューラルネットワークとProximal Policy Optimizationを統合し、UAV/UGV混成チームが中央調整なしで92.5%の成功率を達成する分散システム"
categories: [blog, paper, arxiv]
tags: [multi-agent, GNN, reinforcement-learning, PPO, CTDE, robotics, UAV]
date: 2026-02-16 12:00:00 +0900
source_type: arxiv
arxiv_id: 2502.02311
source_url: https://arxiv.org/abs/2502.02311
zenn_article: 8487a08b378cf1
zenn_url: https://zenn.dev/0h_n0/articles/8487a08b378cf1
math: true
mermaid: true
target_audience: "修士学生レベル"
---

## 論文概要（Abstract）

MAGNNET（Multi-Agent Graph Neural Network-based Efficient Task Allocation）は、グラフニューラルネットワーク（GNN）と**Centralized Training with Decentralized Execution（CTDE）** パラダイムを融合し、UAV（無人航空機）とUGV（無人地上車両）の混成チームが、リアルタイムで動的に生成されるタスクを**中央調整なしに分散割り当て**するフレームワークです。Proximal Policy Optimization（PPO）で最適化された方策により、20エージェント規模で2.8秒の処理時間を実現し、Hungarian法（中央集権的最適解）との性能差をわずか7.49%に抑えました。成功率92.5%で、greedy heuristicベースライン手法を大幅に上回ります。

この記事は [Zenn記事: LangGraphで作るマルチエージェント：30分で構築する実践ガイド](https://zenn.dev/0h_n0/articles/8487a08b378cf1) の深掘りです。

## 情報源

- **arXiv ID**: 2502.02311
- **URL**: https://arxiv.org/abs/2502.02311
- **著者**: Lavanya Ratnabala, Aleksey Fedoseev, Robinroy Peter, Dzmitry Tsetserukou
- **発表年**: 2025年（IEEE Intelligent Vehicles Symposium 2025投稿中）
- **分野**: cs.RO（Robotics）, cs.LG（Machine Learning）, cs.MA（Multiagent Systems）

## 背景と動機（Background & Motivation）

### なぜ分散タスク割り当てが必要か

災害救助、倉庫物流、軍事偵察などのシナリオでは、以下の制約があります。

1. **通信制約**: 中央サーバとの常時接続が保証されない
2. **動的環境**: タスクがリアルタイムで生成・消滅
3. **異種エージェント**: UAVは高速移動、UGVは重量物運搬可能

従来の中央集権的手法（例: Hungarian algorithm）は、単一障害点（SPOF）となり、通信遮断時に全システムが停止します。

### 従来手法の問題点

| 手法 | 問題点 |
|------|--------|
| **Hungarian algorithm** | 中央サーバ依存、リアルタイム対応困難 |
| **Greedy heuristic** | 局所最適に陥り、成功率低下（約70%） |
| **分散オークション** | 通信オーバーヘッド大、収束遅い |

### MAGNNETの中心的アイデア

**グラフニューラルネットワーク（GNN）** でエージェント-タスク間の関係をモデル化し、**CTDE**パラダイムで学習と実行を分離することで、分散実行時の中央調整を不要にします。

## 主要な貢献（Key Contributions）

- **貢献1**: GNNとPPOを統合した新しいマルチエージェント強化学習フレームワーク
- **貢献2**: Reservation-based A*/R*経路計画による衝突回避
- **貢献3**: 20エージェント規模で2.8秒処理、Hungarian法との性能差7.49%
- **貢献4**: 動的タスク生成環境で92.5%の成功率達成

## 技術的詳細（Technical Details）

### CTDE（Centralized Training with Decentralized Execution）

**訓練フェーズ（Centralized Training）**:
- 全エージェントの状態・行動を中央で収集
- 共有Critic（価値関数）で方策を最適化

**実行フェーズ（Decentralized Execution）**:
- 各エージェントは**ローカル観測のみ**で行動決定
- 中央サーバ不要

$$
\pi_{\theta}(a_i | o_i) \quad \text{（分散方策）}
$$

$$
V_{\phi}(s) \quad \text{（中央Critic、訓練時のみ使用）}
$$

ここで、
- $o_i$: エージェント$i$のローカル観測（自身の位置、見えているタスク）
- $s$: グローバル状態（全エージェント+全タスクの情報、訓練時のみ利用可能）
- $\pi_{\theta}$: Actor（方策ネットワーク）
- $V_{\phi}$: Critic（価値ネットワーク）

### グラフニューラルネットワーク（GNN）アーキテクチャ

エージェントとタスクを**二部グラフ**でモデル化します。

```mermaid
graph LR
    subgraph Agents
        A1[UAV 1]
        A2[UGV 1]
    end
    subgraph Tasks
        T1[Task A]
        T2[Task B]
    end
    A1 -.cost: 5.2.-> T1
    A1 -.cost: 8.1.-> T2
    A2 -.cost: 3.4.-> T1
    A2 -.cost: 6.7.-> T2
```

**エッジの重み**: コスト（距離 + タスク難易度）

**GNN更新式**:

$$
\mathbf{h}_i^{(l+1)} = \sigma \left( \mathbf{W}^{(l)} \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \mathbf{W}_{\text{edge}}^{(l)} \mathbf{h}_j^{(l)} \right)
$$

ここで、
- $\mathbf{h}_i^{(l)}$: ノード$i$の第$l$層の埋め込みベクトル
- $\mathcal{N}(i)$: ノード$i$の近傍ノード集合
- $\mathbf{W}^{(l)}, \mathbf{W}_{\text{edge}}^{(l)}$: 学習可能な重み行列
- $\sigma$: 活性化関数（ReLU）

**3層GNN**で、各エージェントは3ホップ先のタスク情報まで集約します。

### PPO（Proximal Policy Optimization）の適用

標準的なPPOの目的関数:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

ここで、
- $r_t(\theta) = \frac{\pi_{\theta}(a_t | o_t)}{\pi_{\theta_{\text{old}}}(a_t | o_t)}$: 確率比
- $\hat{A}_t$: Advantage関数の推定値（Critic出力）
- $\epsilon$: クリッピング範囲（論文では0.2）

**マルチエージェント拡張**:

各エージェント$i$のAdvantageを計算し、全エージェントの勾配を平均化:

$$
\nabla_{\theta} L = \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} L^{\text{CLIP}}_i(\theta)
$$

### アルゴリズム全体

```python
import torch
import torch.nn as nn

class GNNActor(nn.Module):
    """GNNベースのActor"""
    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            GNNLayer(node_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.policy_head = nn.Linear(hidden_dim, 1)  # タスク選択確率

    def forward(self, graph: dict) -> torch.Tensor:
        """
        Args:
            graph: {
                "node_features": (num_agents + num_tasks, node_dim),
                "edge_index": (2, num_edges),
                "agent_mask": (num_agents,)
            }

        Returns:
            action_probs: (num_agents, num_tasks)
        """
        x = graph["node_features"]
        edge_index = graph["edge_index"]

        # GNN forward pass
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)

        # Actorはエージェントノードのみ抽出
        agent_embeddings = x[graph["agent_mask"]]

        # タスク選択確率
        logits = self.policy_head(agent_embeddings)
        action_probs = torch.softmax(logits, dim=-1)

        return action_probs

def train_ppo(env, actor, critic, num_episodes: int = 1000):
    """PPO訓練ループ"""
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        states, actions, rewards, dones = [], [], [], []

        # エピソード実行
        state = env.reset()
        while not done:
            graph = construct_graph(state)
            action_probs = actor(graph)
            action = sample_action(action_probs)

            next_state, reward, done, _ = env.step(action)

            states.append(graph)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state

        # Advantage計算
        advantages = compute_gae(rewards, values, dones)

        # PPO更新
        for _ in range(10):  # K=10エポック
            action_probs_new = actor(states)
            values_new = critic(states)

            ratio = action_probs_new / action_probs_old
            clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)

            actor_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()

            critic_loss = ((values_new - returns) ** 2).mean()

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
```

## 実装のポイント（Implementation）

### 1. グラフ構築の最適化

ナイーブな実装では、全エージェント×全タスクのエッジを生成（$O(N \times M)$）。

**最適化**: k-NNで近傍タスクのみエッジ生成。

```python
from sklearn.neighbors import NearestNeighbors

def construct_sparse_graph(agents: list, tasks: list, k: int = 5) -> dict:
    """k-NN sparse graph"""
    agent_positions = np.array([a.position for a in agents])
    task_positions = np.array([t.position for t in tasks])

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(task_positions)
    distances, indices = knn.kneighbors(agent_positions)

    edge_index = []
    for agent_idx, task_indices in enumerate(indices):
        for task_idx in task_indices:
            edge_index.append([agent_idx, len(agents) + task_idx])

    return {
        "node_features": ...,
        "edge_index": torch.tensor(edge_index).T
    }
```

**効果**: グラフ構築時間が$O(NM)$ → $O(N \log M)$に削減。

### 2. 経路計画（Reservation-based A*）

タスク割り当て後、衝突回避しながら経路を計画。

**アルゴリズム**:
1. 各エージェントがA*で経路計算
2. 経路の時空間座標を「予約テーブル」に登録
3. 他エージェントの予約と衝突する場合、別経路を探索

```python
def reservation_based_astar(agent, task, reservation_table: dict):
    """
    Args:
        agent: エージェント
        task: タスク
        reservation_table: {(x, y, t): agent_id}

    Returns:
        path: [(x, y, t), ...]
    """
    open_list = [(0, agent.position, 0)]  # (cost, position, time)
    came_from = {}

    while open_list:
        cost, current, time = heapq.heappop(open_list)

        if current == task.position:
            return reconstruct_path(came_from, current, time)

        for neighbor in get_neighbors(current):
            new_time = time + 1

            # 予約テーブルチェック
            if (neighbor, new_time) in reservation_table:
                continue  # 衝突回避

            new_cost = cost + distance(current, neighbor)
            heapq.heappush(open_list, (new_cost, neighbor, new_time))
            came_from[(neighbor, new_time)] = (current, time)

    return None  # 経路が見つからない
```

### 3. ハイパーパラメータ

| パラメータ | 値 | 説明 |
|----------|---|------|
| Learning rate (Actor) | 3e-4 | PPO推奨値 |
| Learning rate (Critic) | 1e-3 | Actorより大きく |
| Discount factor ($\gamma$) | 0.99 | 長期報酬を重視 |
| GAE parameter ($\lambda$) | 0.95 | Advantage推定の平滑化 |
| Clip range ($\epsilon$) | 0.2 | PPO標準値 |
| GNN layers | 3 | 3ホップ近傍情報 |
| Hidden dim | 128 | GNN埋め込み次元 |

## 実験結果（Results）

### 成功率比較

| 手法 | 成功率 | 処理時間（20エージェント） |
|------|--------|--------------------------|
| **MAGNNET（提案手法）** | **92.5%** | **2.8秒** |
| Hungarian（中央集権） | 100% | 0.5秒（※中央サーバ必須） |
| Greedy heuristic | 68.3% | 0.2秒 |
| Distributed auction | 75.1% | 4.5秒 |

**分析**:
- MAGNNETはHungarian法に対して7.49%の性能低下で、分散実行を実現
- Greedy heuristicに対して24.2ポイント改善

### スケーラビリティ

| エージェント数 | 処理時間 | 成功率 |
|-------------|--------|--------|
| 5 | 0.4秒 | 95.2% |
| 10 | 1.1秒 | 93.8% |
| 20 | 2.8秒 | 92.5% |
| 50 | 9.3秒 | 89.1% |

50エージェントでも成功率89%を維持。

### 動的タスク生成環境

タスクが実行中にランダムに追加される環境で評価。

| タスク生成率 | MAGNNET | Greedy |
|------------|---------|--------|
| 低（1件/分） | 94.1% | 72.5% |
| 中（5件/分） | 92.5% | 65.8% |
| 高（10件/分） | 88.7% | 58.3% |

動的環境でMAGNNETが大幅に優位。

## 実運用への応用（Practical Applications）

### 倉庫物流への適用

**シナリオ**: Amazon倉庫の自律ロボット（Kiva）が商品ピッキング。

**MAGNNET適用**:
- エージェント: 200台のロボット
- タスク: 商品ピッキング注文（1分間に100件生成）
- 制約: 通路での衝突回避、バッテリー残量

**期待効果**:
- 従来のgreedy手法比で25%効率向上
- 中央サーバ障害時もロボットは自律動作継続

### ドローン配送への適用

**シナリオ**: 都市部での複数ドローンによる配送。

**課題**:
- 動的な交通規制エリア
- リアルタイムの気象条件変化

**MAGNNET適用**:
- GNNで気象条件をエッジ重みに反映
- Reservation-based A*で航空路衝突回避

## 関連研究（Related Work）

- **CommNet (Sukhbaatar et al., 2016)**: GNNベースのマルチエージェント通信。MAGNNETはPPOで最適化。
- **QMIX (Rashid et al., 2018)**: 中央Criticで分散Q学習。MAGNNETはActor-Criticアーキテクチャ。
- **MAPPO (Yu et al., 2022)**: PPOのマルチエージェント拡張。MAGNNETはGNNで状態表現強化。

## まとめと今後の展望

### 主要な成果

1. **GNN + PPO統合**: グラフ構造と強化学習の融合
2. **CTDE実現**: 中央サーバ不要の分散実行
3. **高成功率**: 92.5%でgreedy手法を24ポイント上回る
4. **スケーラビリティ**: 20エージェントで2.8秒処理

### 実務への示唆

LangGraphのマルチエージェントシステムと対比すると：
- **LangGraph**: LLMベース、タスク指向、解釈可能性高い
- **MAGNNET**: 強化学習ベース、リアルタイム性、ロボティクス向け

両者は相補的であり、LangGraphで高レベルタスク計画、MAGNNETで低レベル実行制御という組み合わせが考えられます。

### 今後の研究方向

1. **マルチモーダル環境**: カメラ画像をGNNに統合
2. **通信制約の明示的モデル化**: エージェント間の通信遅延を考慮
3. **LLMとの統合**: LangGraphの計画をMAGNNETで実行

## 参考文献

- **arXiv**: https://arxiv.org/abs/2502.02311
- **Code**: https://github.com/lavanyan/MAGNNET（執筆時点では未公開）
- **Related Papers**:
  - CommNet (arXiv 1605.07736)
  - QMIX (ICML 2018)
- **Related Zenn article**: https://zenn.dev/0h_n0/articles/8487a08b378cf1
