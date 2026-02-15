---
layout: post
title: "RAGアーキテクチャと堅牢性: 設計空間の全体像"
description: "Retriever-centric、Generator-centric、Hybrid、Robustness-oriented設計の比較。Production環境での堅牢性確保まで深掘り"
categories: [blog, paper, arxiv]
tags: [RAG, architecture, robustness, LLM, retrieval]
date: 2026-02-15 10:00:00 +0900
source_type: arxiv
arxiv_id: 2506.00054
source_url: https://arxiv.org/abs/2506.00054
zenn_article: ac14636a973cac
zenn_url: https://zenn.dev/0h_n0/articles/ac14636a973cac
target_audience: "修士学生レベル"
---

## 論文概要

**タイトル**: Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers

**公開日**: 2025年6月（arXiv:2506.00054）

本論文は、RAGシステムの設計空間を体系的に分類し、アーキテクチャパターン、強化技術、堅牢性の最前線を包括的に解説した2025年の最新サーベイです。Production環境での実運用を意識した設計指針を提供します。

**本論文の独自性**:
- RAGアーキテクチャを4種類に分類（Retriever-centric、Generator-centric、Hybrid、Robustness-oriented）
- 各設計パターンのトレードオフを定量評価
- 敵対的入力への堅牢性テストを体系化

## RAGアーキテクチャの4分類

### 1. Retriever-Centric Design（検索中心設計）

**設計思想**: 検索品質がRAG性能を決定するという前提のもと、検索コンポーネントを最適化します。

**主要技術**:

**Dense Retrieval（密検索）の最適化**:
```python
from sentence_transformers import SentenceTransformer
import torch.nn as nn

class OptimizedDenseRetriever(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        
        # ドメイン特化のFine-tuning
        self.projection = nn.Linear(768, 768)
        
    def encode_with_domain_adaptation(self, texts, domain='general'):
        """ドメイン適応エンコーディング"""
        base_embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        
        # ドメイン特化射影
        adapted = self.projection(base_embeddings)
        
        return adapted

# 使用例: 医療ドメインへの適応
retriever = OptimizedDenseRetriever()
medical_query = "糖尿病の治療法"
embeddings = retriever.encode_with_domain_adaptation(
    [medical_query], 
    domain='medical'
)
```

**Query Expansion（クエリ拡張）**:
```python
class QueryExpander:
    def expand_with_llm(self, query):
        """LLMでクエリを複数バリエーションに拡張"""
        expansion_prompt = f"""以下の質問を、意味を保ちながら3つの異なる表現に言い換えてください。

元の質問: {query}

バリエーション:
1."""
        
        variations = self.llm.generate(expansion_prompt).split('\n')
        return [query] + variations  # 元のクエリ + 拡張
    
    def multi_query_retrieval(self, query, top_k=10):
        """複数クエリで検索し、結果を統合"""
        expanded_queries = self.expand_with_llm(query)
        
        all_results = []
        for q in expanded_queries:
            results = self.retriever.search(q, k=top_k)
            all_results.extend(results)
        
        # 重複排除 + スコア集約
        deduplicated = self.aggregate_results(all_results)
        return deduplicated[:top_k]
```

**実測効果**（論文データ）:
- HotpotQA（多段階推論）: Baseline 48% → Query Expansion 61%（+13%）
- Natural Questions: Baseline 54% → 68%（+14%）

### 2. Generator-Centric Design（生成中心設計）

**設計思想**: LLMの生成能力を最大化し、検索結果を効果的に統合します。

**主要技術**:

**Fusion-in-Decoder（FiD）**:
検索した複数文書を個別にエンコードし、Decoderで統合する手法。

```python
class FusionInDecoder:
    def encode_passages_separately(self, query, passages):
        """各文書を独立にエンコード"""
        encoded = []
        for passage in passages:
            # クエリと文書を結合
            input_text = f"question: {query} context: {passage}"
            encoding = self.encoder(input_text)
            encoded.append(encoding)
        
        return encoded
    
    def decode_with_fusion(self, encoded_passages):
        """エンコード済み文書を統合して生成"""
        # Attentionで全文書を考慮
        decoder_output = self.decoder(
            encoder_hidden_states=torch.cat(encoded_passages, dim=1)
        )
        return decoder_output

# 利点: 各文書の情報を独立に保持しつつ、生成時に統合
```

**Chain-of-Thought RAG**:
段階的推論を組み込んだRAG。

```python
def cot_rag(query, documents):
    """Chain-of-Thought RAGの実装"""
    
    # Step 1: 思考プロセスの明示化
    cot_prompt = f"""以下の文書を参照して、段階的に考えながら質問に答えてください。

文書:
{format_documents(documents)}

質問: {query}

回答プロセス:
1. まず、文書から関連する事実を抽出します。
2. 次に、それらの事実がどのように関連しているか分析します。
3. 最後に、論理的に結論を導きます。

ステップ1（事実抽出）:"""
    
    response = llm.generate(cot_prompt, max_tokens=1000)
    return response
```

**性能向上**:
- Complex QA（QASC）: Standard RAG 42% → CoT RAG 58%（+16%）
- Reasoning Task（StrategyQA）: 51% → 69%（+18%）

### 3. Hybrid Design（ハイブリッド設計）

**設計思想**: 検索と生成の両方を同時最適化します。

**End-to-End Training**:
```python
class EndToEndRAG(nn.Module):
    def __init__(self):
        super().__init__()
        self.retriever = DenseRetriever()
        self.generator = T5ForConditionalGeneration()
        
    def forward(self, query, candidate_docs):
        # Step 1: 検索（微分可能）
        retrieval_scores = self.retriever.score(query, candidate_docs)
        top_k_indices = torch.topk(retrieval_scores, k=5).indices
        
        # Step 2: ソフト選択（微分可能）
        soft_selection = torch.softmax(retrieval_scores, dim=0)
        
        # Step 3: 文書の重み付き統合
        weighted_docs = torch.sum(
            soft_selection.unsqueeze(-1) * candidate_docs,
            dim=0
        )
        
        # Step 4: 生成
        output = self.generator(query, weighted_docs)
        
        return output

# 損失関数: 生成精度で検索も最適化
loss = -log_likelihood(output, ground_truth)
loss.backward()  # 検索器も生成器も同時に更新
```

**性能向上**:
- MS MARCO QA: Separate Training 64% → End-to-End 72%（+8%）
- TriviaQA: 68% → 75%（+7%）

### 4. Robustness-Oriented Design（堅牢性重視設計）

**設計思想**: 敵対的入力・ノイズ・分布外データへの耐性を確保します。

**Adversarial Retrieval Training**:
```python
class RobustRAG:
    def train_with_adversarial_examples(self, queries, documents):
        """敵対的サンプルでの訓練"""
        
        for query, doc, label in zip(queries, documents, labels):
            # Step 1: 通常の訓練
            loss_clean = self.compute_loss(query, doc, label)
            
            # Step 2: 敵対的摂動の生成
            # クエリにノイズを追加
            perturbed_query = self.add_adversarial_noise(query)
            
            # Step 3: 摂動データでの訓練
            loss_adv = self.compute_loss(perturbed_query, doc, label)
            
            # Step 4: 両方の損失で最適化
            total_loss = loss_clean + 0.5 * loss_adv
            total_loss.backward()
    
    def add_adversarial_noise(self, query):
        """Gradient-based adversarial attack"""
        query_embedding = self.encoder(query)
        
        # 勾配計算
        query_embedding.requires_grad = True
        loss = self.compute_loss(query_embedding, ...)
        grad = torch.autograd.grad(loss, query_embedding)[0]
        
        # FGSM攻撃
        epsilon = 0.01
        perturbed = query_embedding + epsilon * grad.sign()
        
        return perturbed
```

**Noisy Document Filtering**:
```python
def filter_noisy_documents(retrieved_docs, query, threshold=0.6):
    """ノイズ文書をフィルタリング"""
    filtered = []
    
    for doc in retrieved_docs:
        # 信頼度スコア計算
        relevance = compute_relevance(query, doc)
        factuality = check_factuality(doc)  # 外部知識ベースで検証
        coherence = measure_coherence(doc)
        
        confidence = 0.5 * relevance + 0.3 * factuality + 0.2 * coherence
        
        if confidence > threshold:
            filtered.append(doc)
    
    return filtered
```

**実測効果**（Adversarial QA）:
- Clean Data: Standard RAG 75% / Robust RAG 76%（+1%）
- Noisy Data: Standard RAG 42% / Robust RAG 68%（+26%）

## 強化技術の体系

### 1. Retrieval Enhancement（検索強化）

**Iterative Retrieval（反復検索）**:
```python
class IterativeRetriever:
    def iterative_search(self, query, max_iterations=3):
        """反復的に検索を改善"""
        current_query = query
        all_docs = []
        
        for i in range(max_iterations):
            # 検索
            docs = self.retriever.search(current_query)
            all_docs.extend(docs)
            
            # 初期回答生成
            answer = self.llm.generate(current_query, docs)
            
            # 回答の品質評価
            quality = self.evaluate_answer(answer, query)
            
            if quality > 0.9:  # 十分な品質
                break
            
            # 次のクエリを生成（回答の不足部分を補う）
            current_query = self.refine_query(query, answer, docs)
        
        return all_docs, answer
```

### 2. Context Enhancement（コンテキスト強化）

**Context Compression**:
```python
from transformers import AutoModelForSeq2SeqLM

class ContextCompressor:
    def __init__(self):
        # 抽出型要約モデル
        self.compressor = AutoModelForSeq2SeqLM.from_pretrained(
            "google/pegasus-x-base"
        )
    
    def compress_context(self, documents, target_length=2000):
        """長文書を圧縮"""
        combined = '\n\n'.join(documents)
        
        compression_prompt = f"""以下の文書を{target_length}文字に要約してください。
重要な事実のみを残してください。

文書:
{combined}

要約:"""
        
        compressed = self.compressor.generate(
            compression_prompt,
            max_length=target_length
        )
        
        return compressed
```

### 3. Decoding Enhancement（デコーディング強化）

**Constrained Decoding**:
```python
def constrained_decoding(query, documents, constraints):
    """制約付きデコーディング"""
    
    # 制約: 必ず文書から引用
    constraint_prompt = f"""以下の制約に従って回答してください:
1. 必ず提供された文書の情報のみを使用
2. 文書に含まれない情報は推測しない
3. 不明な場合は「文書に情報がありません」と回答

文書:
{documents}

質問: {query}

制約を守った回答:"""
    
    # LogitsProcessorで制約を強制
    from transformers import LogitsProcessor
    
    class CitationEnforcer(LogitsProcessor):
        def __call__(self, input_ids, scores):
            # 文書に含まれないトークンの確率を下げる
            for token_id in range(scores.shape[-1]):
                token_text = tokenizer.decode([token_id])
                if not is_in_documents(token_text, documents):
                    scores[:, token_id] -= 10.0  # ペナルティ
            return scores
    
    output = llm.generate(
        constraint_prompt,
        logits_processor=[CitationEnforcer()]
    )
    
    return output
```

## 評価フレームワーク

### Retrieval-Aware Evaluation

**Counterfactual Evaluation**:
```python
def counterfactual_evaluation(rag_system, test_cases):
    """反事実評価: 正解文書がない場合の挙動をテスト"""
    
    results = []
    for query, correct_doc, wrong_docs in test_cases:
        # Case 1: 正解文書を含む検索結果
        docs_with_correct = [correct_doc] + wrong_docs[:4]
        answer_correct = rag_system.generate(query, docs_with_correct)
        
        # Case 2: 正解文書を含まない検索結果
        answer_wrong = rag_system.generate(query, wrong_docs[:5])
        
        # 評価
        # 正解文書がある → 正しい回答
        # 正解文書がない → 「わからない」と回答すべき
        results.append({
            'with_correct': is_correct(answer_correct),
            'without_correct': says_unknown(answer_wrong),  # 期待: True
        })
    
    return results

# 理想的なRAGシステム:
# - 正解文書があれば正確に回答
# - 正解文書がなければ「わからない」と回答
```

### Robustness Testing

**Adversarial Attack Simulation**:
```python
def test_adversarial_robustness(rag_system):
    """敵対的攻撃へのロバストネステスト"""
    
    attacks = [
        # Attack 1: Keyword Poisoning
        lambda doc: doc + " irrelevant keywords: " + generate_random_keywords(),
        
        # Attack 2: Paraphrasing
        lambda doc: paraphrase_with_llm(doc),
        
        # Attack 3: Fact Injection
        lambda doc: inject_contradictory_fact(doc),
    ]
    
    robustness_scores = []
    for attack in attacks:
        correct_before = evaluate_accuracy(rag_system, clean_data)
        
        # 攻撃データ生成
        attacked_data = [attack(doc) for doc in clean_data]
        
        correct_after = evaluate_accuracy(rag_system, attacked_data)
        
        robustness = correct_after / correct_before
        robustness_scores.append(robustness)
    
    return {
        'keyword_robustness': robustness_scores[0],
        'paraphrase_robustness': robustness_scores[1],
        'fact_injection_robustness': robustness_scores[2],
    }
```

## 実運用への応用

### Production-Ready RAG Architecture

```python
class ProductionRAG:
    def __init__(self):
        # アーキテクチャ選択: Hybrid + Robustness
        self.retriever = OptimizedDenseRetriever()
        self.generator = RobustGenerator()
        
        # Monitoring
        self.metrics_collector = MetricsCollector()
        
        # Fallback
        self.fallback_retriever = BM25Retriever()  # Dense失敗時
    
    def generate_with_monitoring(self, query):
        start_time = time.time()
        
        try:
            # メイン検索
            docs = self.retriever.search(query)
            
            # ノイズフィルタリング
            filtered_docs = filter_noisy_documents(docs, query)
            
            if len(filtered_docs) < 3:
                # Fallback to BM25
                self.metrics_collector.log('fallback_triggered')
                docs = self.fallback_retriever.search(query)
            
            # 生成
            answer = self.generator.generate(query, filtered_docs)
            
            # メトリクス記録
            latency = time.time() - start_time
            self.metrics_collector.log('latency', latency)
            self.metrics_collector.log('num_docs_retrieved', len(docs))
            
            return answer
            
        except Exception as e:
            self.metrics_collector.log('error', str(e))
            # Graceful degradation
            return "申し訳ございません。一時的にサービスが利用できません。"
```

## まとめ

本論文「Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers」は、RAGの設計空間を体系的に整理した2025年の最新サーベイです。

**主要な洞察**:

1. **アーキテクチャ選択指針**:
   - プロトタイプ → Retriever-Centric（検索最適化で迅速改善）
   - Production → Hybrid（End-to-End最適化で精度最大化）
   - Mission-Critical → Robustness-Oriented（敵対的入力への耐性）

2. **強化技術のトレードオフ**:
   - Iterative Retrieval: 精度+15-20%、レイテンシ3倍
   - Context Compression: トークン削減80%、情報損失5-10%
   - Constrained Decoding: ハルシネーション-50%、創造性-30%

3. **評価の重要性**: Counterfactual Evaluation、Adversarial Testing

**実装者へのアドバイス**:
- **初期段階**: Retriever-Centric + Query Expansion
- **スケール**: Hybrid Design + End-to-End Training
- **本番運用**: Robustness-Oriented + Monitoring

関連するZenn記事「2026年版：RAG検索システムの実装と本番運用ガイド」では、これらの設計パターンの実装例を紹介しています。

## 参考文献

- "Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers." arXiv:2506.00054 (2025).
