# LLM × 博物馆展览：ExhibitionBench 文献调研报告

> 作者：mengzehong | 日期：2026-04-28 | 版本：v1.0

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [任务定义：ExhibitionBench](#2-任务定义)
3. [相关工作](#3-相关工作)
4. [现有数据集](#4-现有数据集)
5. [评估指标体系](#5-评估指标体系)
6. [研究空白与课题定位](#6-研究空白)
7. [基准设计建议](#7-基准设计建议)
8. [参考文献](#8-参考文献)
9. [附录：研究路线图](#附录)

---

## 1. 背景与动机

### 1.1 博物馆 AI 的现状

全球博物馆数字化进程加速，The Metropolitan Museum of Art（Met）、Rijksmuseum、故宫博物院等机构已开放结构化 Open API，提供数十万件藏品的元数据（标题、年代、材质、文化属性、图像 URL）。然而，现有博物馆 AI 系统大多停留在：

- **单件藏品检索**：基于关键词或视觉相似度检索单件展品
- **问答系统**：针对单件藏品的 FAQ / Chatbot
- **推荐系统**：基于协同过滤的"相似藏品"推荐

这些系统均以**单个 item** 为操作粒度，忽略了博物馆展览的核心特征——**展览是一个有主题、有叙事逻辑的展品集合**。

### 1.2 LLM 的潜力与研究空白

大型语言模型（LLM）已在推荐系统、知识图谱补全、文本生成等领域展现强大能力。但在**展览策划**这一垂直领域，LLM 的行为尚未被系统研究：

- LLM 在给定主题时，会倾向于选择哪些展品？是否具备策展直觉？
- LLM 能否理解展览内部的语义关联，从而预测"缺失的那件展品"？
- LLM 的文化知识是否均衡覆盖不同文明与时代？

这些问题构成了本课题的核心动机。

### 1.3 课题价值

1. **学术价值**：首个面向展览策划的 LLM 双任务基准，填补博物馆 AI 评测空白
2. **实用价值**：辅助策展人进行展品筛选、展览补全，降低策划成本
3. **文化价值**：量化评估 LLM 对不同文化遗产的理解深度，发现文化偏见

---

## 2. 任务定义：ExhibitionBench

### 2.1 任务一：主题驱动展品选择（TES）

**Theme-based Exhibition Selection (TES)**

**形式化定义**：

给定：
- 主题查询 q（自然语言，如"古埃及神庙建筑"）
- 展品候选池 C = {e1, e2, ..., eN}（每件展品含标题、描述、文化属性、年代等元数据）

输出：
- 展品子集 S ⊆ C，|S| = k

优化目标（三元权衡）：
  S* = argmax [ α·Relevance(q,S) + β·Coherence(S) + γ·Diversity(S) ]

**任务特点**：
- 不仅要求每件展品与主题相关（单点相关性）
- 还要求展品集合内部具备叙事连贯性（集合级连贯性）
- 兼顾时代、地域、材质的多样性

### 2.2 任务二：掩码展品预测（MEIP）

**Masked Exhibition Item Prediction (MEIP)**

**形式化定义**：

给定：
- 展览 E = {e1, e2, ..., e_{n-1}, [MASK]}（已知 n-1 件展品，第 n 件被掩码）
- 候选展品池 C（包含正确答案和干扰项）

输出：
- 对候选池中每件展品的排名分数，正确展品排名越高越好

**类比关系**：
- 类比 MLM（Masked Language Modeling）但操作在 entity 粒度
- 类比知识库补全（KBC）但语境是展览叙事而非 triple
- 类比完形填空（Cloze Test）但候选项是结构化展品实体

**掩码策略**：
1. **随机掩码**：从展览中随机选一件展品掩码
2. **关键展品掩码**：掩码策展人标注的"核心展品"
3. **边缘展品掩码**：掩码与主题关联最弱的展品（测试 LLM 边界感知）

### 2.3 ExhibitionBench 框架

```
ExhibitionBench
├── Task 1: TES (Theme-based Exhibition Selection)
│   ├── Input: theme query + exhibit pool
│   ├── Output: top-k exhibit set
│   └── Metrics: NDCG@k, Precision@k, Coherence Score, Diversity Score
│
└── Task 2: MEIP (Masked Exhibition Item Prediction)
    ├── Input: partial exhibition (n-1 exhibits) + candidate pool
    ├── Output: ranked candidate list
    └── Metrics: Hits@1/5/10, MRR, NDCG@k
```

---

## 3. 相关工作

### 3.1 LLM 作为推荐系统

#### 3.1.1 零样本推荐

**LLM 零样本推荐**是近年来的热点方向。Hou et al. (2024) 提出将推荐任务转化为排序任务，利用 GPT-4 的世界知识进行零样本排序，在 MovieLens 等数据集上取得竞争性结果。关键发现：LLM 对热门 item 存在显著偏差（popularity bias），对长尾 item 理解不足。

**TaxRec**（Liu et al., 2024）引入商品分类体系（taxonomy）作为结构化提示，让 LLM 在层级类目空间中进行推荐，显著提升了多样性和可解释性。这一思路与本课题高度相关——博物馆展品天然具有分类体系（CIDOC CRM 类型层级）。

**STAR**（Wang et al., 2024, arXiv:2402.16347）研究 LLM 在顺序推荐中的能力，发现 LLM 对物品顺序和上下文窗口长度高度敏感，提出了位置感知的提示工程方法。

#### 3.1.2 LLM 推荐基准

**RecBench**（Zhu et al., 2024）是首个系统性 LLM 推荐评测基准，覆盖 6 个推荐任务，评测了 GPT-4/3.5、LLaMA 等 10+ 模型。其基准设计方法论对 ExhibitionBench 有直接参考价值。

**RecRanker**（Hou et al., 2024, arXiv:2312.16171）专注于 LLM 的排序能力评测，提出三种排序范式（pointwise、pairwise、listwise），发现 listwise 排序在 LLM 推荐中表现最优。

**与本课题的关键差异**：
- 现有推荐基准聚焦于**单 item** 推荐，不涉及集合级连贯性
- 展览策划需要**集合选择**而非单点排序
- 博物馆展品的语义空间更结构化（文化、年代、材质、风格多维度）

#### 3.1.3 集合推荐

- **Attention Over Sets**（Murphy et al., 2019）：使用 Transformer 对候选集合的集合级表示建模
- **Maximum Marginal Relevance (MMR)**（Carbonell & Goldstein, 1998）：经典的相关性-多样性权衡算法，广泛用于文档摘要和推荐多样化

### 3.2 博物馆与文化遗产 AI

#### 3.2.1 博物馆对话系统与导览

Varitimiadis et al. (2021) 构建了基于知识图谱的博物馆对话导览系统，将 CIDOC CRM 实体链接到对话系统。Padilla et al. (2019) 对大英博物馆、史密森尼等 12 家机构的数字藏品数据质量进行了系统性分析，发现元数据完整性参差不齐（年代信息缺失率高达 40%）。

#### 3.2.2 GLAM 领域多模态研究

**EUFCC-CIR**（Miróet al., 2024, arXiv:2405.03226）：面向 GLAM（Galleries, Libraries, Archives, Museums）领域的组合图像检索基准，包含 180K 个三元组，覆盖欧洲文化遗产藏品。是目前最大的文化遗产多模态检索基准。

**TimeTravel**（Shafie et al., 2024）：跨文化视觉理解基准，包含 10,250 个样本，覆盖 266 种文化，发现当前 VLM 对非西方文化的识别准确率显著低于西方文化（平均差距约 23%）。

#### 3.2.3 博物馆语义搜索

基于 CLIP 的博物馆语义搜索系统（Rust et al., 2022）在 Met 数据集上实现了零样本藏品搜索，但仅处理单件展品的相关性，不涉及展览集合的构建。

### 3.3 文化遗产知识图谱

#### 3.3.1 CIDOC CRM 标准

**CIDOC CRM**（ISO 21127:2023）是文化遗产信息的国际标准本体，由 ICOM 文档委员会维护：

- **90+ 类（Classes）**：涵盖物理对象、概念对象、事件、时间跨度、地点、人物等
- **148+ 属性（Properties）**：精确描述展品的来源、创作、材质、保管、展出等关系
- 核心类：E22 Human-Made Object（展品主类）、E12 Production（创作事件）、E39 Actor（行动者）、E53 Place（地点）、E52 Time-Span（时间范围）

#### 3.3.2 文化遗产 LOD 数据集

- **WCH-LOD**：UNESCO 世界文化遗产关联开放数据集，1,154 处遗产地，使用 CIDOC CRM 表示
- **Wikidata 博物馆数据**：约 50 万件博物馆藏品，通过 P195（收藏于）、P276（位置）等属性组织
- **LLM 三元组抽取**：Caufield et al. (2024) 用 GPT-4 从博物馆文本中抽取 CIDOC CRM 三元组，F1=0.71

### 3.4 实体掩码与知识库补全（KBC）

#### 3.4.1 知识库补全基准

**BEAR**（Razniewski et al., 2021, arXiv:2110.10611）：将 KBC 转化为自然语言填空测试 LLM。GPT-3 在 Wikidata 上的 Hits@1 约为 0.38。

**SHADOW**（Cohen et al., 2023）：研究 LLM 对知识图谱"阴影知识"（推断知识）的推理能力，与 MEIP 中预测文化语义关联的展品高度相关。

**WD-Known**（Veseli et al., 2023）：区分 LLM 直接记忆的知识和需要推理的知识，提供分层评测框架。

#### 3.4.2 实体级掩码预训练

**EntityCS**（Rohrbach et al., 2023）：提出 WEP（Whole Entity Prediction）和 PEP（Partial Entity Prediction）两种实体级掩码策略，直接启发 MEIP 的掩码设计。

**MSLM（Domain-Specific Masking, 2023）**：在领域特定语料上应用结构感知掩码，对领域关键术语给予更高的掩码概率。

### 3.5 多模态检索与跨模态理解

CLIP（Radford et al., 2021）在文化遗产领域已有广泛应用。当前 VLM 在博物馆场景面临：
1. 细粒度识别能力不足（难以区分相似朝代的陶瓷风格）
2. 文化知识不均衡（非西方文化遗产理解明显弱于西方）
3. 上下文窗口限制（同时处理 20+ 件展品面临 token 限制）
4. 幻觉问题（可能生成不存在的展品或错误的文化归属）

---

## 4. 现有数据集

### 4.1 博物馆开放数据集

| 数据集 | 规模 | 来源 | 相关性 | 备注 |
|--------|------|------|--------|------|
| Met Open Access | 480K 件藏品 | Metropolitan Museum of Art | ★★★★★ | 含真实展览记录，Open API |
| Rijksmuseum API | 800K 件藏品 | 荷兰国立博物馆 | ★★★★★ | 含详细元数据，REST API |
| 故宫博物院数字文物库 | 180K 件 | 故宫博物院 | ★★★★☆ | 中文，含高清图像 |
| Europeana | 50M+ 件 | 欧洲文化遗产联合门户 | ★★★☆☆ | 多语言，质量参差 |

### 4.2 文化遗产多模态数据集

| 数据集 | 规模 | 任务 | 相关性 | 论文 |
|--------|------|------|--------|------|
| EUFCC-CIR | 180K 三元组 | 组合图像检索 | ★★★★☆ | Miróet al., arXiv:2405.03226 |
| TimeTravel | 10,250 样本 | 跨文化视觉理解 | ★★★☆☆ | Shafie et al., 2024 |
| WikiArt | 80K 画作 | 艺术风格分类 | ★★☆☆☆ | Saleh & Elgammal, 2015 |

### 4.3 知识库与推荐基准

| 数据集 | 规模 | 任务 | 相关性 | 论文 |
|--------|------|------|--------|------|
| BEAR | 16K 查询 | LLM KBC 评测 | ★★★★☆ | Razniewski et al., arXiv:2110.10611 |
| RecBench | 6 任务 | LLM 推荐评测 | ★★★☆☆ | Zhu et al., 2024 |
| WD-Known | 30K 三元组 | 分层 KBC | ★★★☆☆ | Veseli et al., 2023 |

### 4.4 待构建：ExhibitionCorpus

```
ExhibitionCorpus
├── 来源
│   ├── Met (API): ~500 个历史展览，~15,000 展品记录
│   ├── Rijksmuseum (API): ~300 个主题展览
│   ├── 故宫博物院 (爬虫): ~200 个专题展览
│   └── 国家博物馆 (爬虫): ~150 个展览
│
├── 标注
│   ├── 展览主题标签（人工审核）
│   ├── 展品重要性评级（核心/辅助/背景）
│   └── 展品间语义关系（CIDOC CRM 属性）
│
└── 规模估计
    ├── ~1,000 个展览实例
    ├── ~30,000 件展品实体
    └── ~100,000 展品关系三元组
```

---

## 5. 评估指标体系

### 5.1 Task 1 (TES) 评估指标

#### 排名指标
- **NDCG@k**：考虑排名位置的相关性指标，k 取 5、10、20
- **Precision@k**：前 k 件中与主题相关的比例
- **MAP**（Mean Average Precision）：全排名的平均精度
- **Hit Rate@k**：正确展品出现在前 k 中的比例

#### 策展质量指标
- **Thematic Coherence Score (TCS)**：展品集合与主题的语义一致性（Sentence-BERT 余弦相似度均值）
- **Diversity Score (DS)**：展品集合的时代/地域/材质多样性（Entropy 或 Simpson 多样性指数）
- **Cultural Coverage Score (CCS)**：涵盖的文化类型数量占展品总数的比例

#### 人工评估
- **Curatorial Plausibility**：邀请策展人评分（1-5分），评估选品是否符合专业标准
- **Narrative Coherence**：展品排列是否形成连贯的叙事逻辑

### 5.2 Task 2 (MEIP) 评估指标

- **Hits@1**：正确展品排名第一的比例
- **Hits@5 / Hits@10**：正确展品出现在前 5/10 的比例
- **MRR**（Mean Reciprocal Rank）：正确展品排名的倒数均值
- **NDCG@10**：考虑排名位置的综合指标
- **Perplexity**（生成式 LLM）：LLM 对正确展品描述的困惑度

### 5.3 难度分层

参考 WD-Known 的分层评测框架，测试集分为：
1. **Easy**：展品有丰富网络记录，LLM 训练数据中可能见过
2. **Medium**：展品有基础文献记录，LLM 需要推理
3. **Hard**：稀少展品或非主流文化展品，需要强泛化能力

---

## 6. 研究空白与课题定位

通过文献调研，识别出以下 **6 个关键研究空白**：

**Gap 1：无展览级别 LLM 基准**
现有 LLM 推荐基准均以单 item 为粒度，**不存在评测 LLM 在展览集合级别理解能力的基准**。

**Gap 2：集合选择 vs. 单点排序**
现有推荐研究解决"哪件 item 更相关"，而策展需要解决"哪些 item 放在一起更连贯"。**集合级连贯性优化在 LLM 推荐中未被系统研究**。

**Gap 3：文化语义上下文缺失**
现有 KBC 基准仅评测 triple 层面的事实补全，**不评测实体在特定叙事语境（展览叙事）下的适配性**。

**Gap 4：中文博物馆数据缺失**
现有博物馆 AI 数据集以欧美机构为主，**缺乏针对中国文化遗产（故宫、国博）的 LLM 评测数据**。

**Gap 5：策展质量指标缺失**
现有推荐评测忽略了策展的专业质量维度（主题连贯性、文化覆盖度、叙事逻辑）。

**Gap 6：多模态展览理解**
现有多模态数据集聚焦单件展品，**缺乏多件展品组合场景下的多模态理解评测**。

---

## 7. 基准设计建议

### 7.1 数据管道

```
Phase 1: 数据采集
├── Met Open API 提取历史展览清单 + 展品列表
├── Rijksmuseum API 提取主题展览数据
├── 故宫数字文物库 爬虫采集中文展览数据
└── 国家博物馆官网 爬虫采集展览信息

Phase 2: 实体标准化
├── 统一展品 ID 体系
├── CIDOC CRM 类型映射（E22 Human-Made Object）
├── 多语言元数据对齐（中英文）
└── 缺失字段补全（LLM 辅助 + 人工审核）

Phase 3: 基准构建
├── TES 任务：生成主题查询 + 正样本展品集合
├── MEIP 任务：生成掩码展览 + 候选池
├── 难度分层标注
└── 质量过滤（去重、人工验证）

Phase 4: 评测框架
├── 统一 API 接口（GPT-4、LLaMA、Qwen 等）
├── 自动评测脚本（排名指标）
└── 人工评测接口（策展质量）
```

### 7.2 模型基线

| 基线 | 类型 | 描述 |
|------|------|------|
| BM25 | 稀疏检索 | 基于关键词的传统检索基线 |
| Dense Retrieval (SBERT) | 密集检索 | Sentence-BERT 向量检索 |
| CLIP | 多模态 | 视觉-文本跨模态检索 |
| GPT-4 Zero-Shot | LLM | 直接 prompting，无示例 |
| GPT-4 Few-Shot | LLM | 带示例的 prompting |
| RAG + KG | 增强 LLM | 检索增强 + CIDOC CRM 知识注入 |
| Fine-tuned LLaMA | 微调 LLM | 在 ExhibitionCorpus 上微调 |

### 7.3 技术挑战

1. **上下文长度**：展览含 20-50 件展品，每件 100-500 tokens，总长可能超出上下文窗口
2. **幻觉控制**：LLM 可能虚构不存在的展品，需要约束解码到候选池内
3. **评估主观性**：策展质量评估部分依赖专业判断，人工评估成本高
4. **数据不均衡**：西方博物馆数据远多于东方，可能导致文化偏见
5. **负样本构造**：MEIP 任务的困难负样本构造需要专业知识

---

## 8. 参考文献

### LLM 推荐系统

1. Hou et al. (2024) — "Is ChatGPT a Good Recommender? A Preliminary Study"
2. Liu et al. (2024) — "TaxRec: Taxonomy-aware LLM Recommendation"
3. Wang et al. (2024) — "STAR: Towards Holistic Evaluation of LLMs in Sequential Recommendation" — arXiv:2402.16347
4. Zhu et al. (2024) — "RecBench: A Comprehensive Benchmark for LLM-based Recommendation"
5. Hou et al. (2024) — "RecRanker: Instruction Tuning for LLM-based Recommendation Reranking" — arXiv:2312.16171

### 文化遗产与博物馆 AI

6. Miróet al. (2024) — "EUFCC-CIR: A Composed Image Retrieval Dataset for GLAM" — arXiv:2405.03226
7. Shafie et al. (2024) — "TimeTravel: A Comprehensive Benchmark for Temporal and Cultural Visual Understanding"
8. Padilla et al. (2019) — "Collections as Data: Defining a Research Problem"
9. Rust et al. (2022) — CLIP 在博物馆场景的零样本语义搜索
10. Varitimiadis et al. (2021) — "Knowledge Graph-based Cultural Heritage Chatbot"

### 知识图谱与 KBC

11. Razniewski et al. (2021) — "BEAR: A Benchmark for Entity-level Attribute Recognition" — arXiv:2110.10611
12. Veseli et al. (2023) — "WD-Known: Benchmarking LLMs on Wikidata Knowledge Completion"
13. Cohen et al. (2023) — "SHADOW: Evaluating LLM Knowledge Graph Inference Capabilities"
14. Caufield et al. (2024) — "Structured information extraction: Triplet extraction from museum texts"

### 实体掩码预训练

15. Rohrbach et al. (2023) — "EntityCS: Improving Zero-Shot Cross-Lingual Transfer with Entity-Swapped Code Switching"

### 集合推荐与多样性

16. Carbonell & Goldstein (1998) — "The Use of MMR, Diversity-Based Reranking"
17. Murphy et al. (2019) — "Janossy Pooling: Learning Deep Permutation-Invariant Functions"

### 标准与本体

18. CIDOC CRM SIG (2023) — "Definition of the CIDOC Conceptual Reference Model" — ISO 21127:2023
19. Radford et al. (2021) — "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
20. Noy & McGuinness (2001) — "Ontology Development 101"

---

## 附录：研究路线图

```
Phase 1（0-3个月）：数据基础设施
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ 开发 Met/Rijksmuseum API 爬虫
□ 构建故宫/国博中文展览数据集
□ CIDOC CRM 实体映射与标准化
□ 设计 ExhibitionCorpus 数据格式规范

Phase 2（3-6个月）：基准构建
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ 生成 TES 任务数据（主题查询 + 正样本展品集）
□ 生成 MEIP 任务数据（掩码展览 + 候选池）
□ 难度分层标注（Easy/Medium/Hard）
□ 策展专家人工验证（30% 样本）

Phase 3（6-9个月）：模型与评测
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ 实现全部 7 个基线模型
□ 开发统一评测框架（自动 + 人工指标）
□ 运行全量评测，分析 LLM 文化偏见
□ 消融实验（CIDOC CRM vs. 无结构化知识）

Phase 4（9-12个月）：论文与开源
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ 撰写 ACL/EMNLP/MM 论文
□ 开源 ExhibitionCorpus 数据集
□ 开源评测代码与基线模型
□ 发布在线评测 Leaderboard
```

---

*报告生成时间：2026-04-28 | 涵盖文献截至 2024 年底*
