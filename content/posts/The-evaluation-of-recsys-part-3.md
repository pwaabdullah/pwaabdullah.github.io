---
title: "The Evaluation of RecSys - Part 3"
date: 2025-03-12
draft: false
tags: ["recommendation systems", "deep learning", "neural collaborative filtering", "wide and deep", "deepfm", "din", "dlrm", "multi-task learning"]
categories: ["RecSys Series"]
---

# The Evaluation of Recommendation Systems - Part 3

## Context: Why This Post Matters, Who It’s For, and What You’ll Learn

Welcome to Part 3 of our four-part series on evaluating recommendation systems (RecSys)! In the previous installments, we laid the groundwork: Part 1 introduced foundational techniques like collaborative filtering (CF) and Matrix Factorization (MF), which excelled at capturing user-item interactions but assumed linearity, missing complex patterns. Part 2 explored Factorization Machines (FM) and XGBoost, which tackled sparse data and non-linear ranking but fell short on higher-order interactions and sequential behaviors. By 2016, these limitations spurred a seismic shift toward deep neural networks (DNNs), which transformed RecSys by learning intricate feature interactions, automating feature engineering, and addressing diverse tasks like sequential recommendations and multi-task optimization. This post traces that evolution from 2016 to 2023, diving into Neural Collaborative Filtering (NCF), Wide & Deep Learning, DeepFM, Deep Interest Network (DIN), Deep Learning Recommendation Model (DLRM), and Adaptive Task-to-Task Fusion (AdaTT). It’s tailored for data scientists, ML engineers, and tech professionals—particularly those designing large-scale RecSys in domains like e-commerce, streaming, and advertising—who need a deep, technical understanding of these advancements.

## Recap: Where We Left Off

In Part 2, we saw how FM extended MF by modeling pairwise feature interactions, making it a powerhouse for sparse settings like click-through rate (CTR) prediction. Its prediction function, \( \hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle v_i, v_j \rangle x_i x_j \), captured second-order relationships efficiently but couldn’t handle higher-order interactions or non-linear patterns beyond its linear assumptions. XGBoost, meanwhile, leveraged tree ensembles to rank items based on non-linear feature combinations, shining in tasks like top-N recommendations. Yet, it struggled with high-dimensional sparse data and required extensive manual feature engineering, limiting its scalability. These gaps—missing deep non-linearities, higher-order interactions, and sequential modeling—paved the way for DNNs, which, starting in 2016, redefined RecSys by learning complex patterns directly from raw data.

## The Big Picture: The Deep Learning Revolution in RecSys

Picture a recommendation system as a guide helping you navigate a vast library. In Part 2, our guide used simple rules: FM paired clues like your reading history with book traits, while XGBoost ranked options by studying everyone’s preferences. But what if your interests shift over time (say, from mysteries to sci-fi), or the guide needs to predict both what you’ll read and whether you’ll buy it? These earlier methods faltered. DNNs emerged as a smarter guide, capable of deciphering intricate patterns, tracking sequential behaviors, and juggling multiple goals. From 2016’s Wide & Deep to 2023’s AdaTT, this era saw RecSys evolve to handle complex user behaviors with unprecedented accuracy, shaping modern systems in companies like Google, Alibaba, and Facebook.

## Deep Dive: The Evolution of DNNs in RecSys

Let’s explore this journey, starting with Neural Collaborative Filtering, which kicked off the DNN era by rethinking how we model user-item interactions.

### Neural Collaborative Filtering (NCF, 2017)

Traditional MF, a staple from Part 1, predicts user-item interactions via a dot product: \( \hat{r}_{ui} = p_u^T q_i \), where \( p_u \) and \( q_i \) are latent vectors for user \( u \) and item \( i \). This worked well for explicit ratings but assumed linearity, missing non-linear patterns in implicit feedback like clicks or views. In 2017, He et al. proposed Neural Collaborative Filtering (NCF) to overcome this, replacing the dot product with a neural network to capture complex, non-linear relationships. The motivation was clear: real-world preferences aren’t linear—liking sci-fi movies doesn’t linearly predict liking sci-fi books—and DNNs, fresh from successes in vision and NLP, offered a way to model these nuances.

NCF’s architecture comes in three flavors. First, the inputs are simple: one-hot encoded user ID \( \mathbf{u} \) and item ID \( \mathbf{i} \), mapped to dense embeddings \( \mathbf{p}_u \) and \( \mathbf{q}_i \in \mathbb{R}^{32} \) via lookup tables. The Generalized Matrix Factorization (GMF) variant mimics MF but with a neural twist: it computes an element-wise product \( \mathbf{p}_u \odot \mathbf{q}_i \), feeds it through a linear layer with weights \( \mathbf{w} \), and applies a sigmoid activation to output a probability: \( \hat{y}_{ui} = \sigma(\mathbf{w}^T (\mathbf{p}_u \odot \mathbf{q}_i)) \). This retains MF’s linear interaction but learns the weighting neurally. The Multi-Layer Perceptron (MLP) variant takes a different tack, concatenating the embeddings into \( [\mathbf{p}_u, \mathbf{q}_i] \) and passing them through three fully connected layers (e.g., 256, 128, 64 neurons) with ReLU activations: \( \mathbf{z}_1 = \text{ReLU}(\mathbf{W}_1 [\mathbf{p}_u, \mathbf{q}_i] + \mathbf{b}_1) \), followed by more layers, ending in a prediction layer. This captures non-linear interactions unavailable to MF. Finally, Neural Matrix Factorization (NeuMF) combines both, concatenating GMF’s and MLP’s penultimate outputs and applying a final linear layer: \( \hat{y}_{ui} = \sigma(\mathbf{w}^T [\mathbf{z}_{\text{GMF}}, \mathbf{z}_{\text{MLP}}]) \). This hybrid leverages both linear and non-linear modeling.

For implicit feedback (e.g., clicks), NCF uses binary cross-entropy as its loss: \( L = -\sum_{(u,i) \in D} [y_{ui} \log(\hat{y}_{ui}) + (1-y_{ui}) \log(1-\hat{y}_{ui})] \), where \( y_{ui} = 1 \) for observed interactions and 0 otherwise. Since unobserved pairs vastly outnumber observed ones, negative sampling (e.g., 4 negatives per positive) keeps training feasible. The optimizer is Adam, with a learning rate of 0.001, balancing speed and stability. On the MovieLens 1M dataset, NeuMF achieved a Hit Ratio@10 of 0.71, beating MF’s 0.67 by 6%, thanks to its ability to model non-linear patterns. Compared to MLP alone (0.69), NeuMF’s fusion of GMF’s linearity and MLP’s depth proved superior. The special change—swapping a dot product for a neural network—unlocked this flexibility, though NCF ignores auxiliary features like user demographics and can’t model sequential behaviors.

### Wide & Deep Learning (2016)

NCF’s focus on user-item pairs left out contextual features and struggled with generalization in sparse, diverse settings. Enter Wide & Deep Learning, proposed by Cheng et al. at Google in 2016, designed for app recommendations on Google Play. The problem was twofold: linear models like logistic regression memorized specific patterns (e.g., “user installed app A”) but couldn’t generalize to unseen data, while DNNs generalized well but missed rare, critical interactions. Wide & Deep combined a linear “wide” model for memorization with a DNN “deep” model for generalization, aiming to balance both.

The architecture starts with inputs: sparse features (e.g., user ID, app ID) mapped to embeddings (e.g., 32 dimensions) and dense features (e.g., user age) fed raw. The wide component is a linear model: \( \mathbf{y}_{\text{wide}} = \mathbf{w}^T \mathbf{x} + b \), where \( \mathbf{x} \) includes raw features and hand-crafted cross-features (e.g., “user installed app A AND app B”), capturing low-order interactions. Designing these cross-features required domain expertise, a key modification over pure DNNs. The deep component is an MLP with three hidden layers (1024, 512, 256 neurons) using ReLU: \( \mathbf{z}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{e} + \mathbf{b}_1) \), where \( \mathbf{e} \) concatenates embeddings and dense inputs, learning higher-order interactions. The outputs merge via a weighted sum: \( \hat{y} = \sigma(\mathbf{w}_{\text{wide}}^T \mathbf{y}_{\text{wide}} + \mathbf{w}_{\text{deep}}^T \mathbf{z}_{\text{deep}} + b) \), yielding a click probability.

The loss is logistic (binary cross-entropy), optimized differently per component: FTRL with L1 regularization for the wide part (encouraging sparsity) and AdaGrad for the deep part (adapting to dense gradients). On Google Play, Wide & Deep boosted app installations by 3.9% over a wide-only model and 1% over a deep-only model, proving the hybrid’s value. Unlike NCF, it leverages auxiliary features, but the manual engineering of cross-features limits scalability, and it doesn’t address sequential data or higher-order interactions beyond the wide part.

### DeepFM (2017)

Wide & Deep’s reliance on manual feature engineering was a bottleneck, especially for large-scale systems with thousands of features. In 2017, Guo et al. introduced DeepFM, targeting CTR prediction in online advertising (e.g., Criteo dataset), by combining Factorization Machines (FM) with a DNN to automate feature interactions. FM’s strength was modeling pairwise interactions efficiently, but it missed higher-order patterns; DeepFM extended it to capture both low- and high-order interactions without human intervention.

DeepFM’s inputs are sparse features (e.g., user ID, ad ID) mapped to embeddings \( \mathbf{v}_i \in \mathbb{R}^{10} \). The FM component computes: \( \mathbf{y}_{\text{FM}} = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j \), capturing second-order interactions via dot products. The deep component, an MLP with three 200-neuron layers and ReLU, takes the same embeddings: \( \mathbf{z}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{v} + \mathbf{b}_1) \), learning higher-order interactions. Sharing embeddings between FM and DNN ensures consistency and efficiency—a key design choice. The final output combines both: \( \hat{y} = \sigma(\mathbf{y}_{\text{FM}} + \mathbf{w}_{\text{deep}}^T \mathbf{z}_{\text{deep}}) \).

The loss is binary cross-entropy, optimized with Adam (learning rate 0.001). On Criteo, DeepFM hit an AUC of 0.801, edging out Wide & Deep (0.799) and FM (0.785), as it automated feature engineering while retaining FM’s strengths. This modification—replacing Wide & Deep’s manual cross-features with FM—made it scalable, though it still overlooks sequential user behaviors critical for dynamic settings like e-commerce.

### Deep Interest Network (DIN, 2017)

DeepFM’s static modeling couldn’t capture how user interests evolve, say, during an e-commerce browsing session. Zhou et al. at Alibaba introduced the Deep Interest Network (DIN) in 2017 to address this, using an attention mechanism to weigh historical behaviors based on their relevance to a candidate item. Proposed for ad recommendations, DIN recognized that not all past interactions (e.g., clicked items) equally inform the next click, necessitating a dynamic approach.

DIN’s inputs include a user’s behavior sequence \( S_u = \{v_1, v_2, \ldots, v_T\} \) (e.g., clicked items), a candidate ad \( v_a \), and context features, all mapped to embeddings. The core innovation is the attention mechanism: for each historical item \( v_i \), it computes a weight \( \alpha_i = f(\mathbf{v}_i, \mathbf{v}_a) \) using a small MLP: \( f(\mathbf{v}_i, \mathbf{v}_a) = \text{ReLU}(\mathbf{W} [\mathbf{v}_i, \mathbf{v}_a, \mathbf{v}_i \odot \mathbf{v}_a] + \mathbf{b}) \). This weights items by relevance, forming a user interest vector \( \mathbf{s}_u = \sum_{i=1}^T \alpha_i \mathbf{v}_i \). This vector, the candidate embedding, and context embeddings feed into an MLP with three layers (200, 80, 2 neurons) and ReLU, ending in a sigmoid output: \( \hat{y} = \sigma(\mathbf{w}^T \mathbf{z}_{\text{deep}}) \).

The loss is binary cross-entropy, optimized with Adam (learning rate 0.001). On Alibaba’s dataset, DIN’s AUC of 0.82 beat DeepFM’s 0.80 by 2%, highlighting attention’s power in sequential modeling. Unlike DeepFM, DIN adapts to temporal dynamics, but it’s tailored to single-task CTR prediction, not multi-objective scenarios.

### Deep Learning Recommendation Model (DLRM, 2019)

DIN’s single-tower design wasn’t built for the massive scale and diverse features of systems like Facebook’s ad platform. In 2019, Naumov et al. proposed DLRM, a multi-tower architecture for CTR prediction, explicitly modeling pairwise interactions for scalability and interpretability. The need arose from handling billions of sparse features (e.g., ad IDs) alongside dense ones (e.g., user stats), where implicit modeling slowed training.

DLRM’s inputs split into dense features (e.g., user age) and sparse features (e.g., user ID), mapped to embeddings. The dense tower is an MLP with three layers (512, 256, 128 neurons) and ReLU, processing continuous inputs. The sparse tower computes pairwise dot products between embeddings: \( \mathbf{z}_{ij} = \mathbf{v}_i^T \mathbf{v}_j \), forming an interaction vector. These outputs concatenate with the dense tower’s result, feeding a top MLP (128, 1 neurons) with ReLU and sigmoid: \( \hat{y} = \sigma(\mathbf{w}^T \mathbf{z}) \).

The loss is binary cross-entropy, optimized with Adam or SGD. On Criteo, DLRM’s AUC of 0.802 slightly topped DeepFM’s 0.801, with better scalability from its parallel towers—a key change over single-tower designs. However, it focuses on single-task CTR, not multi-task or sequential needs.

### Adaptive Task-to-Task Fusion (AdaTT, 2023)

DLRM’s single-task focus couldn’t handle multi-objective RecSys, like predicting CTR and conversion rate (CVR) together. Multi-Task Learning (MTL) emerged to share representations across tasks, but task conflicts often hurt performance. Yang et al.’s AdaTT, introduced in 2023 at Kuaishou, tackled this with an adaptive task-to-task fusion network, dynamically balancing task interactions.

AdaTT’s inputs—shared features (e.g., user ID, item ID)—map to embeddings feeding a shared bottom MLP: \( \mathbf{z}_{\text{shared}} = \text{ReLU}(\mathbf{W} \mathbf{e} + \mathbf{b}) \). Task-specific towers (e.g., CTR, CVR) process this into \( \mathbf{z}_t = \text{MLP}_t(\mathbf{z}_{\text{shared}}) \). The innovation is an attention-based fusion: \( \mathbf{z}_t' = \sum_{s \neq t} \alpha_{ts} \mathbf{z}_s \), where \( \alpha_{ts} \) weights contributions from other tasks, computed via a task-to-task attention MLP. Outputs are per-task sigmoids: \( \hat{y}_t = \sigma(\mathbf{w}_t^T \mathbf{z}_t') \).

The loss is a weighted sum: \( L = \sum_t \lambda_t L_t \) (binary cross-entropy per task), optimized with Adam. On Kuaishou’s dataset, AdaTT lifted CTR AUC by 1.5% and CVR AUC by 2% over single-task models, thanks to its adaptive fusion—a leap over static MTL. Its complexity, though, demands careful tuning.

## Conclusion: What’s Next

From NCF’s non-linear leap to AdaTT’s multi-task finesse, DNNs have reshaped RecSys, each model solving a prior limitation: NCF broke linearity, Wide & Deep merged memorization and generalization, DeepFM automated engineering, DIN added sequence, DLRM scaled up, and AdaTT tackled multiple goals. Part 4 will explore graph neural networks and transformers, pushing RecSys further into complex, real-time domains.

## References

- He, X., et al. (2017). Neural Collaborative Filtering. *arXiv:1708.05031*.
- Cheng, H.-T., et al. (2016). Wide & Deep Learning for Recommender Systems. *arXiv:1606.07792*.
- Guo, H., et al. (2017). DeepFM: A Factorization-Machine Based Neural Network for CTR Prediction. *arXiv:1703.04247*.
- Zhou, G., et al. (2017). Deep Interest Network for Click-Through Rate Prediction. *arXiv:1706.06978*.
- Naumov, M., et al. (2019). Deep Learning Recommendation Model for Personalization and Recommendation Systems. *arXiv:1906.00091*.
- Yang, S., et al. (2023). AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations. *arXiv:2304.04959*.