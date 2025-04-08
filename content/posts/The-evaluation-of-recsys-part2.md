---
author: ["Abdullah Al Mamun"]
title: "The Evaluation of RecSys - Part 2"
description: " "
summary: "Sample article showcasing basic code syntax and formatting for HTML elements."
date: 2025-03-11
draft: false
tags: ["recommendation systems", "machine learning", "factorization machines", "xgboost"]
categories: ["RecSys Series"]
series: ["Themes Guide"]
ShowToc: true
TocOpen: true
math: true
social:
  fediverse_creator: "@adityatelange@mastodon.social"
---

Welcome to Part 2 of RecSys series! In [Part 1](link-to-part-1), we traced the evolution of RecSys from content-based filtering (CBF) to collaborative filtering (CF), and finally to Matrix Factorization (MF), which introduced latent factor models to tackle sparsity and scalability. However, MF’s linear assumptions and struggles with implicit feedback (e.g., clicks, views) set the stage for more advanced techniques. In this post, we dive into two pivotal methods from the 2010-2015 era: **Factorization Machines (FM)** and **Gradient Boosted Trees (XGBoost)**.

In Part 2, you’ll learn how FM generalizes MF to handle diverse data types, how XGBoost leverages decision trees for ranking, and the strengths and limitations of each.

## 1. Recap

In Part 1, we explored the foundational stages of RecSys:

- **CBF** based on features (e.g., movie genres) but struggled with diversity.
- **CF** leveraged user-item interactions, introducing neighborhood methods and latent factor models.
- **MF** modeled users and items in a latent space, predicting ratings as {{< math.inline >}} \(\hat{r}_{ui} = p_u^T q_i\){{</ math.inline >}}. However, MF assumed linear interactions and worked best with explicit feedback (ratings) only, failed to capture implicit feedback like clicks or views.

These limitations prompted the 2010-2015 era, where machine learning techniques like FM and XGBoost emerged to handle more complex patterns.

## 2. In Layman terms

Imagine you’re shopping in a store for a jacket. In Part 1, MF was like a salesman who suggested jackets based on your ratings, guessing your taste with simple categories like “likes warm jackets” or “prefers casual style.” It worked well when you rated items, but what if you never rating? or what if the assistant only knows you *clicked* on a jacket or *viewed* its picture? MF struggles here because it’s too rigid. Enter **Factorization Machines (FM)** and **XGBoost**—two smarter assistants who arrived around 2010 to fix this.

- **FM:** It's like a smart salesman who looks at everything—you clicked, the weather, and your profile (e.g., you’re a runner), mixing these clues to suggest a waterproof running jacket if it’s rainy. It’s flexible and can handle all kinds of hints, not just ratings.

- **XGBoost:** XGBoost is like a super-smart friend who learns from everyone’s shopping habits to suggest the perfect jacket. It builds a decision flowchart (Actually TREE): “If you like bright colors, and it’s winter, and you often buy on weekends, then try this red parka.” It improves its suggestions step by step.

These assistants are more flexible than MF, handling messy data and complex patterns, but they have limits, which we’ll explore as we move toward deep learning in Part 3.

## 3. Prerequisites

- **Dot product** combines two vectors to measure similarity—think of it as a handshake between features (e.g., user preferences and item traits).
- **Loss function** measures prediction errors (e.g., squared error: {{< math.inline >}} \((y - \hat{y})^2\){{</ math.inline >}}), regularization prevents overfitting, and optimization (e.g., gradient descent) minimizes the loss.
- **One-hot encoding** transforming raw data (e.g., user IDs, item categories) into usable inputs.
- From Part 1, recall MF models ratings as {{< math.inline >}} \(\hat{r}_{ui} = p_u^T q_i\){{</ math.inline >}}, where {{< math.inline >}} \(p_u\){{</ math.inline >}} and {{< math.inline >}} \(q_i\){{</ math.inline >}} are latent vectors, but it struggles with implicit feedback.

For more on these topics, check out [Linear Algebra Basics](https://www.khanacademy.org/math/linear-algebra) or [Intro to Machine Learning](https://www.coursera.org/learn/machine-learning).

## 4. Deep Dive

### 4.1 Factorization Machines (FM)

FM, introduced by Steffen Rendle in 2010, generalizes Matrix Factorization to model interactions between any features, not just users and items. It excels in sparse, high, dimensional settings like CTR prediction in online advertising, where data includes implicit feedback (clicks, views) and diverse features (user demographics, ad categories, context). FM’s ability to capture pairwise feature interactions without manual engineering made it a cornerstone for RecSys.

**How It Works:**  
FM models a prediction (e.g., click probability) as a combination of linear and pairwise feature interactions. For a feature vector {{< math.inline >}} \(\mathbf{x} \in \mathbb{R}^n\){{</ math.inline >}} (where {{< math.inline >}} \(n\){{</ math.inline >}} is the number of features), the prediction is:

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle v_i, v_j \rangle x_i x_j
$$

- {{< math.inline >}} \(w_0\){{</ math.inline >}}: Global bias.
- {{< math.inline >}} \(w_i\){{</ math.inline >}}: Weight for feature {{< math.inline >}} \(x_i\){{</ math.inline >}}.
- {{< math.inline >}} \(\langle v_i, v_j \rangle = v_i^T v_j\){{</ math.inline >}}: Dot product of latent vectors {{< math.inline >}} \(v_i, v_j \in \mathbb{R}^k\){{</ math.inline >}}, modeling the interaction between features {{< math.inline >}} \(x_i\){{</ math.inline >}} and {{< math.inline >}} \(x_j\){{</ math.inline >}}.
- {{< math.inline >}} \(k\){{</ math.inline >}}: Number of latent factors (typically 10-100).

This captures both linear effects ({{< math.inline >}} \(w_i x_i\){{</ math.inline >}}) and pairwise interactions ({{< math.inline >}} \(\langle v_i, v_j \rangle x_i x_j\){{</ math.inline >}}). For example, in CTR prediction, {{< math.inline >}} \(x_i\){{</ math.inline >}} might indicate the user, {{< math.inline >}} \(x_j\){{</ math.inline >}} the ad, and their interaction reflects compatibility.

**Connection to MF:**  
If {{< math.inline >}} \(\mathbf{x}\){{</ math.inline >}} encodes only user {{< math.inline >}} \(u\){{</ math.inline >}} and item {{< math.inline >}} \(i\){{</ math.inline >}} (e.g., {{< math.inline >}} \(x_u = 1\){{</ math.inline >}}, {{< math.inline >}} \(x_i = 1\){{</ math.inline >}}, all others 0), FM reduces to MF:
$$
\hat{y}(\mathbf{x}) = w_0 + w_u + w_i + \langle v_u, v_i \rangle
$$
Here, {{< math.inline >}} \(\langle v_u, v_i \rangle\){{</ math.inline >}} mirrors MF’s {{< math.inline >}} \(p_u^T q_i\){{</ math.inline >}}, but FM’s generality allows modeling additional features like user age or ad category.

**Loss Function:**  
FM supports regression (rating prediction) or classification (click prediction). For regression:

$$
L = \sum_{(\mathbf{x}, y) \in D} (y - \hat{y}(\mathbf{x}))^2 + \lambda (\| \mathbf{w} \|_2^2 + \| V \|_F^2)
$$

For classification (CTR):

$$
L = \sum_{(\mathbf{x}, y) \in D} \log(1 + \exp(-y \hat{y}(\mathbf{x}))) + \lambda (\| \mathbf{w} \|_2^2 + \| V \|_F^2)
$$

- {{< math.inline >}} \(D\){{</ math.inline >}}: Training data.
- {{< math.inline >}} \(y\){{</ math.inline >}}: Target (e.g., 1 for click, -1 for no click).
- {{< math.inline >}} \(\lambda\){{</ math.inline >}}: Regularization to prevent overfitting.
- {{< math.inline >}} \(V\){{</ math.inline >}}: Matrix of latent vectors {{< math.inline >}} \(v_i\){{</ math.inline >}}.

**Optimization:**  
Rendle (2010) proposes three methods:  
- **Stochastic Gradient Descent (SGD):** Updates parameters incrementally for each sample, ideal for large datasets.  
- **Alternating Least Squares (ALS):** Optimizes one parameter at a time, better for batch processing.  
- **Markov Chain Monte Carlo (MCMC):** A Bayesian approach, offering uncertainty estimates but slower.  
SGD is often preferred for scalability, with updates like:

$$
w_i \leftarrow w_i - \eta \frac{\partial L}{\partial w_i}, \quad v_i \leftarrow v_i - \eta \frac{\partial L}{\partial v_i}
$$

- {{< math.inline >}} \(\eta\){{</ math.inline >}}: Learning rate.

**Input and Output:**  
- **Input:** Sparse feature vector {{< math.inline >}} \(\mathbf{x}\){{</ math.inline >}} (e.g., one-hot encoded user ID, item ID, context).  
- **Output:** Predicted score (e.g., click probability, rating).

**Real-World Example:**  
At Meta Ads, FM might model user-ad interactions by combining user demographics (e.g., age, location), ad features (e.g., category, keyword), and context (e.g., device type), predicting the likelihood of a click to optimize ad placement.

**Takeaways:**  
- Captures pairwise feature interactions.  
- Scales well in sparse, high-dimensional data.  
- Excels in CTR prediction and implicit feedback tasks.

**Features:** Handles both explicit and implicit feedback, scales to high-dimensional sparse data, captures feature interactions without manual engineering.  

**Limitations:** Limited to pairwise interactions, misses higher-order patterns (e.g., user-item-context triplets), and assumes linear combinations, which may not capture deep non-linearities.

**Led to:** Deep learning models like DeepFM (2017), which combine FM with neural networks to learn higher-order interactions.

### 4.2 Gradient Boosted Trees (XGBoost)

XGBoost, introduced by Chen & Guestrin in 2016, leverages an ensemble of decision trees for ranking tasks in RecSys, excelling in search (e.g., Bing) and online advertising. It addresses MF and FM’s limitations in capturing non-linear patterns, using second-order optimization for efficiency and scalability.

**How It Works:**  
XGBoost builds a sequence of decision trees, each correcting the errors of the previous ones. For RecSys, it’s often used in learning-to-rank tasks (e.g., ranking search results or videos). Features include user behavior (clicks, watch time), item metadata (category, tags), and context (time, device). The prediction is:

$$
\hat{y}_i = \sum_{t=1}^T f_t(\mathbf{x}_i)
$$

- {{< math.inline >}} \(T\){{</ math.inline >}}: Number of trees.  
- {{< math.inline >}} \(f_t\){{</ math.inline >}}: Output of the {{< math.inline >}} \(t\){{</ math.inline >}}-th tree for input {{< math.inline >}} \(\mathbf{x}_i\){{</ math.inline >}}.

**Loss Function:**  
XGBoost optimizes a regularized objective:

$$
L = \sum_{i=1}^N l(y_i, \hat{y}_i) + \sum_{t=1}^T \Omega(f_t)
$$

- {{< math.inline >}} \(l(y_i, \hat{y}_i)\){{</ math.inline >}}: Loss, e.g., squared loss for regression, or pairwise ranking loss (e.g., LambdaRank).  
- {{< math.inline >}} \(\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \| \mathbf{w}_t \|_2^2\){{</ math.inline >}}: Regularization, where {{< math.inline >}} \(T\){{</ math.inline >}} is the number of leaves, and {{< math.inline >}} \(\mathbf{w}_t\){{</ math.inline >}} are leaf weights.  

For ranking, it uses pairwise loss:

$$
L = \sum_{(i,j) \in P} \log(1 + \exp(-(\hat{y}_i - \hat{y}_j)))
$$

where {{< math.inline >}} \(P\){{</ math.inline >}} is the set of relevant-irrelevant pairs. XGBoost uses a second-order approximation:

$$
L \approx \sum_{i=1}^N \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t(\mathbf{x}_i)^2 \right] + \Omega(f_t)
$$

- {{< math.inline >}} \(g_i = \frac{\partial l}{\partial \hat{y}_i^{(t-1)}}\){{</ math.inline >}}, {{< math.inline >}} \(h_i = \frac{\partial^2 l}{\partial (\hat{y}_i^{(t-1)})^2}\){{</ math.inline >}}.

This enables efficient tree construction, with features like column sampling and parallel processing for scalability.

**Input and Output:**  
- **Input:** Feature vectors (numerical or categorical, e.g., user watch time, item category).  
- **Output:** Ranking scores for items.

**Real-World Example:**  
At Bing, XGBoost ranks search results by modeling features like query relevance, user click history, and page quality, ensuring the most relevant results appear at the top.

**Key Takeaways (XGBoost in 3 Points):**  
- Captures non-linear patterns via tree ensembles.  
- Robust to missing data and interpretable (feature importance).  
- Excels in ranking tasks like search and ads.

**Features:** Captures non-linear patterns, handles mixed feature types via tree splits, robust to missing data, interpretable (feature importance scores).  
**Limitations:** Struggles with extremely high-dimensional sparse data (e.g., one-hot encoded user/item IDs), computationally expensive for large datasets, requires careful feature engineering. 

**Led to:** Neural Collaborative Filtering (NCF) and other deep learning methods that automatically learn feature representations.

## 6. Conclusion

Part 2 has taken us from MF to FM’s flexible feature interactions and XGBoost’s non-linear ranking power. FM excels in CTR prediction by modeling sparse, implicit data, while XGBoost dominates ranking tasks with its ability to capture complex patterns. However, both methods hit limits—FM’s pairwise focus and XGBoost’s reliance on feature engineering couldn’t keep up with the complexity of modern RecSys. In Part 3, we’ll explore how deep learning overcomes these limitations, tackling unstructured data like images and text with models like Neural Collaborative Filtering and DeepFM, which leverage neural networks for higher-order interactions and automated feature learning.

## 7. References

- Rendle, S. (2010). Factorization Machines. *2010 IEEE International Conference on Data Mining (ICDM)*, 995-1000. [https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf)  
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *arXiv preprint arXiv:1603.02754*. [https://arxiv.org/pdf/1603.02754](https://arxiv.org/pdf/1603.02754)  
- He, X., Zhang, H., Kan, M.-Y., & Chua, T.-S. (2017). Fast Matrix Factorization for Online Recommendation with Implicit Feedback. *Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’17)*, 549–558. [https://dl.acm.org/doi/10.1145/3459637.3482492](https://dl.acm.org/doi/10.1145/3459637.3482492)