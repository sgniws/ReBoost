# Method

## 3. Method

In this section, we first present the problem definition for multimodal medical image segmentation (Section 3.1). Subsequently, we introduce the proposed Sustained Boosting Framework, which comprises modality-specific feature extractors (Section 3.2), dynamically generated boosting heads (Section 3.3), and an adaptive classifier assignment mechanism (Section 3.4). Finally, we elaborate on the sustained boosting objective function designed for the joint optimization of all boosting heads (Section 3.5).

### 3.1 Preliminaries

Given a multimodal dataset $\mathcal{D} = \{(\mathbf{X}_i, \mathbf{Y}_i)\}_{i=1}^N$ with $N$ samples. For each sample, the input $\mathbf{X}_i = \{\mathbf{x}_i^{(m)}\}_{m=1}^M$ consists of volumetric data from $M$ modalities, where $\mathbf{x}_i^{(m)} \in \mathbb{R}^{1 \times D \times H \times W}$. The corresponding annotation $\mathbf{Y}_i \in \{0, 1\}^{K \times D \times H \times W}$ represents binary segmentation masks for $K$ classes. Our goal is to learn a mapping function $\mathcal{F}: \mathbf{X} \to \hat{\mathbf{Y}}$ that maximizes segmentation performance by adaptively boosting the representational capacity of weak modalities.

### 3.2 Modality-Specific Feature Extraction

To address inter-modality heterogeneity, we adopt an independent multi-stream architecture. Each modality $m$ is equipped with an independent feature extractor $\phi^{(m)}(\cdot; \theta^{(m)})$ (typically instantiated as a U-Net backbone). For an input $\mathbf{x}_i^{(m)}$, the extractor generates a high-dimensional feature representation $\mathbf{F}_i^{(m)}$:

$$
\mathbf{F}_i^{(m)} = \phi^{(m)}(\mathbf{x}_i^{(m)}; \theta^{(m)}) \in \mathbb{R}^{C_f \times D \times H \times W}
$$

where $C_f$ denotes the feature channel dimension. Unlike conventional methods, $\phi^{(m)}$ does not directly output segmentation results but serves as a general feature encoder, providing rich semantic features for the subsequent dynamic boosting heads.

### 3.3 Dynamic Boosting Head Generation

To mitigate the dynamic imbalance in learning capabilities across modalities during training, we design a Configurable Segmentation Head. For modality $m$, we dynamically maintain a set of segmentation heads $\mathcal{H}^{(m)} = \{h_t^{(m)}\}_{t=1}^{n_m}$ during training, where $n_m$ represents the current number of heads for that modality.

**Head Architecture.** Each segmentation head $h_t^{(m)}$ consists of two components: a private projection module $\psi_t^{(m)}$ and a shared semantic mapping module $\omega$.
1.  **Private Projection**: Designed to extract residual features specific to the head from the shared features $\mathbf{F}_i^{(m)}$.
    $$
    \mathbf{H}_{i,t}^{(m)} = \sigma(\mathcal{T}_t^{(m)}(\mathbf{F}_i^{(m)}))
    $$
    where $\mathcal{T}_t^{(m)}$ is a parameterized transformation (e.g., point-wise convolution), and $\sigma$ is a non-linear activation function.
2.  **Shared Semantic Mapping**: To facilitate semantic alignment across modalities, all heads from all modalities share the same final mapping layer $\omega(\cdot; \mathbf{W}_s)$, projecting features into the label space:
    $$
    \mathbf{P}_{i,t}^{(m)} = \omega(\mathbf{H}_{i,t}^{(m)}; \mathbf{W}_s) \in \mathbb{R}^{K \times D \times H \times W}
    $$

**Intra- and Inter-Modality Fusion.**
Within a modality, we employ an Additive Boosting strategy, summing the outputs of all heads in the set to obtain the boosted prediction $\hat{\mathbf{P}}_i^{(m)}$ for that modality:
$$
\hat{\mathbf{P}}_i^{(m)} = \sum_{t=1}^{n_m} \mathbf{P}_{i,t}^{(m)}
$$
Across modalities, we adopt a Late Fusion strategy, averaging the boosted predictions from all modalities to obtain the final output $\mathbf{O}_i$:
$$
\mathbf{O}_i = \frac{1}{M} \sum_{m=1}^{M} \hat{\mathbf{P}}_i^{(m)}
$$

### 3.4 Adaptive Classifier Assignment (ACA)

To dynamically identify and boost weak modalities, we introduce the Adaptive Classifier Assignment (ACA) mechanism. ACA periodically evaluates the confidence of each modality and injects new segmentation heads into the weakest one.

**Confidence Estimation.** The confidence $s^{(m)}$ of modality $m$ is defined as the expectation of the predicted probability within the foreground region:
$$
s^{(m)} = \mathbb{E}_{i \in \mathcal{B}} \left[ \text{mean}_{\Omega} \left( \mathbf{Y}_i \odot \varsigma(\hat{\mathbf{P}}_i^{(m)}) \right) \right]
$$
where $\varsigma(\cdot)$ is the Sigmoid function, $\Omega$ represents the spatial domain, and $\odot$ denotes the Hadamard product.

**Assignment Rule.** Every $T_{check}$ training epochs, we compute the confidence of all modalities. If the difference in confidence between the strongest and weakest modalities exceeds a threshold $\tau$, a boosting operation is triggered:
$$
\text{if } \max_{m} s^{(m)} - \eta \cdot \min_{m} s^{(m)} > \tau, \quad \text{then } n_{m^*} \leftarrow n_{m^*} + 1
$$
where $m^* = \arg\min_{m} s^{(m)}$ is the currently weakest modality, and $\eta$ is a hyperparameter controlling the tolerance for discrepancy. The newly added head is initialized and included in the joint optimization.

### 3.5 Sustained Boosting Objective

Traditional Gradient Boosting employs a stepwise training strategy, whereas our sustained boosting strategy requires end-to-end joint optimization of all heads. To this end, we design a hybrid loss function comprising three terms.

**Residual Label Definition.** For the $k$-th head of modality $m$, the objective is to fit the residual not yet resolved by the previous $k-1$ heads. We define the residual label $\hat{\mathbf{Y}}_{i,k}^{(m)}$ as:
$$
\hat{\mathbf{Y}}_{i,k}^{(m)} = \text{clamp}\left(\mathbf{Y}_i - \lambda \sum_{j=1}^{k-1} \mathbf{Y}_i \odot \varsigma(\text{sg}[\mathbf{P}_{i,j}^{(m)}]), \min=0\right)
$$
where $\text{sg}[\cdot]$ denotes the stop-gradient operation, and $\lambda$ is the residual decay coefficient.

**Boosting Loss Terms.** For a modality with $n_m > 1$ heads, its boosting loss $\mathcal{L}_{boost}^{(m)}$ consists of three parts:

1.  **Residual Loss ($\mathcal{L}_{res}$)**: Supervises the newly added head $n_m$ to fit the residual label. Since the residual label is a soft label, we employ the Binary Cross-Entropy loss:
    $$
    \mathcal{L}_{res} = \mathcal{L}_{BCE}(\varsigma(\mathbf{P}_{i,n_m}^{(m)}), \hat{\mathbf{Y}}_{i,n_m}^{(m)})
    $$

2.  **Joint Prediction Loss ($\mathcal{L}_{joint}$)**: Ensures that the ensemble output of all heads can accurately approximate the ground truth:
    $$
    \mathcal{L}_{joint} = \mathcal{L}_{seg}(\sum_{j=1}^{n_m} \mathbf{P}_{i,j}^{(m)}, \mathbf{Y}_i)
    $$
    where $\mathcal{L}_{seg}$ is a combination of standard Dice and BCE losses.

3.  **Previous Prediction Loss ($\mathcal{L}_{prev}$)**: To prevent updates to the shared feature extractor $\phi^{(m)}$ from degrading the performance of existing heads, we introduce a stability constraint:
    $$
    \mathcal{L}_{prev} = \mathcal{L}_{seg}(\sum_{j=1}^{n_m-1} \mathbf{P}_{i,j}^{(m)}, \mathbf{Y}_i)
    $$

**Total Objective.** The final total optimization objective $\mathcal{L}_{total}$ is composed of the loss for the final fused prediction and the weighted boosting losses of each modality:
$$
\mathcal{L}_{total} = \mathcal{L}_{seg}(\mathbf{O}_i, \mathbf{Y}_i) + \frac{\gamma}{|\mathcal{M}_b|} \sum_{m \in \mathcal{M}_b} \mathcal{L}_{boost}^{(m)}
$$
where $\mathcal{M}_b = \{m \mid n_m > 1\}$ is the set of boosted modalities, and $\gamma$ is a balancing coefficient. This objective function achieves collaborative optimization of the feature extractor, existing heads, and newly added heads.
