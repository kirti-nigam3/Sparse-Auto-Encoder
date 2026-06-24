# Sparse Action Encoder (SAE) for Driving Video Representation Learning

## A Research Proposal for CVPR/ICCV/NeurIPS Submission

---

## 1. Research Hypothesis

**Primary Hypothesis:** A sparse autoencoder trained on temporal video clips with a future motion prediction auxiliary objective will be forced, by the structure of the prediction task, to develop a disentangled latent basis whose individual neurons correspond to *action-level* driving behaviors rather than scene-level appearance features — without any action label supervision.

**Sub-hypotheses:**

- **H1:** Adding a future optical flow / trajectory prediction head to the reconstruction objective biases the encoder gradient to attend to ego-motion and agent-motion patterns rather than static texture.
- **H2:** L1 sparsity on the bottleneck will cause polysemanticity collapse: distinct action programs (braking, turning, lane-change) will compete for a small number of "winner" neurons per clip, producing monosemantic action neurons.
- **H3:** The degree to which individual neurons predict held-out action labels (collected only at evaluation time, never during training) is a falsifiable proxy for whether the hypothesis holds.
- **H4:** Longer clip windows (32–64 frames at 10 Hz) are strictly necessary to encode actions; frame-level encoders cannot satisfy the future prediction objective and will therefore not develop temporal abstractions.

---

## 2. Background and Motivation

### 2.1 Why Standard Frame-Level SAEs Fail for Actions

A frame-level sparse autoencoder minimizes:

$$\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|z\|_1$$

where $x \in \mathbb{R}^{H \times W \times 3}$ is a single RGB frame. The reconstruction gradient $\nabla_z \|x - \hat{x}\|_2^2$ is dominated by high-frequency texture and color statistics (Fourier modes with large amplitude). The sparsity term then carves those dominant directions into a dictionary of appearance atoms: sky, road markings, car surfaces, etc. Actions are entirely absent from the signal — a braking clip and a cruising clip look nearly identical at any single frame.

### 2.2 Why Future Prediction Breaks the Appearance Bias

If the decoder must additionally predict the optical flow field $\mathbf{f}_{t \to t+k}$ or the future frame $x_{t+k}$, then the latent code $z$ must contain information about *what is about to happen*. Two clips with identical visual appearance but different futures (e.g., car about to brake vs. car about to accelerate) must therefore map to different $z$ values. This is the mechanism by which future prediction injects action-discriminative structure into the latent space — a form of *predictive coding* as a self-supervised action induction signal.

### 2.3 Why Sparsity Produces Interpretable Neurons

Mechanistic interpretability research (Anthropic, 2023–2024) demonstrates that superposition — the packing of many features into fewer neurons — is the primary obstacle to neuron-level interpretability. Sparsity regularization applied to a *bottleneck wider than necessary* (overcomplete dictionary) resolves superposition: since $z$ has more dimensions than needed for reconstruction alone, the model can afford to dedicate individual neurons to individual concepts rather than linearly combining them. The prediction objective then selects which concepts are *action-relevant*.

---

## 3. Architecture Design

### 3.1 Architecture Recommendation: Hybrid (3D CNN Stem + Temporal Transformer Body)

| Option | Pros | Cons |
|---|---|---|
| 3D CNN only | Fast, translation-equivariant, good for local motion | Limited long-range temporal context; fixed receptive field |
| Video Transformer only | Global temporal attention; flexible context | Data-hungry; quadratic cost in $T$; may overfit to static patches |
| **Hybrid (ours)** | 3D CNN captures short-range local motion; Transformer pools global action context over the clip | Moderate cost; best empirical balance for 32-frame clips |

The hybrid is the correct choice because:
- Turning and lane-changing have *local* optical flow signatures (early frames) AND *global* heading change (across all frames) — requiring both local and global temporal modeling.
- The 3D CNN reduces spatial resolution efficiently before the Transformer, keeping attention complexity manageable.

---

### 3.2 Encoder Architecture $\mathcal{E}$

**Input:** Clip $\mathbf{X} \in \mathbb{R}^{T \times H \times W \times 3}$, e.g. $T=32$, $H=224$, $W=384$ (16:9 crop)

#### Stage 1 — 3D CNN Stem

A ResNet3D-18 or I3D-lite backbone processes overlapping 3D patches:

$$\mathbf{F} = \text{Conv3D-Stem}(\mathbf{X}) \in \mathbb{R}^{\frac{T}{4} \times \frac{H}{16} \times \frac{W}{16} \times C}$$

with $C = 512$. The stem uses:
- 3D depthwise convolutions with kernel $(3,3,3)$
- Temporal stride 2 at layers 2 and 3 (spatial stride as standard ResNet)
- No temporal pooling in the final stage to preserve motion resolution

#### Stage 2 — Factorized Spatial-Temporal Transformer

Reshape $\mathbf{F}$ into tokens $\mathbf{H}_0 \in \mathbb{R}^{N \times C}$ where $N = \frac{T}{4} \cdot \frac{H}{16} \cdot \frac{W}{16}$.

Apply $L=6$ blocks of factorized attention (following TimeSformer):

$$\mathbf{H}_l^{(s)} = \text{SpatialMSA}(\text{LN}(\mathbf{H}_{l-1})) + \mathbf{H}_{l-1}$$
$$\mathbf{H}_l = \text{TemporalMSA}(\text{LN}(\mathbf{H}_l^{(s)})) + \mathbf{H}_l^{(s)}$$

Temporal attention operates across the $\frac{T}{4}$ temporal tokens at each spatial position independently, allowing the model to capture action-level dynamics.

#### Stage 3 — Temporal Pooling to Clip Embedding

Apply mean-pooling across spatial dimensions followed by a learned linear projection:

$$\mathbf{e} = W_{\text{proj}} \cdot \text{MeanPool}_{\text{spatial}}(\mathbf{H}_L) \in \mathbb{R}^{T' \times d}$$

where $T' = T/4$ and $d = 1024$. Optionally compress further with a 1D causal transformer to $\mathbf{e}_\text{clip} \in \mathbb{R}^d$.

---

### 3.3 Sparse Bottleneck Architecture $\mathcal{B}$

This is the core mechanistic interpretability component.

#### Overcomplete Sparse Autoencoder Layer

Given clip embedding $\mathbf{e} \in \mathbb{R}^d$:

$$\mathbf{z}_\text{pre} = W_\text{enc} \mathbf{e} + \mathbf{b}_\text{enc}, \quad W_\text{enc} \in \mathbb{R}^{M \times d}, \quad M \gg d$$

We use $M = 4d$ to $8d$ (e.g., $d = 1024$, $M = 4096$ or $8192$).

Apply ReLU + Top-K gating (TopK-SAE, Anthropic 2024):

$$\mathbf{z} = \text{TopK}(\text{ReLU}(\mathbf{z}_\text{pre}), k)$$

where $k$ is a hyperparameter controlling the *hard sparsity level* (e.g., $k = 32$ out of $M = 4096$ neurons active per clip — sparsity $\approx 0.8\%$). The TopK approach has been shown to be more stable than pure L1 at enforcing a target sparsity level.

Reconstruction from sparse code:

$$\hat{\mathbf{e}} = W_\text{dec} \mathbf{z} + \mathbf{b}_\text{dec}, \quad W_\text{dec} \in \mathbb{R}^{d \times M}$$

**Column normalization constraint:** $\|W_\text{dec}[:, i]\|_2 = 1 \; \forall i$ (dictionary atoms are unit-norm to prevent trivial solutions where one large-magnitude neuron captures everything).

**Dead neuron prevention:** Auxiliary loss to reactivate neurons that have not fired in the last $B = 1000$ batches:

$$\mathcal{L}_\text{dead} = \sum_{i \in \mathcal{D}} \max(0, \epsilon - \bar{a}_i)$$

where $\mathcal{D}$ is the set of dead neurons and $\bar{a}_i$ is the exponential moving average of neuron $i$'s activation frequency.

---

### 3.4 Decoder Architecture $\mathcal{D}$

The decoder has two heads:

#### Head 1 — Reconstruction Decoder

Reconstructs the *clip-level feature embedding* (not raw pixels — this is a feature-space SAE):

$$\hat{\mathbf{e}} = W_\text{dec} \mathbf{z} + \mathbf{b}_\text{dec}$$

Optionally cascade through a lightweight MLP to map back to 3D CNN feature space if pixel-level reconstruction is desired. For research phase 1, feature-space reconstruction is sufficient and avoids the instability of pixel decoders.

#### Head 2 — Future Motion Prediction Decoder (Key Innovation)

Given sparse code $\mathbf{z}$, predict the optical flow field of the *next* $\Delta T$ frames after the clip:

$$\hat{\mathbf{f}}_{T \to T+\Delta T} = \mathcal{D}_\text{flow}(\mathbf{z}) \in \mathbb{R}^{\Delta T \times \frac{H}{4} \times \frac{W}{4} \times 2}$$

$\mathcal{D}_\text{flow}$ is a 3D ConvTranspose network with skip connections from the encoder's intermediate features. The flow target $\mathbf{f}^*$ is computed offline using RAFT on the raw video and stored as a label (it is a *geometric* label, not a semantic/action label — no human annotation required).

**Why optical flow rather than future frames?** Optical flow isolates motion from appearance. Predicting future RGB frames incentivizes the decoder to model both lighting and motion; predicting flow forces the code to prioritize motion structure, more directly linking $z$ to action dynamics.

---

### 3.5 Temporal Modeling Strategy

| Level | Component | What it models |
|---|---|---|
| Frame-level | 3D CNN (3×3×3 kernels) | Local motion patterns, short optical flow |
| Segment-level | Temporal MSA in Transformer | Action phase transitions within a clip (e.g., onset of braking) |
| Clip-level | Temporal pooling + SAE bottleneck | Holistic action identity for the full 32-frame window |
| Prediction-level | Future flow decoder | Causal consequence of the current action state |

The temporal hierarchy ensures that neither short-range nor long-range temporal structure is discarded before the sparse bottleneck.

---

## 4. Multi-Objective Training Loss

### 4.1 Complete Loss Function

$$\mathcal{L}_\text{total} = \lambda_1 \mathcal{L}_\text{recon} + \lambda_2 \mathcal{L}_\text{sparse} + \lambda_3 \mathcal{L}_\text{future} + \lambda_4 \mathcal{L}_\text{temporal} + \lambda_5 \mathcal{L}_\text{dead}$$

---

### 4.2 Reconstruction Loss $\mathcal{L}_\text{recon}$

Feature-space mean squared error between original and reconstructed clip embedding:

$$\mathcal{L}_\text{recon} = \frac{1}{d} \|\mathbf{e} - \hat{\mathbf{e}}\|_2^2 = \frac{1}{d} \|\mathbf{e} - W_\text{dec}\mathbf{z} - \mathbf{b}_\text{dec}\|_2^2$$

For pixel-level reconstruction (optional):

$$\mathcal{L}_\text{recon}^\text{pixel} = \frac{1}{T \cdot H \cdot W} \sum_{t=1}^{T} \|\mathbf{x}_t - \hat{\mathbf{x}}_t\|_1 + \lambda_\text{SSIM}(1 - \text{SSIM}(\mathbf{x}_t, \hat{\mathbf{x}}_t))$$

L1 is preferred over L2 for pixel reconstruction to avoid blurry outputs.

---

### 4.3 Sparsity Loss $\mathcal{L}_\text{sparse}$

For TopK gating, sparsity is *enforced by construction*; the auxiliary sparsity loss is:

$$\mathcal{L}_\text{sparse} = \frac{1}{M} \|\mathbf{z}\|_1$$

This penalizes large activation magnitudes even among the TopK active neurons, encouraging compact representations. An alternative is the *tanh sparsity penalty* (used in Anthropic's SAE work):

$$\mathcal{L}_\text{sparse}^\text{tanh} = \frac{1}{M} \sum_{i=1}^{M} \tanh\left(\frac{8 \cdot \mathbf{z}_i}{\bar{a}_i}\right)$$

where $\bar{a}_i$ is the running mean activation of neuron $i$, normalizing the penalty relative to typical activation scale.

---

### 4.4 Future Motion Prediction Loss $\mathcal{L}_\text{future}$

Let $\mathbf{f}^* \in \mathbb{R}^{\Delta T \times h \times w \times 2}$ be the ground-truth optical flow computed by RAFT for the $\Delta T$ frames following the current clip:

$$\mathcal{L}_\text{flow} = \frac{1}{\Delta T \cdot h \cdot w} \sum_{\tau=1}^{\Delta T} \sum_{p} \|\hat{\mathbf{f}}_{\tau,p} - \mathbf{f}^*_{\tau,p}\|_2$$

where $p$ indexes spatial positions.

For richer action modeling, additionally predict:

1. **Ego-trajectory:** future ego-vehicle $(x, y, \theta)$ displacement sequence from IMU/GPS:

$$\mathcal{L}_\text{ego} = \frac{1}{\Delta T} \sum_{\tau=1}^{\Delta T} \|\hat{\boldsymbol{\delta}}_\tau - \boldsymbol{\delta}^*_\tau\|_2$$

2. **Agent bounding box flow:** future motion of detected agents (from an off-the-shelf detector, run offline):

$$\mathcal{L}_\text{agent} = \frac{1}{N_\text{agents}} \sum_{n} \|\hat{\mathbf{v}}_n - \mathbf{v}^*_n\|_2$$

The combined future prediction loss:

$$\mathcal{L}_\text{future} = \mathcal{L}_\text{flow} + \mu_1 \mathcal{L}_\text{ego} + \mu_2 \mathcal{L}_\text{agent}$$

**Why this forces action encoding:**

- A clip with the ego-vehicle *about to turn left* has a distinctive future ego-trajectory $\boldsymbol{\delta}^*$. The gradient of $\mathcal{L}_\text{ego}$ w.r.t. $z$ will strengthen neurons that co-activate with leftward curvature.
- A clip with a pedestrian *about to cross* has a distinctive agent flow pattern $\mathbf{v}^*_n$. This gradient strengthens neurons correlating with lateral pedestrian motion.
- Braking clips have $\|\boldsymbol{\delta}^*\|_2 \to 0$ and distinctive decelerating flow. Neurons encoding braking are reinforced.

---

### 4.5 Temporal Consistency Loss $\mathcal{L}_\text{temporal}$

Consecutive overlapping clips should produce smoothly varying latent codes (actions do not instantaneously change). For clip pair $(i, j)$ with temporal overlap $\alpha_{ij} \in [0, 1]$:

$$\mathcal{L}_\text{temporal} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \alpha_{ij} \cdot \|\mathbf{z}_i - \mathbf{z}_j\|_2^2$$

where $\mathcal{P}$ is the set of consecutive clip pairs within a sequence. The weight $\alpha_{ij}$ is proportional to the temporal overlap fraction, so highly overlapping clips are penalized more for divergent codes.

---

### 4.6 Dead Neuron Auxiliary Loss $\mathcal{L}_\text{dead}$

$$\mathcal{L}_\text{dead} = \frac{1}{|\mathcal{D}|} \sum_{i \in \mathcal{D}} \left(1 - \frac{\bar{a}_i}{\epsilon_\text{target}}\right)^2$$

where $\mathcal{D} = \{i : \bar{a}_i < \epsilon\}$ and $\epsilon_\text{target} = 0.01$ (target activation frequency of 1%).

---

### 4.7 Loss Weighting Schedule

| Loss | $\lambda$ initial | $\lambda$ final | Rationale |
|---|---|---|---|
| $\mathcal{L}_\text{recon}$ | 1.0 | 1.0 | Always on |
| $\mathcal{L}_\text{sparse}$ | 0 → warm-up | 0.04 | Warm up to avoid representation collapse |
| $\mathcal{L}_\text{future}$ | 0.5 | 1.0 | Anneal up to emphasize prediction |
| $\mathcal{L}_\text{temporal}$ | 0.1 | 0.1 | Fixed |
| $\mathcal{L}_\text{dead}$ | 0.01 | 0.01 | Fixed |

---

## 5. Preprocessing Pipeline

### 5.1 Video Ingestion

```
Raw NVIDIA .mp4 / .h264 / .bag files
      ↓
Decode with FFmpeg (hardware decode on GPU)
      ↓
Validate: check frame rate, resolution, corruption
      ↓
Segment into driving sequences
(remove parked/idle: v < 2 km/h for > 5 s → discard)
      ↓
Write to .tfrecord / .hdf5 shards (pre-decoded frames)
```

### 5.2 Frame Extraction and Temporal Window Generation

- **Target frame rate:** 10 Hz (downsample from native 30 Hz with stride 3)
- **Rationale:** Most driving actions unfold over 2–6 seconds (20–60 frames at 10 Hz); 30 Hz is redundant and triples storage.
- **Clip stride:** $S = T/2$ (50% overlap between consecutive clips). Increases training diversity and is required for the temporal consistency loss.
- **Clip boundary:** clips do not cross sequence boundaries (scene cuts, parking events, or sequence gaps > 1 second).

### 5.3 Frame Sampling Strategy

**Uniform sampling** within each clip (every 3rd frame from original 30 Hz → 10 Hz effective). Do not use random frame sampling during training — temporal consistency loss requires reproducible clip boundaries.

For data augmentation:
- **Temporal jitter:** ±2 frames random shift of clip start position (during training only)
- **Speed augmentation:** sample every 2nd or 4th frame to simulate faster/slower driving contexts
  - 60% standard (stride 3), 20% fast (stride 2), 20% slow (stride 4)

### 5.4 Spatial Preprocessing

```
Raw frame (variable resolution, typically 1920×1080)
      ↓
Center crop to 16:9 (remove hood, sky extremes)
      ↓
Resize to 384×224 (2× downscale of 768×448)
      ↓
Normalize: ImageNet mean/std per channel
      ↓
Random horizontal flip (with flow field sign flip for left/right)
      ↓
Color jitter (brightness ±0.2, contrast ±0.2, saturation ±0.1)
      ↓
Random temporal reversal (rare, p=0.1, for reverse-motion robustness)
```

> **Note:** Do NOT apply heavy augmentation that destroys motion structure (e.g., extreme zoom, rotation > 5°) as the optical flow prediction head is motion-sensitive.

### 5.5 Optical Flow Pre-computation

Compute RAFT-Large flow offline on all frame pairs (high quality, no need for online computation):

```
For each consecutive frame pair (t, t+1) in the extended window [1, T + ΔT]:
    f_{t→t+1} = RAFT(x_t, x_{t+1})
Save to .npy files indexed by sequence + frame index
```

Downscale flow to $\frac{H}{4} \times \frac{W}{4} = 56 \times 96$ for decoder target.

---

## 6. Clip Length Recommendation

### 6.1 Analysis

| Clip Length | Duration @ 10 Hz | Action Coverage | Memory | Recommendation |
|---|---|---|---|---|
| 16 frames | 1.6 s | Partial — captures onset or execution of action but rarely full arc | Low | Research baseline only |
| **32 frames** | **3.2 s** | **Good — covers most action arcs (lane change: ~2–3 s; braking: ~1–4 s; turn: ~3–5 s)** | **Medium** | **Primary recommendation** |
| 64 frames | 6.4 s | Excellent arc coverage, but includes multiple unrelated actions per clip | High | Secondary experiment |

### 6.2 Justification for 32 Frames

- A standard lane change takes 2.5–4.0 seconds: fully captured in 32 frames.
- A braking event from 50 km/h to stop takes approximately 3–4 seconds: fits in 32 frames.
- A signalized left turn takes 3–7 seconds: partially captured (onset + execution).
- 64 frames risks the clip containing multiple actions (lane change *then* deceleration) which conflicts with the goal of monosemantic action neurons.
- 16 frames is insufficient for temporal consistency loss and for the future prediction horizon.

---

## 7. Label-Free Training Strategy

The system is entirely self-supervised. No action labels are used during training. The only supervision signals are:

| Signal | Source | Annotation type |
|---|---|---|
| Clip reconstruction | Raw video frames | None (self-supervised) |
| Future optical flow | RAFT applied to consecutive frames | Geometric (no semantics) |
| Future ego-trajectory | IMU/GPS from NVIDIA dataset | Sensor (no human label) |
| Future agent motion | Off-the-shelf object detector + tracker | Algorithmic (no human label) |
| Temporal consistency | Consecutive clip pairs | Structural (no human label) |

This is philosophically aligned with self-supervised video representation learning (SimCLR-Video, VideoMAE, V-JEPA) but with the key distinction that the bottleneck is *sparse* rather than dense, and the prediction target is *motion-specific* rather than RGB-level.

---

## 8. Experimental Plan

### 8.1 Dataset

**Primary datasets:**
- **nuScenes** — 1000 scenes, 20s each, 12 cameras, 10 Hz, with IMU/GPS
- **Waymo Open Dataset** — 1000 sequences, ~20s, front camera
- **BDD100K** — 100K videos, 40s each, dashcam, diverse conditions
- **COMMA.ai** — 45 hours of highway/urban driving, GPS + IMU

Initial experiments: **nuScenes** (cleaner, shorter sequences, well-studied) + **COMMA.ai** (longer sequences for pretraining).

### 8.2 Training Protocol

| Phase | Description | Duration |
|---|---|---|
| Phase 0: Pretraining | Train encoder (3D CNN + Transformer) on reconstruction + temporal consistency only. No SAE. | 50 epochs |
| Phase 1: SAE warm-up | Freeze encoder; train SAE bottleneck on frozen features with reconstruction + L1 sparse loss only. | 20 epochs |
| Phase 2: Joint fine-tuning | Unfreeze encoder; train end-to-end with all 5 losses. | 50 epochs |
| Phase 3: Future head | Add future prediction decoder; fine-tune full system. | 30 epochs |

The phased approach prevents the SAE from collapsing before the encoder has learned a useful feature space.

### 8.3 Ablation Studies

| Ablation | Variable | Values |
|---|---|---|
| A1: Sparsity coefficient | $k$ (TopK) | 8, 16, 32, 64, 128 |
| A2: Latent dimension | $M$ | 512, 1024, 2048, 4096, 8192 |
| A3: Clip duration | $T$ | 8, 16, 32, 64 |
| A4: Future prediction horizon | $\Delta T$ | 0 (ablated), 4, 8, 16, 32 |
| A5: Encoder architecture | Backbone | 3D ResNet, TimeSformer, Hybrid (ours) |
| A6: Flow vs. ego-trajectory | Target type | RAFT only, ego only, agent only, combined |
| A7: Feature-space vs. pixel SAE | Reconstruction level | Feature, pixel, both |

For each ablation, the primary metric is **Action Neuron Purity** (NSS-based; defined in Section 10).

---

## 9. Neuron Interpretability Techniques

### 9.1 Top-K Activating Clips

For each neuron $i$, collect the top-100 clips by activation magnitude $z_i$ from a held-out evaluation set. Visualize as a $10 \times 10$ grid of video thumbnails. A monosemantic neuron will show visually coherent action themes across all 100 clips.

### 9.2 Activation Maps (Spatiotemporal Grad-CAM)

Compute the gradient of neuron $i$'s activation with respect to input frames (Grad-CAM generalized to 3D):

$$\mathbf{A}^i_{t, y, x} = \text{ReLU}\left(\sum_c \alpha^i_c \cdot \mathbf{F}^c_{t, y, x}\right)$$

$$\alpha^i_c = \frac{1}{T \cdot H' \cdot W'} \sum_{t, y, x} \frac{\partial z_i}{\partial \mathbf{F}^c_{t, y, x}}$$

This produces a spatiotemporal heatmap showing *where and when* in the clip the neuron is responding. For a braking neuron, the map should highlight the road ahead and the onset frames of braking.

### 9.3 Latent Traversal

Fix the sparse code $\mathbf{z}$ for a given clip; vary neuron $i$'s activation from 0 to $a_\text{max}$ while keeping all others fixed. Decode the modified code and visualize the predicted future flow $\hat{\mathbf{f}}$:

$$\hat{\mathbf{f}}(\alpha) = \mathcal{D}_\text{flow}(W_\text{dec}(\mathbf{z} + \alpha \cdot \mathbf{e}_i))$$

If neuron $i$ is a braking neuron, increasing $z_i$ should produce decreasing ego-flow magnitude (the car appears to decelerate).

### 9.4 Feature Visualization via Optimization

Find the clip embedding that maximally activates neuron $i$ subject to being on the data manifold:

$$\mathbf{e}^* = \arg\max_{\mathbf{e} \in \mathcal{M}} z_i(\mathbf{e})$$

Approximated by gradient ascent on the encoder with a natural video prior, or by retrieval from the dataset as a tractable alternative.

---

## 10. Latent Space Evaluation

### 10.1 Dimensionality Reduction

Collect sparse codes $\{\mathbf{z}_n\}_{n=1}^N$ for $N = 50{,}000$ evaluation clips.

**UMAP:** 2D/3D projection with `n_neighbors=15`, `min_dist=0.1`, `metric='cosine'`. Color by:
- Post-hoc annotated action class (lane change, turn, brake, etc.)
- Speed bin (0–20, 20–50, 50+ km/h)
- Scene type (highway, urban, intersection)

**t-SNE:** 2D projection with `perplexity=50` for local structure comparison.

A good model will show clean clusters in UMAP space that spatially correspond to action categories, even though no action label was used in training.

### 10.2 Clustering with HDBSCAN

**HDBSCAN** on the UMAP embedding (density-based, no need to specify $k$):
- `min_cluster_size=100`
- `min_samples=10`
- Report: number of clusters, noise fraction

**Metrics:**
- **Calinski-Harabasz Index:** measures cluster compactness and separation
- **Davies-Bouldin Index:** measures intra-cluster similarity vs. inter-cluster difference
- **Silhouette Score:** average clip-to-centroid distance vs. nearest-cluster distance

### 10.3 Cluster-to-Action Alignment

After HDBSCAN clustering, sample 50 clips per cluster and manually assign a majority action label (evaluation-only annotation, never used in training). Compute:

$$\text{Action Purity} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \frac{\max_a |\{n \in c : \text{label}(n) = a\}|}{|c|}$$

A purity > 0.7 indicates each cluster is dominated by one action type.

---

## 11. Quantitative Metrics for Action Concept Emergence

### 11.1 Primary Metrics

**Neuron Selectivity Score (NSS):**

$$\text{NSS}_i = D_\text{KL}(P_i \| U) = \sum_a P_i(a) \log \frac{P_i(a)}{1/A}$$

where $P_i(a) = \mathbb{E}[\mathbf{1}[n_\text{most active} = i] | \text{label} = a]$ and $A$ is the number of action classes. High NSS = neuron is selective for specific actions.

**Linear Probe Accuracy (LP-Acc):** Train a linear classifier on top of frozen $\mathbf{z}$ representations to predict post-hoc action labels. Compare against:
- Dense encoder without SAE (upper bound)
- Frame-level SAE (baseline)
- Random sparse code (lower bound)

**Action Separation (AS):** Mean pairwise cosine distance between cluster centroids in $\mathbf{z}$-space for different action classes:

$$\text{AS} = \frac{2}{A(A-1)} \sum_{a \neq a'} \left(1 - \frac{\bar{\mathbf{z}}_a \cdot \bar{\mathbf{z}}_{a'}}{\|\bar{\mathbf{z}}_a\| \|\bar{\mathbf{z}}_{a'}\|}\right)$$

Higher AS = better-separated action representations.

**Future Prediction MAE:** Mean absolute error of predicted future optical flow vs. RAFT ground truth on held-out clips.

### 11.2 Secondary Metrics

- **Sparsity statistics:** Average $\ell_0$ norm of $\mathbf{z}$, neuron utilization rate (fraction of neurons with $\bar{a}_i > 0.01$)
- **Reconstruction $R^2$:** Fraction of feature variance explained by the SAE reconstruction
- **Dead neuron fraction:** Fraction of neurons with utilization < 0.001
- **Cross-action confusion matrix:** Most-selective neuron per clip vs. action label

---

## 12. Failure Mode Analysis and Mitigations

| Failure Mode | Cause | Mitigation |
|---|---|---|
| Scene appearance dominates latent | Reconstruction loss too large relative to future prediction | Increase $\lambda_3$; use feature-space reconstruction; apply style transfer augmentation |
| All neurons die (dead neuron problem) | Sparsity too aggressive in early training | Warm-up schedule for $\lambda_2$; use TopK with gradual $k$ reduction; auxiliary dead neuron loss |
| Polysemantic neurons (actions superposed) | $M$ too small or $k$ too large | Increase $M$; decrease $k$; apply neuron split heuristic (Anthropic 2024) |
| Highway-only neurons, no urban action neurons | Dataset imbalance | Balanced sampling by scene type; oversample intersection/pedestrian clips |
| Left/right asymmetry | More right turns in data | Horizontal flip augmentation (with flow sign correction); class-balanced sampling |
| Future prediction overfits to static scenes | Low-motion clips are easier to predict | Sample minimum-motion threshold; weight loss by ego-speed; discard clips with $v < 5$ km/h |
| SAE learns trajectory style, not action semantics | Ego-trajectory target too dominant | Balance $\mu_1$ (ego) and $\mathcal{L}_\text{flow}$ terms; add agent-based prediction targets |
| Encoder collapses to constant representation | Temporal consistency loss too strong | Reduce $\lambda_4$; use contrastive temporal loss instead |

---

## 13. Related Work Positioning

| Work | Method | Limitation vs. Ours |
|---|---|---|
| VideoMAE (He et al. 2022) | Masked autoencoder on video | Dense latent; no sparsity; no action interpretability |
| V-JEPA (Lecam et al. 2024) | Joint-embedding predictive architecture | Dense; feature-level prediction; not interpretable |
| SimCLR-Video | Contrastive; frame-level | Appearance-biased; no temporal action modeling |
| Anthropic SAE (Templeton et al. 2024) | SAE on language model activations | Language domain; no temporal video modeling |
| ActionCLIP (Wang et al. 2022) | Video-text contrastive | Requires text/action labels |
| **SAE-Drive (Ours)** | Sparse temporal AE + future prediction | **No action labels; interpretable action neurons; driving-specific** |

---

## 14. Publication Roadmap

### 14.1 Narrative Arc

The paper's core claim:

> *"Sparse autoencoders with future motion prediction self-organize to discover action-centric representations from raw driving video, without any action supervision."*

This is falsifiable (via linear probe and neuron purity), novel (no prior work combines SAE mechanistic interpretability with video action discovery), and practically significant (zero-label interpretability for safety-critical autonomy systems).

### 14.2 Expected Contributions

1. **Architecture:** First hybrid 3D CNN + Transformer + overcomplete sparse bottleneck for action discovery in driving video.
2. **Training:** Future optical flow + ego-trajectory joint prediction as an action induction signal without semantic labels.
3. **Mechanistic Interpretability:** First application of SAE-style neuron-level interpretability to autonomous driving, discovering action neurons corresponding to real-world driving behaviors.
4. **Evaluation:** Action Neuron Purity (NSS-based), Action Separation metric, and standardized UMAP+HDBSCAN evaluation protocol for video representation interpretability.
5. **Benchmark:** Post-hoc annotated interpretability evaluation set (500–1000 clips, 11 action classes) released as a community benchmark.

### 14.3 Milestones

| Milestone | Deliverable | Timeline |
|---|---|---|
| M0 | Data pipeline + RAFT flow precomputation complete | Week 1–2 |
| M1 | Baseline 3D CNN encoder pretrained on reconstruction | Week 3–5 |
| M2 | SAE bottleneck trained on frozen features; first neuron visualizations | Week 6–8 |
| M3 | Future prediction head integrated; ablation A4 complete | Week 9–11 |
| M4 | Full end-to-end training; ablations A1–A3 complete | Week 12–15 |
| M5 | UMAP/HDBSCAN analysis; post-hoc action annotation (500 clips) | Week 16–17 |
| M6 | Quantitative metrics (LP-Acc, NSS, AS); ablations A5–A7 | Week 18–20 |
| M7 | Neuron interpretability visualizations (top-K clips, activation maps, traversals) | Week 21–22 |
| M8 | Paper writing, figure production | Week 23–26 |

---

## 15. Summary of Design Decisions

| Decision | Choice | Justification |
|---|---|---|
| Encoder | Hybrid 3D CNN + Temporal Transformer | Best local + global temporal coverage |
| Bottleneck | Overcomplete TopK-SAE ($M = 4d$, $k = 32$) | Resolves superposition; enforces hard sparsity |
| Prediction target | Optical flow + ego-trajectory + agent motion | Motion-specific signals; no semantic annotation required |
| Clip length | 32 frames @ 10 Hz (3.2 s) | Covers most action arc durations |
| Loss | Reconstruction + TopK-L1 + future flow/ego + temporal consistency | Each term serves a specific representational objective |
| Evaluation | NSS, LP-Acc, AS, Action Purity, UMAP/HDBSCAN | Multi-level: neuron, cluster, global structure |
| Training | 3-phase (pretrain → SAE warm-up → joint fine-tune) | Prevents collapse; ensures stable sparse basis formation |

---

## 16. Implementation Checklist

When proceeding to implementation, the following components are required:

- [ ] PyTorch model code (encoder, SAE bottleneck, decoder heads)
- [ ] RAFT flow precomputation script
- [ ] Dataset loader with temporal window generation
- [ ] Training loop with all 5 losses and warm-up scheduler
- [ ] Neuron interpretability analysis scripts (top-K, 3D Grad-CAM, traversal)
- [ ] UMAP + HDBSCAN evaluation notebook
- [ ] Linear probe evaluation harness
- [ ] Ablation study configuration files (YAML / Hydra)
