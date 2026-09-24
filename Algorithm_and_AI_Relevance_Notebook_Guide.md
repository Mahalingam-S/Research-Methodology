# RESEARCH NOTEBOOK ENTRY

**Date:** 07 September 2026  
**Student Name:** Mahalingam S.  
**Register No:** CB.SC.P2CSE26016  
**Team No:** Team 22  
**Research Topic:** An AI Framework for Automated Examination Monitoring Using Facial Recognition, Object Detection, and Spatial-Temporal Behavior Analysis  
**Sub-Area (Member 2):** Physical Malpractice Detection, Overhead CCTV Vision & Spatial-Temporal Rule Engine  

---

# Algorithm and its AI Relevance

## Overview of Survey Topic & Master Paper Benchmark Mapping

Our research project establishes a $0 additional hardware cost automated examination supervision framework by repurposing pre-existing laboratory CCTV cameras. The technical implementation leverages 5 core master journal papers (2025–2026) across Team 22:

* **Member 1 (Sibichandaru C.):** Identity Verification & Pre-Exam Biometric Authentication *(Grounded by Paper 5 - Springer 2026)*.
* **Member 2 (Mahalingam S. - My Role):** Physical Malpractice Detection, Overhead CCTV Vision, Mouth Movement / Talking Detection & False-Positive Rule Engine *(Grounded by Paper 1 - Elsevier 2025 & Paper 2 - Nature 2025)*.
* **Member 3 (Vidyasagar S.):** Multimodal Feature Fusion, Temporal Risk Aggregation & Explainable AI (XAI) Evidence Audit *(Grounded by Paper 3 & Paper 4 - Springer 2026)*.

---

## A. Learn the Mathematical Steps in Algorithm (Paper Baseline Algorithms)

### 1. Sub-Area: Physical Malpractice & Mouth Movement Detection (Member 2 - Mahalingam S.)

#### Algorithm A1: ResNet50V2 + LSTM Deep Transfer Learning for Mouth Movement & Talking Detection *(From Paper 1 - Elsevier Procedia CS 2025)*
In Paper 1 (Imaha et al., 2025), deep transfer learning is applied to overhead CCTV video streams to classify suspicious candidate behaviors—specifically **talking to others**, referencing paper notes, and using mobile phones.

**Mathematical Steps:**
1. **Spatial Feature Extraction via ResNet50V2 Backbone:**
   For an input CCTV frame $X_t$, pre-trained residual convolutional blocks extract a high-dimensional spatial feature vector $\mathbf{h}_t$:
   $$\mathbf{h}_t = \phi_{\text{ResNet50V2}}(X_t) \in \mathbb{R}^{2048}$$

2. **Temporal Sequence Modeling via Long Short-Term Memory (LSTM):**
   To catch temporal mouth movement and talking across consecutive video frames $t = 1, \dots, T$, the hidden state $\mathbf{s}_t$ and memory cell $\mathbf{c}_t$ are updated:
   $$\mathbf{f}_t = \sigma(W_f \cdot [\mathbf{s}_{t-1}, \mathbf{h}_t] + b_f) \quad \text{(Forget Gate)}$$
   $$\mathbf{i}_t = \sigma(W_i \cdot [\mathbf{s}_{t-1}, \mathbf{h}_t] + b_i) \quad \text{(Input Gate)}$$
   $$\tilde{\mathbf{c}}_t = \tanh(W_c \cdot [\mathbf{s}_{t-1}, \mathbf{h}_t] + b_c) \quad \text{(Candidate Cell State)}$$
   $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(Updated Cell State)}$$
   $$\mathbf{o}_t = \sigma(W_o \cdot [\mathbf{s}_{t-1}, \mathbf{h}_t] + b_o) \quad \text{(Output Gate)}$$
   $$\mathbf{s}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(Hidden State)}$$

3. **Behavior Classification (Talking vs. Normal):**
   $$P(\text{Talking} \mid X_{1:T}) = \text{Softmax}(W_s \cdot \mathbf{s}_T + b_s)$$

---

#### Algorithm A2: Geometric Mouth Aspect Ratio (MAR) & Lip Motion Distance *(From Paper 3 & 4 - Springer 2026)*
Extracts 2D/3D facial landmarks around candidate lips ($P_1, P_2, \dots, P_8$) to measure mouth opening and lip movement velocity.

**Mathematical Steps:**
1. **Mouth Aspect Ratio (MAR):**
   $$\text{MAR}(t) = \frac{\|P_2(t) - P_8(t)\|_2 + \|P_3(t) - P_7(t)\|_2}{2 \cdot \|P_1(t) - P_5(t)\|_2}$$
   where $\|P_i - P_j\|_2 = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$ is the Euclidean distance.

2. **Lip Motion Velocity / Temporal Variation ($\Delta \text{MAR}$):**
   $$\Delta \text{MAR}(t) = |\text{MAR}(t) - \text{MAR}(t-1)|$$

3. **Mouth Movement Energy over Temporal Window $K$:**
   $$E_{\text{mouth}}(t) = \frac{1}{K} \sum_{k=0}^{K-1} \Delta \text{MAR}(t - k)$$
   High values of $E_{\text{mouth}} > 0.08$ indicate continuous candidate speech / whispering.

---

#### Algorithm A3: Lightweight YOLOv12 Object Detection with C3Ghost & A2C2f *(From Paper 2 - Nature Scientific Reports 2025)*
In Paper 2 (Wang et al., 2025), real-time detection of small gadgets (smartphones, smartwatches, cheat sheets) is achieved by replacing standard convolutional blocks with **C3Ghost** feature extraction modules.

**Mathematical Steps:**
1. **Ghost Convolution (Primary + Cheap Operations):**
   - Primary Convolution: $Y' = X * K'$ (where $K'$ is primary filters, generating $m$ intrinsic features).
   - Cheap Operation: $y_{i,j} = \Phi_{i,j}(y'_i)$ (generating ghost feature maps via linear operations).
   - Output Feature Map: $Y = [Y', Y' * \Phi]$.
   - Reduces computational complexity by factor $s = \frac{n}{m} \approx 2$ to $3 \times$.

2. **Bounding Box Loss (CIoU / Complete IoU):**
   $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$
   where $\rho(\cdot)$ is Euclidean distance of center points, $c$ is diagonal length of smallest enclosing box, and $v = \frac{4}{\pi^2} \left( \arctan\frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan\frac{w}{h} \right)^2$.

---

### 2. Sub-Area: Biometric Authentication (Member 1 - Sibichandaru C.)

#### Algorithm A4: Multi-Facet ResNet + KNN Facial Verification *(From Paper 5 - Springer Discover AI 2026)*
Extracts deep feature vectors $f(x) \in \mathbb{R}^{128}$ normalized to unit length $\|f(x)\|_2 = 1$.
$$\text{Cosine Similarity}(u, v) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

---

### 3. Sub-Area: Multimodal Fusion & Temporal Aggregation (Member 3 - Vidyasagar S.)

#### Algorithm A5: Weighted Multimodal Risk Aggregation & Temporal Moving Average *(From Paper 4 - Springer 2026)*
$$R_{\text{total}}(t) = (0.40 \cdot R_{\text{auth}}) + (0.40 \cdot R_{\text{object}}) + (0.20 \cdot R_{\text{pose}})$$
$$\bar{R}(t) = \frac{1}{W} \sum_{k=0}^{W-1} R_{\text{total}}(t - k) \quad (\text{Window } W = 90 \text{ frames} \approx 3\text{ seconds})$$

---

## B. Approach for an AI Perspective to the Algorithm and Modified Algorithm

### 1. AI Perspective & Modified Algorithm for Member 2 (Mahalingam S.)

#### The Problem in Baseline Algorithms:
Baseline algorithms (like naive MAR thresholds or frame-by-frame posture CNNs) trigger immediate cheating alerts whenever a student's lips move ($\text{MAR} > 0.35$). In real examination environments, candidates frequently read exam question prompts aloud or silently to themselves while facing forward. Existing baseline algorithms produce unacceptably high false-positive alert rates.

#### The Modified AI Solution: Gaze-Pose-Lip Multi-Condition Rule Engine
Our modified algorithm integrates **Mouth Movement ($\text{MAR}$)**, **Lip Motion Energy ($E_{\text{mouth}}$)**, **3D Head Pose Yaw Angle ($|\theta_{\text{yaw}}|$)**, and **Temporal Window Filter ($\Delta t$)**:

$$\text{Flag\_Malpractice}(t) = \left[ (|\theta_{\text{yaw}}(t)| > 30^\circ) \land (\text{MAR}(t) > 0.35) \land (E_{\text{mouth}}(t) > 0.08) \land (\Delta t \ge 3.0\text{ s}) \right] \lor \left[ \text{Gadget\_Detected}(t) == \text{True} \right]$$

```
ALGORITHM 1: Modified Spatial-Temporal Mouth Movement & Malpractice Engine
--------------------------------------------------------------------------------
Input  : Video Stream F(t), 3D Facial Landmarks L(t), YOLO Detections D(t)
Output : Proctoring Status (NORMAL / MALPRACTICE_ALERT)

1: Extract 3D Head Euler Yaw angle theta_yaw from L(t)
2: Calculate Mouth Aspect Ratio MAR(t) and Lip Motion Velocity delta_MAR(t)
3: Calculate Mouth Energy E_mouth over sliding window K = 30 frames
4: Run YOLOv12 / YOLOv8 object detector to retrieve D(t)
5: FOR each object d in D(t) DO:
6:     IF d.class IN {"mobile_phone", "smartwatch", "cheat_sheet"} AND d.confidence >= 0.50 THEN:
7:         TRIGGER INSTANT ALERT: "Prohibited Gadget Detected"
8:         RETURN MALPRACTICE_ALERT
9:     END IF
10: END FOR
11: IF (|theta_yaw| > 30.0°) AND (MAR > 0.35) AND (E_mouth > 0.08) THEN:
12:     Increment talking_counter by 1
13: ELSE:
14:     Reset talking_counter to 0 (Classified as Benign Question Self-Reading)
15: END IF
16: IF talking_counter >= (3.0 * FPS) THEN:
17:     TRIGGER ALERT: "Unauthorized Communication & Persistent Gaze Deviation"
18:     RETURN MALPRACTICE_ALERT
19: ELSE:
20:     RETURN NORMAL
21: END IF
--------------------------------------------------------------------------------
```

---

### 2. Explainable AI (Grad-CAM Visual Heatmap Modification)
To eliminate black-box decision opacity, gradient attributions $\alpha_k^c$ are extracted from convolutional feature maps to overlay visual heatmaps on detected gadgets and mouth/face regions:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right) \quad \text{where} \quad \alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$

---

# Complete Evaluation Metrics and Formulas

Below is the comprehensive list of all evaluation metrics planned to be evaluated in our research framework:

### 1. Intersection over Union (IoU)
$$\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

### 2. Precision
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

### 3. Recall (Sensitivity / Detection Rate)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

### 4. F1-Score
$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

### 5. Mean Average Precision at IoU 0.5 (mAP@0.5)
$$\text{AP} = \sum_{k=1}^{n} P(k) \, \Delta R(k), \qquad \text{mAP@0.5} = \frac{1}{N_{\text{classes}}} \sum_{c=1}^{N_{\text{classes}}} \text{AP}_c$$

### 6. Mean Absolute Error (MAE) for 3D Head Pose Angles
$$\text{MAE}_{\text{pose}} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\theta_{\text{yaw}}^{(i)} - \hat{\theta}_{\text{yaw}}^{(i)}| + |\theta_{\text{pitch}}^{(i)} - \hat{\theta}_{\text{pitch}}^{(i)}| + |\theta_{\text{roll}}^{(i)} - \hat{\theta}_{\text{roll}}^{(i)}|}{3}$$

### 7. Cosine Similarity (Face Verification)
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

### 8. False Positive Rate (FPR)
$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

### 9. Mouth Aspect Ratio (MAR) & Lip Energy ($E_{\text{mouth}}$)
$$\text{MAR} = \frac{\|P_2 - P_8\|_2 + \|P_3 - P_7\|_2}{2 \cdot \|P_1 - P_5\|_2}, \qquad E_{\text{mouth}} = \frac{1}{K} \sum_{k=0}^{K-1} |\text{MAR}(t-k) - \text{MAR}(t-k-1)|$$

### 10. Real-Time Speed (Frames Per Second - FPS)
$$\text{FPS} = \frac{N_{\text{frames}}}{\sum_{i=1}^{N_{\text{frames}}} t_{\text{inference}}^{(i)}}$$

### 11. Multimodal Aggregate Risk Score ($R_{\text{total}}$)
$$R_{\text{total}} = (0.40 \cdot R_{\text{auth}}) + (0.40 \cdot R_{\text{object}}) + (0.20 \cdot R_{\text{pose}})$$
