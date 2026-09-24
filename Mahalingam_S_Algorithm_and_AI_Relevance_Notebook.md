# RESEARCH LAB NOTEBOOK ENTRY

**Date:** 07 September 2026  
**Student Name:** Mahalingam S.  
**Register No:** CB.SC.P2CSE26016  
**Team & Role:** Team No. 22 | Member 2 – System Architecture & Overhead CCTV Vision  
**Course & Dept:** 19CSE601 Research Methodology | Department of Computer Science & AI, Amrita Vishwa Vidyapeetham  
**Research Topic:** An AI Framework for Automated Examination Monitoring Using Facial Recognition, Object Detection, and Spatial-Temporal Behavior Analysis  
**Sub-Area Allocation:** Physical Malpractice Detection, Overhead CCTV Computer Vision, Mouth Movement / Talking Detection & Spatial-Temporal Rule Engine  

---

# Algorithm and its AI relevance

## Overview of Member 2 Sub-Area Scope & Paper Mapping

As **Member 2 (Mahalingam S.)**, my primary technical responsibilities in Team 22 focus on:
1. **$0 Additional Hardware Cost Overhead CCTV Integration:** Repurposing pre-installed laboratory cameras to supervise student workstations simultaneously.
2. **Mouth Movement & Talking Detection:** Catching unauthorized candidate communication using temporal deep transfer learning and facial landmark geometry.
3. **Real-Time Prohibited Gadget Detection:** Identifying smartphones, smartwatches, and cheat sheets using lightweight YOLO object detection.
4. **False-Positive Elimination Rule Engine:** Formulating a spatial-temporal rule engine combining head pose yaw rotation, mouth aspect ratio, lip motion energy, and time thresholding to filter out benign question self-reading.

### Primary Benchmark Master Papers for Member 2:
* **Paper 1 (Elsevier Procedia Computer Science, 2025):** *Smart CCTV Exam Monitoring to Detect and Alert Suspicious Activities Using Deep Transfer Learning* (Imaha et al.) — Validates overhead CCTV monitoring and deep transfer learning for talking and posture detection.
* **Paper 2 (Nature Scientific Reports, 2025):** *Online Exam Cheating Detection and Blockchain Trusted Deposit Based on YOLOv12* (Wang et al.) — Establishes lightweight YOLO detection benchmarks for small prohibited gadgets on desks.

---

## A. Learn the Mathematical Steps in Algorithm (Member 2 Baseline Algorithms)

### 1. ResNet50V2 + LSTM Deep Transfer Learning for Overhead CCTV Mouth Movement & Talking Detection *(Paper 1 - Elsevier 2025)*

In Paper 1 (Imaha et al., 2025), deep transfer learning is applied to overhead CCTV video streams to extract spatial-temporal features and classify suspicious candidate actions, specifically **talking to others**, referencing notes, and phone handling.

#### Mathematical Steps:
1. **Spatial Feature Vector Extraction via ResNet50V2 Backbone:**
   Each input CCTV video frame $X_t$ at time $t$ is passed through pre-trained residual convolutional layers to extract a 2048-dimensional feature embedding $\mathbf{h}_t$:
   $$\mathbf{h}_t = \phi_{\text{ResNet50V2}}(X_t) \in \mathbb{R}^{2048}$$

2. **Temporal Sequence Modeling via Long Short-Term Memory (LSTM):**
   To capture temporal mouth movement and candidate speech patterns across consecutive frames $t = 1, \dots, T$, the LSTM memory cell $\mathbf{c}_t$ and hidden state $\mathbf{s}_t$ are updated using gate equations:
   $$\mathbf{f}_t = \sigma(W_f \cdot [\mathbf{s}_{t-1}, \mathbf{h}_t] + b_f) \quad \text{(Forget Gate)}$$
   $$\mathbf{i}_t = \sigma(W_i \cdot [\mathbf{s}_{t-1}, \mathbf{h}_t] + b_i) \quad \text{(Input Gate)}$$
   $$\tilde{\mathbf{c}}_t = \tanh(W_c \cdot [\mathbf{s}_{t-1}, \mathbf{h}_t] + b_c) \quad \text{(Candidate State)}$$
   $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(Updated Memory Cell)}$$
   $$\mathbf{o}_t = \sigma(W_o \cdot [\mathbf{s}_{t-1}, \mathbf{h}_t] + b_o) \quad \text{(Output Gate)}$$
   $$\mathbf{s}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(Hidden State Vector)}$$

3. **Behavior Class Probability Output:**
   $$P(\text{Talking} \mid X_{1:T}) = \text{Softmax}(W_s \cdot \mathbf{s}_T + b_s)$$

---

### 2. Geometric Mouth Aspect Ratio (MAR) & Lip Motion Energy *(Paper 3 & 4 - Springer 2026)*

Extracts 2D/3D facial landmarks around lips ($P_1, P_2, \dots, P_8$) to measure lip separation and temporal speech velocity.

#### Mathematical Steps:
1. **Mouth Aspect Ratio (MAR):**
   $$\text{MAR}(t) = \frac{\|P_2(t) - P_8(t)\|_2 + \|P_3(t) - P_7(t)\|_2}{2 \cdot \|P_1(t) - P_5(t)\|_2}$$
   where $\|P_i - P_j\|_2 = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$ represents Euclidean landmark distance.

2. **Lip Motion Velocity / Temporal Variation ($\Delta \text{MAR}$):**
   $$\Delta \text{MAR}(t) = |\text{MAR}(t) - \text{MAR}(t-1)|$$

3. **Mouth Movement Energy ($E_{\text{mouth}}$) Over Window $K$:**
   $$E_{\text{mouth}}(t) = \frac{1}{K} \sum_{k=0}^{K-1} \Delta \text{MAR}(t - k) \quad (\text{where } K = 30 \text{ frames} \approx 1.0\text{ s})$$
   A sustained value of $E_{\text{mouth}} > 0.08$ indicates active continuous talking / whispering.

---

### 3. Lightweight YOLO Object Detection with C3Ghost & CIoU Loss *(Paper 2 - Nature 2025)*

Paper 2 (Wang et al., 2025) detects small prohibited objects on candidate desks (smartphones, smartwatches, cheat sheets) in real-time.

#### Mathematical Steps:
1. **Bounding Box Parameter Prediction:**
   Predictions $(t_x, t_y, t_w, t_h)$ are transformed into normalized bounding box coordinates:
   $$b_x = \sigma(t_x) + c_x, \qquad b_y = \sigma(t_y) + c_y$$
   $$b_w = p_w \cdot e^{t_w}, \qquad b_h = p_h \cdot e^{t_h}$$

2. **Ghost Convolution Module (C3Ghost Efficiency):**
   Splits feature generation into primary convolutions $Y' = X * K'$ and cheap linear operations $Y = [Y', Y' * \Phi]$, reducing computation by factor $s \approx 2 \text{ to } 3\times$.

3. **Complete IoU (CIoU) Loss Function:**
   $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{\text{gt}})}{c^2} + \alpha v$$
   where $\rho(\cdot)$ is the Euclidean distance between box center points, $c$ is the diagonal length of the smallest enclosing bounding box, and $v = \frac{4}{\pi^2} \left( \arctan\frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan\frac{w}{h} \right)^2$.

---

### 4. 3D Head Pose Euler Angle Rotation Estimation ($\text{Yaw}, \text{Pitch}, \text{Roll}$)

Projects 3D facial landmarks to estimate head orientation angles ($\theta_{\text{yaw}}, \theta_{\text{pitch}}, \theta_{\text{roll}}$) relative to the overhead/frontal camera axis.

#### Mathematical Steps:
1. **Perspective-n-Point (PnP) Pose Matrix:**
   $$\mathbf{s} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{R}_{3\times3} & \mathbf{T}_{3\times1} \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$
   where $\mathbf{K}$ is the intrinsic camera matrix, $\mathbf{R}$ is the 3D rotation matrix, and $\mathbf{T}$ is translation.

2. **Euler Yaw Angle Extraction ($\theta_{\text{yaw}}$):**
   $$\theta_{\text{yaw}} = \text{atan2}(R_{21}, R_{11}) \times \frac{180^\circ}{\pi}$$
   $\theta_{\text{yaw}} > 30^\circ$ indicates sideways head rotation toward a neighboring candidate desk.

---

## B. Approach for an AI Perspective to the Algorithm and Modified Algorithm

### 1. The Problem in Baseline Algorithms
Conventional baseline algorithms (such as frame-level CNN posture classifiers or naive MAR thresholding) trigger immediate cheating alerts whenever a candidate's mouth opens ($\text{MAR} > 0.35$). 

In real laboratory examinations, candidates frequently read exam question prompts silently to themselves while looking straight down at their desk. Baseline algorithms treat every lip movement as cheating, producing unacceptably high false-positive alert rates that disrupt invigilation.

---

### 2. The Modified AI Solution: Spatial-Temporal Gaze-Pose-Lip Rule Engine

To solve false positives, I formulated a multi-condition spatial-temporal rule engine. Under this modified algorithm, lip movement ($\text{MAR} > 0.35$) and mouth energy ($E_{\text{mouth}} > 0.08$) are classified as **benign question self-reading** unless co-occurring with a 3D Head Pose Yaw rotation exceeding 30 degrees toward an adjacent desk ($|\theta_{\text{yaw}}| > 30^\circ$) sustained over a temporal sliding window $\Delta t \ge 3.0\text{ seconds}$.

#### Modified Rule Engine Formulation:

$$\text{Flag\_Malpractice}(t) = \left[ (|\theta_{\text{yaw}}(t)| > 30^\circ) \land (\text{MAR}(t) > 0.35) \land (E_{\text{mouth}}(t) > 0.08) \land (\Delta t \ge 3.0\text{ s}) \right] \lor \left[ \text{Gadget\_Detected}(t) == \text{True} \right]$$

---

### 3. Pseudo-Code of Member 2 Modified AI Rule Engine

```
ALGORITHM 1: Modified Spatial-Temporal Mouth Movement & Malpractice Engine (Member 2)
--------------------------------------------------------------------------------
Input  : Frame Stream F(t), 3D Facial Landmarks L(t), YOLO Detections D(t)
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
17:     TRIGGER ALERT: "Persistent Gaze Deviation & Unauthorized Communication"
18:     RETURN MALPRACTICE_ALERT
19: ELSE:
20:     RETURN NORMAL
21: END IF
--------------------------------------------------------------------------------
```

---

### 4. Explainable AI (Grad-CAM Visual Heatmap Activation)

To eliminate the "black-box" nature of AI detection, feature activation gradients $A^k$ are extracted from convolutional feature maps to overlay visual heatmaps on detected gadgets and mouth/head regions:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right) \quad \text{where} \quad \alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$

---

# Evaluation Metrics and Formulas (Member 2 Scope)

Below is the complete list of evaluation metrics planned for evaluation in Member 2's computer vision and rule engine framework, complete with exact mathematical formulas:

### 1. Intersection over Union (IoU)
Measures bounding-box overlap accuracy against ground-truth labels for prohibited gadgets.

$$\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

---

### 2. Precision
Measures the proportion of correctly identified malpractice alerts out of all triggered alerts.

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

---

### 3. Recall (Sensitivity / Detection Rate)
Measures the proportion of actual physical cheating instances correctly caught by the system.

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

---

### 4. F1-Score
Harmonic mean balancing precision and recall into a single metric.

$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

---

### 5. Mean Average Precision at IoU 0.5 (mAP@0.5)
Evaluates object detection performance across object classes (mobile_phone, smartwatch, cheat_sheet) at 50% IoU threshold.

$$\text{AP} = \sum_{k=1}^{n} P(k) \, \Delta R(k), \qquad \text{mAP@0.5} = \frac{1}{N_{\text{classes}}} \sum_{c=1}^{N_{\text{classes}}} \text{AP}_c$$

---

### 6. Mean Absolute Error (MAE) for 3D Head Pose Angles
Quantifies rotation error in degrees between predicted Euler angles $(\hat{\theta}_{\text{yaw}}, \hat{\theta}_{\text{pitch}}, \hat{\theta}_{\text{roll}})$ and ground-truth orientation.

$$\text{MAE}_{\text{pose}} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\theta_{\text{yaw}}^{(i)} - \hat{\theta}_{\text{yaw}}^{(i)}| + |\theta_{\text{pitch}}^{(i)} - \hat{\theta}_{\text{pitch}}^{(i)}| + |\theta_{\text{roll}}^{(i)} - \hat{\theta}_{\text{roll}}^{(i)}|}{3}$$

---

### 7. Mouth Aspect Ratio ($\text{MAR}$) & Lip Motion Energy ($E_{\text{mouth}}$)
Quantifies lip opening distance and temporal motion velocity to catch candidate talking.

$$\text{MAR} = \frac{\|P_2 - P_8\|_2 + \|P_3 - P_7\|_2}{2 \cdot \|P_1 - P_5\|_2}, \qquad E_{\text{mouth}} = \frac{1}{K} \sum_{k=0}^{K-1} |\text{MAR}(t-k) - \text{MAR}(t-k-1)|$$

---

### 8. False Positive Rate (FPR)
Measures false alarm frequency during normal, benign candidate behavior (e.g., self-reading question prompts).

$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

---

### 9. Real-Time Processing Speed (Frames Per Second - FPS)
Evaluates live video processing throughput on common laboratory computer hardware.

$$\text{FPS} = \frac{N_{\text{frames}}}{\sum_{i=1}^{N_{\text{frames}}} t_{\text{inference}}^{(i)}}$$

*(Target: $\ge 25\text{ FPS}$ for smooth real-time supervision)*
