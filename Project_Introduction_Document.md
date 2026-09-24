# INTRODUCTION

## An AI Framework for Automated Examination Monitoring Using Facial Recognition, Object Detection, and Spatial-Temporal Behavior Analysis

**Course:** 19CSE601 – Research Methodology  
**Department:** Department of Computer Science & AI, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore  
**Authors / Team No. 22:**  
- **Sibichandaru C.** (Member 1 – Biometric Authentication & Identity Management)  
- **Mahalingam S.** (Member 2 – System Architecture & Overhead CCTV Vision Pipeline)  
- **Vidyasagar S.** (Member 3 – Multimodal Temporal Risk Fusion & Blockchain Evidence Log)  
**Supervisor / Guide:** Dr. Senthil Kumar Thangavel  

---

### 1. Background and Research Motivation

In higher education institutions, maintaining academic integrity during laboratory assessments and computer-based examinations is a fundamental requirement. Traditional invigilation methods rely almost entirely on human proctors physically monitoring examination halls. However, in large laboratory environments with dozens of computer workstations, manual supervision exhibits severe operational limitations:
1. **Invigilator Fatigue and Visual Blind Spots:** Human invigilators experience cognitive fatigue over prolonged exam durations, making it impossible to continuously monitor all candidates simultaneously.
2. **Covert Malpractice:** Modern cheating techniques have evolved beyond simple paper notes to include covert electronic gadgets (smartphones, smartwatches, micro-earpieces) and subtle peer-to-peer visual signaling.
3. **Scalability Bottlenecks:** Maintaining a strict student-to-proctor ratio requires substantial manpower and operational expenditure, which scales poorly during large-scale university examinations.

To overcome these human limitations, automated proctoring systems powered by Computer Vision (CV) and Artificial Intelligence (AI) have emerged as a vital area of research.

---

### 2. Problem Statement and Industry Bottlenecks

Despite significant advancements in commercial and academic AI proctoring tools, existing solutions suffer from three critical bottlenecks when deployed in institutional computer laboratories:

1. **High Hardware Procurement Costs ($ Per-Desk Equipment):**  
   Most commercial proctoring software requires dedicated, desk-mounted micro-webcams or dual-camera setups at every single workstation. Installing individual hardware across hundreds of university lab desks incurs prohibitive equipment costs, complex cabling, and high maintenance overhead.
2. **High False-Alarm Rates (Rigid Single-Frame Triggers):**  
   Conventional visual proctoring engines generate frequent false-positive alerts whenever a student shifts in their chair, stretches their neck, or silently reads an exam question to themselves (causing lip movements). These false alarms induce unnecessary anxiety for honest students and overload invigilators with unverified warning logs.
3. **Opaque "Black-Box" AI Decision-Making:**  
   Most deep learning models flag suspicious behavior without providing visual or quantitative explainability (Explainable AI - XAI). Disciplinary committees are left with binary "cheating / clean" scores without visual heatmaps or feature attributions justifying *why* an alert was generated.
4. **Evidence Vulnerability:**  
   Standard digital video logs stored on local hard drives remain susceptible to accidental corruption, unauthorized deletion, or post-exam evidence tampering.

---

### 3. Proposed Framework and Solution Architecture

To address these cost, accuracy, and transparency challenges, our research proposes a **low-cost, multi-layered AI framework for automated examination monitoring** designed specifically for institutional laboratory halls.

Our proposed system introduces five integrated technical components:

```
+-----------------------------------------------------------------------------------+
|                        PROPOSED 5-LAYER AI PROCTORING PIPELINE                     |
+-----------------------------------------------------------------------------------+
|  1. Candidate Authorization   --> ResNet + KNN Facial Recognition (Identity Check) |
|  2. Overhead CCTV Vision      --> $0 New Hardware Cost Ceiling CCTV Stream         |
|  3. Real-Time Object Detector --> YOLOv12 Small Gadget Detection (Phone/Notes)     |
|  4. False-Positive Rule Engine--> Head Pose (Yaw > 30°) + Mouth Aspect Ratio (MAR)|
|  5. Evidence Log Deposition   --> Immutable Blockchain & IPFS Cryptographic Hash   |
+-----------------------------------------------------------------------------------+
```

* **Zero Additional Hardware Expenditure ($0 Cost):**  
  Repurposes pre-installed overhead laboratory CCTV camera feeds to achieve wide-angle spatial coverage of multiple candidate workstations simultaneously.
* **Biometric Identity Verification:**  
  Employs deep feature vector extraction (ResNet backbone) coupled with K-Nearest Neighbors (KNN) classification to authenticate candidate identities at check-in and prevent student impersonation (achieving 99.00% verification accuracy).
* **Real-Time Gadget Detection:**  
  Integrates a lightweight **YOLOv12** object detection backbone fine-tuned to spot prohibited physical paraphernalia (smartphones, smartwatches, extra papers) on candidate desks at real-time processing speeds ($\ge 25\text{ FPS}$).
* **Spatial-Temporal Rule Engine for False-Alarm Mitigation:**  
  Couples **Head Pose Rotation (>30° Yaw)** with **Mouth Aspect Ratio (MAR)** temporal tracking to accurately differentiate benign self-reading from covert sideways peer whispering.
* **Blockchain Evidence Deposit:**  
  Cryptographically hashes flagged malpractice video clips onto a consortium **Hyperledger Fabric Blockchain & IPFS** log to guarantee an immutable, tamper-proof audit trail for institutional review committees.

---

### 4. Core Objectives of the Study

The main objectives of this research project are:
1. **To design and evaluate a zero-hardware-cost visual surveillance architecture** using existing overhead room CCTV streams.
2. **To implement a real-time object detection model (YOLOv12)** capable of accurately classifying small prohibited gadgets on lab desks.
3. **To develop a multi-condition spatial-temporal rule engine** combining head pose angle estimation and lip movement (MAR) to eliminate false alarms during question self-reading.
4. **To integrate an Explainable AI (XAI) attribution layer (Grad-CAM / SHAP)** that provides visual heatmap evidence for human proctor verification.
5. **To ensure complete evidence integrity** by logging cryptographically signed malpractice events on a distributed blockchain ledger.

---

### 5. Research Novelty and Contributions

The key novel contributions of our work include:
- **Economic Scalability:** Eliminates the need for per-desk micro-cameras, reducing institutional deployment expense to $0 in additional hardware.
- **Context-Aware Behavioral Analysis:** Replaces rigid single-frame alert triggers with a calibrated temporal risk scoring engine ($Risk_{Total} = 0.40 \cdot Auth + 0.40 \cdot Object + 0.20 \cdot Pose$).
- **Auditability and Transparency:** Combines visual XAI attributions with cryptographic blockchain verification to establish a legally defensible and fair proctoring workflow.

---

### 6. Outline of the Project Report

The remainder of this report is organized as follows:
- **Section 2 (Literature Review):** Synthesizes foundational research across 5 master journal papers (Elsevier, Nature, Springer).
- **Section 3 (Methodology & System Architecture):** Details the 7-stage processing pipeline, dataset preparation, and YOLOv12 model training.
- **Section 4 (Implementation & Experimental Results):** Presents empirical performance metrics, mAP scores, FPS evaluations, and false-positive reduction analysis.
- **Section 5 (Conclusion & Future Work):** Summarizes research findings and outlines future enhancements, including multimodal audio analytics and edge-device optimization.
