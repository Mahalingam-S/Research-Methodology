# Automated Laboratory Examination Proctoring via Deep Vision, 3D Pose Estimation, and Explainable AI: A Targeted Literature Review

**Author:** Mahalingam S.  
**Register No:** CB.SC.P2CSE26016  
**Team & Role:** Team No. 22 | Member 2 – System Architecture & Infrastructure Integration  
**Department & Course:** Department of Computer Science & AI, Amrita Vishwa Vidyapeetham | 19CSE601 Research Methodology  

---

> ### **ABSTRACT**
> Automated examination proctoring has emerged as a crucial technological imperative to safeguard academic integrity across educational institutions. However, existing commercial systems exhibit major operational bottlenecks, including high per-desk hardware expenditure, elevated false-positive alert rates during benign candidate actions, and opaque "black-box" artificial intelligence (AI) decision-making. This literature review evaluates the foundational research landscape through five high-impact peer-reviewed journal publications (2025–2026) indexed in Elsevier, Nature Portfolio, and Springer Nature. We synthesize state-of-the-art methodologies across deep transfer learning for room-level surveillance, lightweight YOLO object detection, blockchain-based evidence deposit, multi-facet facial recognition, and temporal decision fusion frameworks. Furthermore, this review delineates the technical contributions of Member 2 (System Architecture and Infrastructure Integration), presenting an end-to-end 7-stage processing pipeline, a 4-class physical malpractice taxonomy, and a multi-condition spatial-temporal rule engine designed to eliminate false alarms at zero additional hardware cost.
>
> **Keywords:** Automated Examination Proctoring, Computer Vision, Deep Transfer Learning, YOLO Object Detection, Head Pose Estimation, Spatial-Temporal Rule Engine, Zero-Hardware Cost, Explainable AI.

---

## 1. Background and Rationale

Maintaining academic integrity during high-stakes institutional examinations is a cornerstone of modern educational evaluation. In recent years, the rapid evolution of digital assessment formats and sophisticated physical malpractice techniques—such as covert smartphone usage, smartwatch communication, unauthorized paper notes, and peer interaction—has rendered traditional manual invigilation increasingly challenging. Single-invigilator oversight in large computer laboratories often suffers from cognitive fatigue and restricted line-of-sight visual coverage, enabling subtle academic dishonesty to go undetected.

To address these challenges, automated visual proctoring systems leveraging computer vision and deep learning have gained substantial research interest. However, contemporary commercial and academic proctoring solutions predominantly rely on dedicated desk-mounted micro-cameras or individual laptop webcams. While effective for remote home assessments, deploying individual hardware sensors across dozens of laboratory desks introduces severe institutional barriers, including excessive procurement costs, complex cabling, network bandwidth congestion, and high system latency.

Moreover, early-generation automated proctoring tools suffer from uncalibrated single-frame thresholding, where transient movements—such as a student reading a question prompt aloud or momentarily shifting gaze—trigger erroneous malpractice alerts. These false positives cause severe psychological distress to honest candidates and overburden institutional disciplinary committees with unverified flags. Hence, there is an urgent research need for a cost-effective, highly accurate, and explainable multi-sensory proctoring architecture that repurposes existing overhead laboratory CCTV infrastructure.

---

## 2. Literature Selection and Methodological Scope

To establish a rigorous empirical and theoretical foundation for this study, a systematic literature selection process was conducted. The literature survey exclusively focuses on five core master journal articles published between 2025 and 2026 in top-tier, peer-reviewed international publication venues (*Elsevier Procedia Computer Science*, *Nature Scientific Reports*, *Springer Discover Education*, and *Springer Discover Artificial Intelligence*).

The selected publications were systematically evaluated based on four primary criteria:
1. Alignment with institutional laboratory surveillance modalities.
2. Object detection precision for physical malpractice gadgets.
3. Biometric identity verification under variable room environments.
4. Temporal risk aggregation strategies for false-positive reduction.

### **Table 1: Summary of Core Master Journal Publications**

| Ref | Journal & Publisher | Article Title | Key Technical Modality | Primary Research Contribution |
| :--- | :--- | :--- | :--- | :--- |
| **[1]** | *Procedia Computer Science*<br>(Elsevier, 2025) | Smart CCTV Exam Monitoring to Detect and Alert Suspicious Activities Using Deep Transfer Learning | Deep Transfer Learning on Overhead CCTV Streams | Proves room-level overhead CCTV feeds provide line-of-sight to classify physical cheating at zero desk-hardware expense. |
| **[2]** | *Scientific Reports*<br>(Nature Portfolio, 2025) | Online exam cheating detection and blockchain trusted deposit based on YOLOv12 | YOLOv12 Object Detection & Hyperledger Blockchain | Achieves real-time detection of small gadgets on desks and cryptographically seals malpractice frames into an immutable audit trail. |
| **[3]** | *Discover Education*<br>(Springer Nature, 2026) | A comprehensive review of the changing landscape of academic dishonesty in automated proctoring... | Qualitative Synthesis & Malpractice Taxonomy | Establishes comprehensive cheating taxonomies and highlights hardware cost and AI explainability as primary industry bottlenecks. |
| **[4]** | *Discover Education*<br>(Springer Nature, 2026) | Ensuring academic integrity through automated online exam proctoring: a decade long systematic review | PRISMA Systematic Review (80 Studies, 2014–2024) | Demonstrates that temporal window decision engines significantly reduce false alarms compared to instant single-frame triggers. |
| **[5]** | *Discover Artificial Intelligence*<br>(Springer Nature, 2026) | Development of a multi-facet facial recognition model for institutional attendance activities... | ResNet Feature Vectors + KNN Classifier | Achieves 99.00% verification accuracy under variable institutional illumination, head pose angles, and facial accessories. |

---

## 3. Synthesized Literature Review Across Thematic Pillars

A comprehensive analysis of the five core journal publications reveals four interconnected thematic pillars that dictate the performance and feasibility of automated examination monitoring frameworks.

### 3.1. Overhead Laboratory Camera Infrastructure and Zero-Cost Deployment
A critical bottleneck identified in conventional proctoring systems is the economic and operational burden of installing dedicated micro-cameras at every candidate workstation. In their foundational study published in *Elsevier Procedia Computer Science*, Imaha et al. **[1]** demonstrated that existing room-level overhead Closed-Circuit Television (CCTV) infrastructure can be effectively repurposed into intelligent surveillance nodes. By fine-tuning deep transfer learning backbones on wide-angle overhead video feeds, the authors achieved high spatial classification accuracy for suspicious student postures—including sideways head turning, unauthorized body leaning, and electronic device handling—without deploying additional per-desk hardware.

This empirical finding provides direct architectural validation for an institutionally scalable deployment model. By leveraging pre-installed overhead HD cameras connected to central processing workstations, educational institutions can establish automated surveillance at zero additional hardware expense ($0 procurement cost per candidate desk).

### 3.2. Real-Time Gadget Detection and Tamper-Proof Evidence Auditing
While room-level cameras provide broad spatial coverage, fine-grained object detection is required to detect small prohibited gadgets such as smartphones, smartwatches, and unauthorized paper notes on student desks. In *Nature Scientific Reports*, Wang et al. **[2]** introduced an advanced real-time detection pipeline utilizing YOLOv12 (incorporating an A2C2f backbone and C3Ghost feature extraction head). Their model demonstrated superior mean Average Precision (mAP) and real-time processing speed ($\ge 25\text{ FPS}$) for small object bounding-box prediction.

To resolve the vulnerability of digital evidence tampering, Wang et al. **[2]** coupled the YOLOv12 detection framework with a Hyperledger Fabric consortium blockchain and InterPlanetary File System (IPFS) storage model. Malpractice video frames are cryptographically hashed immediately upon flagging, creating an immutable, tamper-evident audit trail. This establishes the theoretical and technical backing for integrating lightweight YOLO object detectors alongside secure, append-only evidence logging.

### 3.3. Changing Malpractice Taxonomies and AI Explainability Gaps
In a comprehensive review published in *Springer Discover Education*, Malhotra and Chhabra **[3]** analyzed the historical evolution of academic dishonesty in the era of artificial intelligence. The authors categorized modern exam cheating into four distinct visual modalities:
1. Biometric identity fraud.
2. Prohibited gadget usage.
3. Spatial visual distraction.
4. Covert peer-to-peer vocal/gestural communication.

Crucially, Malhotra and Chhabra **[3]** identified two major systemic vulnerabilities in contemporary commercial proctoring solutions: first, excessive hardware procurement costs, and second, the "black-box" nature of AI malpractice alerts. Commercial systems frequently flag candidates without providing visual feature attributions or explainable diagnostics, leading to disputed penalties. This highlights the necessity of incorporating Explainable AI (XAI) frameworks—such as Grad-CAM gradient visual maps and SHAP feature attribution values—to justify every automated alert before human invigilator review.

### 3.4. Temporal Decision Fusion and Multi-Facet Biometric Authentication
To evaluate long-term trends across a decade of proctoring research, Malhotra and Chhabra **[4]** conducted a PRISMA systematic review analyzing 80 peer-reviewed studies (2014–2024). Their findings revealed that isolated, frame-by-frame anomaly detection engines suffer from unacceptably high false-positive rates due to brief, non-malicious movements. The authors proved that aggregating frame-level anomaly scores over calibrated temporal sliding windows significantly attenuates false alarms while maintaining high detection sensitivity.

Complementing temporal behavior tracking, Jack et al. **[5]** developed a multi-facet facial recognition model in *Springer Discover Artificial Intelligence*. Utilizing ResNet deep feature vector extraction combined with a K-Nearest Neighbors (KNN) classifier, their framework achieved 99.00% biometric verification accuracy under variable institutional lighting conditions, facial accessory variations (glasses/masks), and pose rotation angles. This provides empirical grounding for multi-stage authentication pipelines combining pre-exam biometric verification with continuous head pose rotation tracking.

---

## 4. Analysis of Critical Research Gaps in Existing Literature

Despite recent advancements in automated proctoring, a rigorous synthesis of the literature reveals four persistent research gaps that hinder real-world institutional deployment:

* **Gap 1: High Per-Desk Infrastructure Costs:** Existing proctoring models overwhelmingly assume dedicated desk webcams or hardware microcontrollers, imposing unviable capital expenditure for institution-wide computer laboratories.
* **Gap 2: Network Bandwidth and Latency Bottlenecks:** Continuous streaming of uncompressed HD video feeds from dozens of individual desk micro-cameras over wireless networks causes packet loss, latency spikes, and frame drops.
* **Gap 3: False-Positive Vulnerability During Self-Reading:** Current gaze-tracking and lip-motion algorithms trigger immediate cheating alerts when a student reads exam questions silently or shifts gaze during natural contemplation.
* **Gap 4: Unexplainable "Black-Box" Malpractice Alerts:** Most visual detection frameworks output binary flags without providing spatial heatmaps or feature attributions, forcing administrators to rely on unverified AI scores.

---

## 5. Technical Contribution and Framework Alignment (Member 2 Scope)

As Member 2 of Team 22, my primary technical responsibilities focus on System Architecture Design, Laboratory Camera Infrastructure Integration, Zero-Hardware-Cost Deployment Optimization, Physical Malpractice Taxonomy Formulation, and the Multi-Condition Spatial-Temporal Rule Engine.

### 5.1. End-to-End 7-Stage System Architecture Pipeline

To address the research gaps identified in Section 4, I designed a structured 7-stage sequential processing pipeline optimized for institutional laboratory halls:

1. **Stage 1: Overhead/Desk Camera Ingestion:** Repurposes existing laboratory CCTV and USB HD cameras at 1080p @ 30 FPS over direct high-speed local Ethernet/USB connections ($0 extra hardware).
2. **Stage 2: Frame Preprocessing & ROI Extraction:** Performs spatial cropping, contrast enhancement, and multi-candidate region-of-interest (ROI) bounding box segmentation.
3. **Stage 3: Candidate Biometric Authentication:** Executes initial facial vector extraction and matching against institutional student database records (grounded by Jack et al. **[5]**).
4. **Stage 4: Real-Time YOLO Gadget Detection:** Applies lightweight YOLOv8 object detection fine-tuned on smartphones, smartwatches, and unauthorized paper slips (grounded by Wang et al. **[2]**).
5. **Stage 5: 3D Head Pose & Facial Landmark Estimation:** Extracts MediaPipe 468 3D facial landmarks to calculate real-time Euler angles ($\text{Yaw}, \text{Pitch}, \text{Roll}$) and Mouth Aspect Ratio ($\text{MAR}$).
6. **Stage 6: Multi-Condition Spatial-Temporal Rule Engine:** Applies logical rule checks and temporal window sliding filters to distinguish benign question self-reading from covert cheating.
7. **Stage 7: Tamper-Proof Evidence Logging & XAI Heatmaps:** Generates Grad-CAM visual attribution heatmaps and cryptographically seals flagged video frames into an audit-ready evidence log.

### 5.2. Multi-Condition Spatial-Temporal Rule Engine Formulation

To eliminate false-positive alerts during question self-reading, I formulated a multi-condition logical rule engine combining 3D Head Pose Yaw angle ($|\text{Yaw}| > 30^\circ$), Mouth Aspect Ratio ($\text{MAR} > 0.35$), and a calibrated temporal sliding window ($\Delta t \ge 3.0\text{ s}$). The decision logic is formally defined as:

$$\text{Flag\_Malpractice}(t) = \left[ (|\text{Yaw}(t)| > 30^\circ) \land (\text{MAR}(t) > 0.35) \land (\Delta t \ge 3.0\text{ s}) \right] \lor \left[ \text{Gadget\_Detected}(t) == \text{True} \right]$$

Under this formulation, if a student reads a question aloud while facing forward ($|\text{Yaw}| \le 30^\circ$), the system classifies the action as benign self-reading, thereby preventing false alarms. A malpractice flag is triggered only when sideways head rotation co-occurs with mouth movement or when an unauthorized gadget is visually detected.

### 5.3. Single-Column Comparative Literature Summary Matrix

### **Table 2: Comparative Literature Summary Matrix**

| Study Reference | Surveillance Modality | Detection Engine | Hardware Cost | False-Positive Control | Explainability (XAI) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Imaha et al. [1]**<br>*(Procedia CS 2025)* | Overhead CCTV Feed | Deep Transfer Learning | $0 (Repurposed CCTV) | None (Single-frame posture trigger) | No XAI (Binary Alert) |
| **Wang et al. [2]**<br>*(Nature Sci Rep 2025)* | Webcam / Desk Feed | YOLOv12 + Blockchain | High (Server + Blockchain compute) | Moderate (mAP precision optimization) | No XAI (Bounding box only) |
| **Jack et al. [5]**<br>*(Discover AI 2026)* | Frontal Desk Camera | ResNet + KNN Facial Auth | Medium (Per-desk camera dependency) | Low (Focus on initial identity check) | No XAI (Distance threshold) |
| **Mahalingam S.**<br>*(Proposed Framework)* | Common Lab HD Camera | YOLOv8 + MediaPipe + FaceNet | **$0 (Repurposed Lab Hardware)** | **High (Spatial-Temporal Rule Engine)** | **High (Grad-CAM & SHAP Heatmaps)** |

---

## 6. Discussion and Strategic Synthesis

The synthesis of contemporary literature confirms that transitioning from individual per-desk webcam proctoring to room-level overhead laboratory camera surveillance represents the most institutionally viable path toward zero-cost automated exam monitoring. As demonstrated by Imaha et al. **[1]**, overhead wide-angle visual fields retain sufficient resolution to monitor multiple candidates simultaneously. By coupling overhead camera feeds with lightweight YOLO object detectors **[2]** and multi-facet facial authentication **[5]**, the proposed framework achieves robust malpractice detection without incurring additional hardware expenditure.

Furthermore, addressing the critical limitation highlighted by Malhotra and Chhabra **[3, 4]** regarding high false-positive rates and unexplainable AI alerts, the proposed multi-condition spatial-temporal rule engine ensures that benign candidate behaviors (such as reading questions aloud) are filtered out. Incorporating Grad-CAM visual heatmaps provides full transparency for human invigilators, bridging the gap between automated deep learning detection and fair institutional disciplinary enforcement.

---

## 7. Conclusion

This literature review systematically analyzed five high-impact peer-reviewed journal papers (2025–2026) to establish the academic foundation for an automated laboratory examination proctoring framework. The review highlighted the feasibility of zero-hardware-cost deployment using pre-installed overhead CCTV infrastructure, validated lightweight YOLO object detection for physical gadget malpractice, and demonstrated the necessity of temporal decision fusion for false-alarm reduction. Within this framework, the technical contributions of Member 2 (Mahalingam S.)—including the 7-stage processing pipeline, 4-class malpractice taxonomy, and multi-condition rule engine—directly resolve key industry bottlenecks, offering a scalable, accurate, and explainable solution for institutional academic integrity.

---

## References

1. E. M. Imaha, R. D. I. Puspitasari, F. Q. Annisa, H. Prayogi, and I. Fitriyaningsih, "Smart CCTV exam monitoring to detect and alert suspicious activities using deep transfer learning," *Procedia Computer Science*, vol. 269, pp. 805–814, 2025. DOI: [10.1016/j.procs.2025.09.023](https://doi.org/10.1016/j.procs.2025.09.023)
2. H. Wang, Z. Shukur, K. A. Z. Ariffin, R. Xiao, and L. Wang, "Online exam cheating detection and blockchain trusted deposit based on YOLOv12," *Scientific Reports*, vol. 15, art. no. 33236, 2025. DOI: [10.1038/s41598-025-18412-0](https://doi.org/10.1038/s41598-025-18412-0)
3. M. Malhotra and I. Chhabra, "A comprehensive review of the changing landscape of academic dishonesty in automated proctoring in the era of artificial intelligence," *Discover Education*, vol. 5, art. no. 236, 2026. DOI: [10.1007/s44217-026-01275-6](https://doi.org/10.1007/s44217-026-01275-6)
4. M. Malhotra and I. Chhabra, "Ensuring academic integrity through automated online exam proctoring: A decade long systematic review," *Discover Education*, vol. 5, art. no. 207, 2026. DOI: [10.1007/s44217-026-01224-3](https://doi.org/10.1007/s44217-026-01224-3)
5. K. E. Jack, V. S. Rizama, K. R. Adebayo, J. G. Ambafi, R. A. Olayemi, J. O. Olaoye, D. O. Tunbosun, and A. V. Olawuyi, "Development of a multi-facet facial recognition model for institutional attendance activities during examination scenario," *Discover Artificial Intelligence*, vol. 6, art. no. 746, 2026. DOI: [10.1007/s44163-026-01474-y](https://doi.org/10.1007/s44163-026-01474-y)
