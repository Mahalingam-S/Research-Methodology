# Automated Examination Proctoring via Deep Vision, 3D Pose Estimation & Spatial-Temporal AI

**Author:** Mahalingam S.  
**Register No:** CB.SC.P2CSE26016  
**Course & Dept:** 19CSE601 Research Methodology | Department of Computer Science & AI, Amrita Vishwa Vidyapeetham  
**Team & Role:** Team No. 22 | Member 2 – System Architecture & Infrastructure Integration  

---

## 📌 Executive Summary

This repository contains the complete **Research Methodology** portfolio, literature survey, algorithm analysis notebooks, slides, and architectural documentation authored by **Mahalingam S.** for automated laboratory examination proctoring.

The research addresses core bottlenecks in contemporary assessment surveillance—specifically **high per-desk hardware expenditure**, **elevated false-positive alert rates during benign candidate actions**, and **opaque "black-box" AI decisions**. By synthesizing state-of-the-art methodologies across **over-head CCTV deep transfer learning**, **lightweight YOLO object detection**, **3D head pose estimation**, and **spatial-temporal rule engines**, this work establishes a zero-hardware-cost institutional proctoring framework.

---

## 📄 Key Research Documents & Deliverables

| Category | File / Artifact | Description |
| :--- | :--- | :--- |
| **Journal Format Literature Review** | [`Mahalingam_S_Literature_Review_Journal_Format.md`](./Mahalingam_S_Literature_Review_Journal_Format.md)<br>[`Mahalingam_S_Literature_Review_Journal_Format.docx`](./Mahalingam_S_Literature_Review_Journal_Format.docx) | Comprehensive 7-stage literature review analyzing 5 high-impact master journal papers (Elsevier, Nature Portfolio, Springer, 2025–2026). |
| **Algorithm & AI Relevance Notebook** | [`Mahalingam_S_Algorithm_and_AI_Relevance_Notebook.md`](./Mahalingam_S_Algorithm_and_AI_Relevance_Notebook.md)<br>[`Mahalingam_S_Algorithm_and_AI_Relevance_Notebook.docx`](./Mahalingam_S_Algorithm_and_AI_Relevance_Notebook.docx) | Deep-dive mathematical formulations of ResNet+LSTM temporal talking detection, Mouth Aspect Ratio (MAR), YOLOv12 detection, 3D Perspective-n-Point pose estimation, and Spatial-Temporal Rule Engines. |
| **Contributions & Literature Survey** | [`Mahalingam_S_Contributions_and_Literature_Survey.docx`](./Mahalingam_S_Contributions_and_Literature_Survey.docx) | Detailed breakdown of Member 2 technical contributions, malpractice taxonomies, and research gap resolution. |
| **Presentation Slides** | [`Mahalingam_S_Literature_Review.pptx`](./Mahalingam_S_Literature_Review.pptx)<br>[`Member_2_Mahalingam_Review_Presentation.pptx`](./Member_2_Mahalingam_Review_Presentation.pptx) | Structured presentation decks summarizing literature findings, system pipeline, and research methodology. |
| **Personal Study & Reading Guides** | [`Mahalingam_3_Papers_Reading_Guide.txt`](./Mahalingam_3_Papers_Reading_Guide.txt)<br>[`Mahalingam_5_Papers_Printout_Summary.docx`](./Mahalingam_5_Papers_Printout_Summary.docx)<br>[`Mahalingam_Personal_Study_Guide.txt`](./Mahalingam_Personal_Study_Guide.txt)<br>[`Mahalingam_Paper_Download_Links.txt`](./Mahalingam_Paper_Download_Links.txt) | Individual study notes, reading guides, download links, and paper summaries prepared during literature analysis. |
| **Core Research Papers & Reference** | [`Automated_Lab_Camera_Examination_Proctoring_Research_Paper.docx`](./Automated_Lab_Camera_Examination_Proctoring_Research_Paper.docx)<br>[`Project_Introduction_Document.md`](./Project_Introduction_Document.md)<br>[`Technical_Architecture_and_Features_Guide.docx`](./Technical_Architecture_and_Features_Guide.docx)<br>[`Pace - 1/`](./Pace%20-%201/) | Full research paper draft, project introduction, technical architecture guide, and original benchmark PDFs (Papers 1–5). |

---

## 📚 Master Journal Publications Reviewed (2025–2026)

| Ref | Journal & Publisher | Title | Key Modality | Primary Contribution |
| :---: | :--- | :--- | :--- | :--- |
| **[1]** | *Procedia Computer Science*<br>(Elsevier, 2025) | Smart CCTV Exam Monitoring to Detect and Alert Suspicious Activities Using Deep Transfer Learning | Deep Transfer Learning on Overhead CCTV Streams | Proves room-level overhead CCTV feeds provide line-of-sight to classify physical cheating at zero desk-hardware expense. |
| **[2]** | *Scientific Reports*<br>(Nature Portfolio, 2025) | Online exam cheating detection and blockchain trusted deposit based on YOLOv12 | YOLOv12 Object Detection & Hyperledger Blockchain | Real-time detection of small gadgets on desks and cryptographically seals malpractice frames into an immutable audit trail. |
| **[3]** | *Discover Education*<br>(Springer Nature, 2026) | A comprehensive review of the changing landscape of academic dishonesty in automated proctoring... | Qualitative Synthesis & Malpractice Taxonomy | Establishes comprehensive cheating taxonomies and highlights hardware cost and AI explainability as primary industry bottlenecks. |
| **[4]** | *Discover Education*<br>(Springer Nature, 2026) | Ensuring academic integrity through automated online exam proctoring: a decade long systematic review | PRISMA Systematic Review (80 Studies, 2014–2024) | Demonstrates that temporal window decision engines significantly reduce false alarms compared to instant single-frame triggers. |
| **[5]** | *Discover Artificial Intelligence*<br>(Springer Nature, 2026) | Development of a multi-facet facial recognition model for institutional attendance activities... | ResNet Feature Vectors + KNN Classifier | Achieves 99.00% verification accuracy under variable institutional illumination, head pose angles, and facial accessories. |

---

## 🏗️ Technical Architecture & Key Contributions

### 1. 7-Stage End-to-End Processing Pipeline
1. **Stage 1: Overhead/Desk Camera Ingestion:** Repurposes existing overhead laboratory CCTV ($0 extra hardware).
2. **Stage 2: Frame Preprocessing & ROI Extraction:** Spatial cropping and candidate bounding box segmentation.
3. **Stage 3: Candidate Biometric Authentication:** ResNet facial feature matching against institutional database.
4. **Stage 4: Head Pose & Visual Gaze Vectoring:** 3D Perspective-n-Point pose tracking ($\text{Yaw} > \pm 35^\circ$).
5. **Stage 5: Mouth Aspect Ratio (MAR) & Lip Motion:** Geometric lip distance and speech energy tracking.
6. **Stage 6: Prohibited Gadget Detection:** Lightweight YOLO object detector for phones, smartwatches, and notes.
7. **Stage 7: Spatial-Temporal Rule Engine & XAI:** Multi-frame temporal window decision engine with Grad-CAM visual heatmaps.

### 2. Spatial-Temporal False Alarm Elimination Rule Engine
Combines head pose rotation, lip aspect ratio, motion velocity, gadget detection scores, and temporal persistence sliding windows ($\Delta T$) to distinguish silent question reading from actual academic dishonesty.

$$\text{MalpracticeFlag}(t) = \begin{cases} \text{TRUE} & \text{if } \sum_{\tau = t - \Delta T}^{t} \mathcal{S}(\tau) \ge \Theta_{\text{malpractice}} \\ \text{FALSE} & \text{otherwise (Benign question reading / minor movement)} \end{cases}$$

---

## 🛠️ Repository Scope & Disclaimer

This repository is maintained as a personal academic showcase of **Mahalingam S.**'s research methodology contributions. 
- **Omitted Content:** Implementation source codes, model training pipelines, scratch test scripts, raw execution logs, and non-author team materials have been omitted to maintain clean research focus.
- **Included Content:** Full literature review manuscripts, AI relevance mathematical notebooks, research presentations, paper reading guides, and architectural design documentation.

---

## 👨‍💻 Author

* **Mahalingam S.**  
* Department of Computer Science & AI  
* Amrita Vishwa Vidyapeetham  
* GitHub: [@Mahalingam-S](https://github.com/Mahalingam-S)
