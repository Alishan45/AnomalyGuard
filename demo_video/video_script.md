# AnomalyGuard — Professional Video Demonstration Script
**Duration:** ~2 minutes 15 seconds  
**Narrator Tone:** Professional, authoritative, clear FinTech engineer voice  
**Format:** 1920x1080 (16:9), 30 FPS  

---

### Scene 1: Introduction & The Real-World Challenge (0:00 - 0:25)
- **Visual:** Cinematic dark tech intro with animated glowing FinTech transaction telemetry, transitioning into the AnomalyGuard title card and the critical operational notice: *"An anomaly is a review signal, not proof of fraud."*
- **Voiceover Narration:**  
  *"Every single day, financial institutions process millions of electronic payment transactions. Detecting emerging fraud patterns while preventing crippling alert fatigue for human analysts is one of the toughest challenges in operational risk management. Welcome to AnomalyGuard: a reproducible, unsupervised machine learning pipeline and triage system built for payment anomaly detection and operational review."*

---

### Scene 2: Dataset Architecture & Exploratory Analysis (0:25 - 0:50)
- **Visual:** High-resolution exploratory distribution dashboard showing transaction amount log-skew, Beta-distributed device risk scores, and merchant categorical volumes.
- **Voiceover Narration:**  
  *"Starting on Day 1 with exploratory data analysis, we inspected ten thousand multi-channel payment records across eight retail categories. Financial transaction amounts are notoriously right-skewed with long tails, while device authentication scores follow a Beta distribution. By understanding these natural distributions, we established our core review objective: isolating statistical rarities without introducing evaluation leakage."*

---

### Scene 3: Preprocessing, Travel Velocity & Baseline Benchmark (0:50 - 1:15)
- **Visual:** Feature engineering schematics highlighting Physical Travel Velocity ($\Delta d / \Delta t$) with a 800 km/h cutoff indicator, followed by the statistical IQR and Z-score rule triggers.
- **Voiceover Narration:**  
  *"On Day 2, we engineered critical domain behavioral features. Chief among them is physical travel velocity, which calculates geographic displacement over time to flag physically impossible travel exceeding 800 kilometers per hour. We stabilized variance with log transformations and normalized features with RobustScaler. We then built a transparent rule-based baseline using IQR fences and Z-scores, creating an operational benchmark that flagged 389 records."*

---

### Scene 4: Isolation Forest & Score Calibration (1:15 - 1:40)
- **Visual:** Isolation Forest calibrated score distribution curve, 2D multivariate scatter plot of amount vs distance, and ranked top anomaly inspection with human review drivers.
- **Voiceover Narration:**  
  *"On Day 3, we deployed our core model: an unsupervised Isolation Forest with 150 isolation trees. By randomly partitioning the feature space, anomalous records require far fewer splits to isolate. We calibrated raw decision functions into normalized anomaly scores between 0 and 1, automatically generating feature deviation notes to explain to human analysts exactly why each transaction was flagged."*

---

### Scene 5: Threshold Sensitivity & Evaluation Trade-offs (1:40 - 2:05)
- **Visual:** Sensitivity analysis curve across percentiles, confusion matrix, and model comparison chart demonstrating Isolation Forest's 0.9840 F1-Score vs LOF and Baseline.
- **Voiceover Narration:**  
  *"On Day 4 and Day 5, we quantified operational trade-offs across contamination percentiles. In financial operations, lowering thresholds increases fraud recall but burdens review teams with false positives. At a 2.5% contamination rate, Isolation Forest achieved an outstanding F1-score of 0.9840 with 98.4% precision and recall, significantly outperforming both the rule baseline and Local Outlier Factor."*

---

### Scene 6: Interactive Streamlit Dashboard & Conclusion (2:05 - 2:25)
- **Visual:** Live interactive Streamlit web dashboard in action: switching tabs, adjusting sensitivity sliders, CSV export, single transaction deep-dive, and closing call to action.
- **Voiceover Narration:**  
  *"Finally, we deployed the entire pipeline as a production-grade Streamlit web application. Analysts can inspect distributions, tune contamination thresholds dynamically, review top risk drivers, and export review queues in real time. AnomalyGuard transforms complex unsupervised AI into actionable, transparent operational intelligence. Thank you for watching."*
