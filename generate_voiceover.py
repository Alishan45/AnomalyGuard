"""
Generate neural voiceover audio for each scene using edge-tts with robust retry logic.
"""

import asyncio
import os
import edge_tts

VOICE = "en-US-AndrewNeural"

SCENES = {
    "scene1": (
        "Every single day, financial institutions process millions of electronic payment transactions. "
        "Detecting emerging fraud patterns while preventing crippling alert fatigue for human analysts is one of the toughest challenges in operational risk management. "
        "Welcome to AnomalyGuard: a reproducible, unsupervised machine learning pipeline and triage system built for payment anomaly detection and operational review."
    ),
    "scene2": (
        "Starting on Day 1 with exploratory data analysis, we inspected ten thousand multi-channel payment records across eight retail categories. "
        "Financial transaction amounts are notoriously right-skewed with long tails, while device authentication scores follow a Beta distribution. "
        "By understanding these natural distributions, we established our core review objective: isolating statistical rarities without introducing evaluation leakage."
    ),
    "scene3": (
        "On Day 2, we engineered critical domain behavioral features. Chief among them is physical travel velocity, which calculates geographic displacement over time to flag physically impossible travel exceeding 800 kilometers per hour. "
        "We stabilized variance with log transformations and normalized features with RobustScaler. "
        "We then built a transparent rule-based baseline using IQR fences and Z-scores, creating an operational benchmark that flagged 389 records."
    ),
    "scene4": (
        "On Day 3, we deployed our core model: an unsupervised Isolation Forest with 150 isolation trees. "
        "By randomly partitioning the feature space, anomalous records require far fewer splits to isolate. "
        "We calibrated raw decision functions into normalized anomaly scores between 0 and 1, automatically generating feature deviation notes to explain to human analysts exactly why each transaction was flagged."
    ),
    "scene5": (
        "On Day 4 and Day 5, we quantified operational trade-offs across contamination percentiles. "
        "In financial operations, lowering thresholds increases fraud recall but burdens review teams with false positives. "
        "At a 2.5% contamination rate, Isolation Forest achieved an outstanding F1-score of 0.9840 with 98.4% precision and recall, significantly outperforming both the rule baseline and Local Outlier Factor."
    ),
    "scene6": (
        "Finally, we deployed the entire pipeline as a production-grade Streamlit web application. "
        "Analysts can inspect distributions, tune contamination thresholds dynamically, review top risk drivers, and export review queues in real time. "
        "AnomalyGuard transforms complex unsupervised AI into actionable, transparent operational intelligence. Thank you for watching."
    )
}

async def generate_single(name, text):
    out_path = f"demo_video/voiceover/{name}.mp3"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"Already exists: {out_path} ({os.path.getsize(out_path)} bytes)")
        return
        
    for attempt in range(1, 5):
        try:
            print(f"Generating {out_path} (Attempt {attempt})...")
            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(out_path)
            print(f"Success: {out_path} ({os.path.getsize(out_path)} bytes)")
            await asyncio.sleep(1.0)
            return
        except Exception as e:
            print(f"Error on {out_path}: {e}. Retrying in {attempt * 2}s...")
            await asyncio.sleep(attempt * 2.0)
            
    print(f"Failed to generate {out_path} after 4 attempts.")

async def generate_all():
    os.makedirs("demo_video/voiceover", exist_ok=True)
    for name, text in SCENES.items():
        await generate_single(name, text)

if __name__ == "__main__":
    asyncio.run(generate_all())
