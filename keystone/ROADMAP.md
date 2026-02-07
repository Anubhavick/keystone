# 🗺️ Keystone Project Roadmap

## 🚀 Project Status
**Overall Progress**: ~90% Complete

---

## ✅ Completed Milestones

### 1. Infrastructure & Setup
- [x] Project directory structure created (`frontend`, `backend`, `data`).
- [x] Virtual environment and dependencies configured.
- [x] GitHub Models integration (GPT-4o) for high-quality inference.
- [x] Testing framework (`pytest`) set up.

### 2. Core Backend Engine
- [x] **Document Processor**: PDF/Text ingestion with custom robust chunking.
- [x] **Vector Store**: Semantic knowledge base using ChromaDB and SentenceTransformers.
- [x] **Claim Extractor**: Hybrid AI/Rule-based system to decompose text into atomic facts.
- [x] **Fact Verifier**: NLI-based (Natural Language Inference) engine to support/contradict claims.

### 3. Frontend Interface
- [x] **Streamlit Dashboard**: Fully interactive web UI.
- [x] **Visualizations**: Split-screen view, confidence metrics, and evidence citation cards.
- [x] **Export**: JSON/CSV reporting.

### 4. Data & Evaluation
- [x] **Dataset Support**: HaluEval and FEVER datasets downloaded.
- [x] **Benchmarking**: Integration scripts ready for running checks.

---

## 🚧 Upcoming / In Progress

### 5. Correction & Healing (Next Step)
- [ ] **Correction Engine**: 
    - Automatically rewrite "Contradicted" claims to align with the evidence.
    - Transform the system from a passive checker to an active fixer.

### 6. System Refinement
- [ ] **Full Evaluation Run**: Execute verified benchmarks on 20k HaluEval examples.
- [ ] **Confidence Calibration**: Fine-tune trust scores based on NLI probabilities.
- [ ] **Performance Optimization**: Optimize vector retrieval latency.

---

## 🔮 Future Vision
- [ ] **Browser Extension**: Real-time checking of web content.
- [ ] **Multi-Modal Support**: Fact-checking images and charts.
- [ ] **API Deployment**: Expose Keystone as a REST API for other apps.
