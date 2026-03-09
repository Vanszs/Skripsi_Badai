---
name: thesis-sentinel
description: "Specialized guardian for the Gede-Pangrango Thesis Project. Use this skill to validate model integrity (3 variables) and ensure documentation consistency. Hybrid persistence has been REMOVED from thesis scope."
---

# Thesis Sentinel Skill

This skill provides specialized knowledge and workflows for maintaining the integrity of the **"Nowcasting Probabilistik Dinamika Cuaca Mikro"** thesis project.

## 🎯 Core Responsibilities
When the user asks to "check", "audit", "validate", or "fix" the workspace, ALWAYS check these 3 pillars:

### 1. Model Configuration Integrity
The system MUST always adhere to the **3-Variable Configuration**:
- **Target Variables:** `['precipitation', 'wind_speed_10m', 'relative_humidity_2m']`
- **Exclusion:** `temperature_2m` MUST be a *feature* only, NEVER a *target*.
- **Nodes:** `PANGRANGO_NODES` (Puncak, Cibodas, Cianjur) - verify coordinates if suspicious.

### 2. ~~Hybrid Weight Calibration~~ — DIHAPUS
Hybrid persistence telah di-drop dari thesis. Thesis fokus pada **pure diffusion**
(Retrieval-Augmented Diffusion + Spatio-Temporal Graph Conditioning vs MLP Baseline).
`run_inference_hybrid()` di inference.py masih ada secara kode tetapi TIDAK digunakan.

*Jika user menyebut hybrid, ingatkan bahwa hybrid sudah dihapus dari scope thesis.*

### 3. Documentation Alignment
Ensure `docs/COMPREHENSIVE_DOCUMENTATION.md` matches the code:
- **Title:** Must match the 3D/Multi-Variable scope.
- **Scope:** "Algorithm Only" (No App).
- **Nuance:** "Dinamika Cuaca Mikro" (Predicting changes/risks).

## 🛠️ Automated Workflows

### A. Quick Health Check
Run this check to get a snapshot of the system status:
```bash
python .agent/skills/thesis-sentinel/scripts/health_check.py
```

### B. Full Performance Evaluation
To verify if metrics are still "READY":
```bash
python final_proven_eval.py
```

## 🚨 Critical Alerts (Stop Work if Found)
1. **SITARO References:** If you see "Siau", "Tagulandang", or "Biaro" in active code -> **PURGE IT**.
2. **4-Variable Model:** If `NUM_TARGETS = 4` -> **REFUSE TO PROCEED** until fixed to 3.
3. **App Development:** If user asks for GUI/Android App -> **REMIND SCOPE** (Algorithm Only).

---
*Created by Antigravity Agent for Bevan's Thesis.*
