# Fairness Evaluation Report: Qwen2.5-1.5B on adult

## 1. Overall Performance
- **Accuracy**: 0.7357
- **Precision**: 0.4437
- **Recall**: 0.2410
- **F1**: 0.3124

## 2. Fairness Analysis Table
| Analysis Level / Condition | DP Gap | EO Gap | Unfairness Score |
| :--- | :--- | :--- | :--- |
| **race** (Base) | 0.0718 | 0.1125 | **0.0922** |
| **gender** (Base) | 0.0511 | 0.1052 | **0.0782** |
| **age_mapped** (Base) | 0.0642 | 0.1641 | **0.1142** |
| **race & gender** (Base) | 0.0997 | 0.1719 | **0.1358** |
| race & gender: black | 0.0254 | 0.0576 | 0.0415 |
| race & gender: white | 0.0490 | 0.1032 | 0.0761 |
| race & gender: female | 0.0507 | 0.0686 | 0.0596 |
| race & gender: male | 0.0743 | 0.1142 | 0.0943 |
| **race & age_mapped** (Base) | 0.1289 | 0.2609 | **0.1949** |
| race & age_mapped: black | 0.0553 | 0.1490 | 0.1021 |
| race & age_mapped: white | 0.0666 | 0.1675 | 0.1170 |
| race & age_mapped: 61+ | 0.0623 | 0.0933 | 0.0778 |
| race & age_mapped: 24-40 | 0.0736 | 0.1118 | 0.0927 |
| race & age_mapped: 41-60 | 0.0728 | 0.1053 | 0.0890 |
| **gender & age_mapped** (Base) | 0.1423 | 0.3675 | **0.2549** |
| gender & age_mapped: female | 0.1117 | 0.2654 | 0.1885 |
| gender & age_mapped: male | 0.0469 | 0.1415 | 0.0942 |
| gender & age_mapped: 61+ | 0.0953 | 0.2181 | 0.1567 |
| gender & age_mapped: 24-40 | 0.0254 | 0.1100 | 0.0677 |
| gender & age_mapped: 41-60 | 0.0926 | 0.1386 | 0.1156 |
