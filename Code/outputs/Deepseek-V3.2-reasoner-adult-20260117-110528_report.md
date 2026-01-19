# Fairness Evaluation Report: Deepseek-V3.2-reasoner on adult

## 1. Overall Performance
- **Accuracy**: 0.8564
- **Precision**: 0.7040
- **Recall**: 0.6289
- **F1**: 0.6644

## 2. Fairness Analysis Table
| Analysis Level / Condition | DP Gap | EO Gap | Unfairness Score |
| :--- | :--- | :--- | :--- |
| **race** (Base) | 0.1241 | 0.1046 | **0.1144** |
| **gender** (Base) | 0.1737 | 0.2167 | **0.1952** |
| **age_mapped** (Base) | 0.2152 | 0.1886 | **0.2019** |
| **race & gender** (Base) | 0.2155 | 0.2552 | **0.2354** |
| race & gender: black | 0.0720 | 0.0203 | 0.0461 |
| race & gender: white | 0.1787 | 0.2301 | 0.2044 |
| race & gender: female | 0.0368 | 0.1220 | 0.0794 |
| race & gender: male | 0.1435 | 0.1378 | 0.1406 |
| **race & age_mapped** (Base) | 0.3216 | 0.3506 | **0.3361** |
| race & age_mapped: black | 0.1419 | 0.2521 | 0.1970 |
| race & age_mapped: white | 0.2248 | 0.1962 | 0.2105 |
| race & age_mapped: 41-60 | 0.1798 | 0.1462 | 0.1630 |
| race & age_mapped: 61+ | 0.1341 | 0.0713 | 0.1027 |
| race & age_mapped: 24-40 | 0.0969 | 0.1544 | 0.1257 |
| **gender & age_mapped** (Base) | 0.3595 | 0.3777 | **0.3686** |
| gender & age_mapped: female | 0.1213 | 0.1561 | 0.1387 |
| gender & age_mapped: male | 0.2407 | 0.1883 | 0.2145 |
| gender & age_mapped: 41-60 | 0.2382 | 0.2216 | 0.2299 |
| gender & age_mapped: 61+ | 0.2431 | 0.3138 | 0.2784 |
| gender & age_mapped: 24-40 | 0.1081 | 0.1816 | 0.1448 |
