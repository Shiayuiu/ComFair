# Fairness Evaluation Report: Deepseek-V3.2-chat on adult

## 1. Overall Performance
- **Accuracy**: 0.8225
- **Precision**: 0.7101
- **Recall**: 0.4886
- **F1**: 0.5789

## 2. Fairness Analysis Table
| Analysis Level / Condition | DP Gap | EO Gap | Unfairness Score |
| :--- | :--- | :--- | :--- |
| **race** (Base) | 0.1113 | 0.1119 | **0.1116** |
| **gender** (Base) | 0.1563 | 0.2358 | **0.1960** |
| **age_mapped** (Base) | 0.1494 | 0.1268 | **0.1381** |
| **race & gender** (Base) | 0.2055 | 0.3072 | **0.2563** |
| race & gender: white | 0.1578 | 0.2360 | 0.1969 |
| race & gender: black | 0.0896 | 0.2000 | 0.1448 |
| race & gender: female | 0.0477 | 0.0712 | 0.0594 |
| race & gender: male | 0.1159 | 0.1072 | 0.1116 |
| **race & age_mapped** (Base) | 0.2464 | 0.4447 | **0.3456** |
| race & age_mapped: white | 0.1562 | 0.1385 | 0.1473 |
| race & age_mapped: black | 0.0916 | 0.3691 | 0.2303 |
| race & age_mapped: 41-60 | 0.1548 | 0.1721 | 0.1634 |
| race & age_mapped: 61+ | 0.0787 | 0.2752 | 0.1769 |
| race & age_mapped: 24-40 | 0.0902 | 0.0850 | 0.0876 |
| **gender & age_mapped** (Base) | 0.2762 | 0.3482 | **0.3122** |
| gender & age_mapped: female | 0.0609 | 0.1568 | 0.1089 |
| gender & age_mapped: male | 0.1705 | 0.1237 | 0.1471 |
| gender & age_mapped: 41-60 | 0.2153 | 0.2698 | 0.2426 |
| gender & age_mapped: 61+ | 0.1730 | 0.1530 | 0.1630 |
| gender & age_mapped: 24-40 | 0.1048 | 0.2084 | 0.1566 |
