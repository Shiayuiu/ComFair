# Fairness Evaluation Report: Gemini-3-Flash on adult

## 1. Overall Performance
- **Accuracy**: 0.8411
- **Precision**: 0.6607
- **Recall**: 0.7234
- **F1**: 0.6906

## 2. Fairness Analysis Table
| Analysis Level / Condition | DP Gap | EO Gap | Unfairness Score |
| :--- | :--- | :--- | :--- |
| **race** (Base) | 0.1698 | 0.1847 | **0.1773** |
| **gender** (Base) | 0.2257 | 0.2596 | **0.2426** |
| **age_mapped** (Base) | 0.2875 | 0.3079 | **0.2977** |
| **race & gender** (Base) | 0.3017 | 0.3222 | **0.3120** |
| race & gender: white | 0.2283 | 0.2652 | 0.2468 |
| race & gender: black | 0.1199 | 0.1161 | 0.1180 |
| race & gender: female | 0.0735 | 0.0570 | 0.0653 |
| race & gender: male | 0.1819 | 0.2061 | 0.1940 |
| **race & age_mapped** (Base) | 0.4213 | 0.4902 | **0.4558** |
| race & age_mapped: white | 0.3026 | 0.3267 | 0.3146 |
| race & age_mapped: black | 0.1599 | 0.2870 | 0.2234 |
| race & age_mapped: 61+ | 0.1878 | 0.1072 | 0.1475 |
| race & age_mapped: 41-60 | 0.2614 | 0.2911 | 0.2762 |
| race & age_mapped: 24-40 | 0.1187 | 0.1634 | 0.1411 |
| **gender & age_mapped** (Base) | 0.4610 | 0.5741 | **0.5176** |
| gender & age_mapped: female | 0.1350 | 0.2335 | 0.1842 |
| gender & age_mapped: male | 0.3304 | 0.3516 | 0.3410 |
| gender & age_mapped: 61+ | 0.3008 | 0.4194 | 0.3601 |
| gender & age_mapped: 41-60 | 0.3260 | 0.3406 | 0.3333 |
| gender & age_mapped: 24-40 | 0.1306 | 0.1694 | 0.1500 |
