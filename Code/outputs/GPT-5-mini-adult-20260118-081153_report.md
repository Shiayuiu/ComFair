# Fairness Evaluation Report: GPT-5-mini on adult

## 1. Overall Performance
- **Accuracy**: 0.8269
- **Precision**: 0.6265
- **Recall**: 0.6982
- **F1**: 0.6604

## 2. Fairness Analysis Table
| Analysis Level / Condition | DP Gap | EO Gap | Unfairness Score |
| :--- | :--- | :--- | :--- |
| **race** (Base) | 0.1945 | 0.2964 | **0.2454** |
| **gender** (Base) | 0.2445 | 0.3966 | **0.3206** |
| **age_mapped** (Base) | 0.2403 | 0.2223 | **0.2313** |
| **race & gender** (Base) | 0.3256 | 0.5340 | **0.4298** |
| race & gender: black | 0.1032 | 0.2111 | 0.1572 |
| race & gender: white | 0.2498 | 0.4027 | 0.3262 |
| race & gender: female | 0.0758 | 0.1313 | 0.1036 |
| race & gender: male | 0.2224 | 0.3229 | 0.2727 |
| **race & age_mapped** (Base) | 0.3966 | 0.4865 | **0.4416** |
| race & age_mapped: black | 0.1044 | 0.0804 | 0.0924 |
| race & age_mapped: white | 0.2569 | 0.2457 | 0.2513 |
| race & age_mapped: 61+ | 0.2102 | 0.3163 | 0.2632 |
| race & age_mapped: 24-40 | 0.1397 | 0.2384 | 0.1891 |
| race & age_mapped: 41-60 | 0.2922 | 0.4061 | 0.3491 |
| **gender & age_mapped** (Base) | 0.4394 | 0.6001 | **0.5197** |
| gender & age_mapped: female | 0.1189 | 0.1814 | 0.1502 |
| gender & age_mapped: male | 0.2679 | 0.2231 | 0.2455 |
| gender & age_mapped: 61+ | 0.2918 | 0.4693 | 0.3805 |
| gender & age_mapped: 24-40 | 0.1716 | 0.3677 | 0.2697 |
| gender & age_mapped: 41-60 | 0.3206 | 0.4187 | 0.3696 |
