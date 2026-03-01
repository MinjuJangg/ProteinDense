# 🧬 ProteinDense
**Domain-Guided Dense Captioning for Protein Function Prediction**  
ICBCB 2026 / Generative Genomics Workshop  

---

## Overview

ProteinDense는 단백질 기능 예측을 위한 **domain-aware dense captioning framework**

기존 protein captioning 연구는 단백질 전체 서열을 하나의 벡터로 압축하여 기능을 생성했으며,  
이로 인해 기능적 모듈성(domain-level modularity)을 반영하지 못하는 한계

본 연구는 단백질을

> **Sequence → Domain → Protein → Function**

구조로 재해석하고,  
도메인 단위에서 먼저 의미를 학습한 후 이를 통합하는 **계층적 생성 모델**을 제안


## Motivation

### Experimental & Computational Gap

- NGS 발전으로 단백질 서열 데이터는 폭증
- 실제 wet-lab 기능 검증은 비용이 높고 느림
- 전체 단백질 중 1% 미만만 실험적으로 기능이 확인됨

→ 자동화된 protein function captioning 필요

## Limitations of Previous Work

기존 연구의 공통 한계:

- 단백질–텍스트 modality gap 감소에 집중
- sequence-level representation만 사용 (한개의 벡터로 압축)
- domain-level modular reasoning 부재
- 복합 기능 설명 시 정보 손실 발생

## Key Idea

> 단백질 기능을 한 문장으로 생성하기 전에,  
> 각 **Domain을 먼저 설명(Dense Captioning)** 하고 이를 통합

Vision 분야의 Dense Captioning 개념을 단백질에 적용:

- Region-level → Domain-level
- Global caption → Hierarchical caption

# Architecture

## Inference Path

```
Protein Sequence
↓
InterProScan
↓
Domain Extraction
↓
Domain Embedder (ESM2-3B)
↓
Domain Captioner (LLAMA + LoRA)
↓
Protein Captioner (Fusion LLM)
↓
Final Function Description
```

## Training Path

1. Domain-level Prototype Regression
2. Protein–Text Contrastive Alignment
3. Instruction + Soft Prompting 기반 LLM Fine-tuning
4. LoRA 적용


## Part 1. Domain Captioner

### Domain Feature Extractor

- Protein Encoder: **ESM2-3B (Frozen)**
- Domain Projector: MLP + GeLU
- L2 Normalization

### Prototype Regression

- Many-to-One Regression
- 여러 sequence variant → 하나의 functional prototype
- Loss: Cosine Similarity

→ Domain semantic space 정렬


## Part 2. Protein Captioner

### Protein–Text Contrastive Alignment

- Protein: Mean / Std pooling
- Text: LLaMA intermediate layer pooling
- Mean-level alignment
- Distribution-level alignment

Loss:
- Contrastive loss (mean + variance)

## Part 3 & 4. LLM-Tuning

- Domain Captioner(LLAMA3-8B) Fine-Tuning(Lora)
- Fusion Captioner(LLAMA3-8B) Fine-Tuning(Lora)


## Dataset

### Protein-Level Dataset

- UniProtKB / Swiss-Prot
- Total: 241,850 (Filtered)

### Domain-Level Dataset

- InterPro
- 20K protein subset
- 25,854 domain instances
- 7,230 unique domain types



## Results

### Protein Captioner

| Method | Exact | BLEU-2 | ROUGE-L | BioBERT |
|--------|-------|--------|---------|---------|
| Baseline (Prot2Text) | 10.3 | 17.3 | 23.8 | 76.0 |
| Ours (Generated Domain) | 14.4 | 24.6 | 30.3 | 79.8 |
| Ours (GT Domain) | 18.5 | 31.9 | 40.1 | 84.0 |

→ Domain 정보가 일관된 성능 향상 유도


### Domain Captioner Strategy

| Strategy | Exact | BLEU-4 | ROUGE-L |
|-----------|-------|--------|---------|
| Contrastive | 1.0 | 4.7 | 16.5 |
| Prototype Regression | 5.8 | 11.4 | 21.0 |

→ Prototype Regression이 가장 효과적


## Qualitative Analysis

#### Case 1 – Mechanism Correction
Baseline: Hydrolysis  
Ours: Phosphorylation (Correct enzymatic mechanism)

#### Case 2 – Functional Direction Correction
Baseline: Activator  
Ours: Repressor (Correct direction recovered)

#### Case 3 – Hallucination Mitigation
Baseline: Motor protein hallucination  
Ours: Correct exocyst complex membership

## Domain ↔ Caption Alignment

Prototype Regression 사용 시

- Intra similarity 증가
- Inter similarity 감소
- Alignment Gap 최대화

→ Domain semantic alignment 개선


## Conclusion

ProteinDense는

- Domain-level modeling
- Prototype Regression 기반 정렬
- Hierarchical caption generation

을 통해 기존 sequence-level 접근 대비  
일관된 성능 향상을 달성

Domain quality가 높을수록 prediction 성능이 단계적으로 향상됨을 확인
