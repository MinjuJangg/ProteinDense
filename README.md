# 🧬 ProteinDense
**Domain-Guided Dense Captioning for Protein Function Prediction**  
ICBCB 2026 / Generative Genomics Workshop  

---

## Overview

ProteinDense는 단백질 기능 예측을 위해 기존의 sequence-level captioning을 확장한 **domain-aware dense captioning framework**입니다.

기존 연구는 단백질 전체 서열을 하나의 벡터로 요약하여 기능을 예측했지만,  
이 방식은 도메인 수준의 기능적 모듈성과 구조적 정보를 충분히 반영하지 못하는 한계가 있었습니다.

본 연구는 단백질을 여러 functional domain의 조합으로 분해하고,  
각 도메인을 개별적으로 설명한 뒤 이를 통합하여 단백질 전체 기능을 생성하는 구조적 접근을 제안합니다.

> Sequence → Domain → Protein → Function

---

## Motivation

### Limitations of Previous Protein Captioning

- 단백질 전체를 하나의 벡터로 요약 → 정보 손실
- 도메인 구조 반영 부족
- 복잡 기능 설명에 한계
- Protein ↔ Text 간 modality gap 존재

### Key Question

> 단백질을 하나의 서열이 아니라  
> 여러 기능 도메인의 조합으로 모델링하면  
> 기능 예측 성능과 해석 가능성을 개선할 수 있는가?

---

## Core Idea

### Dense Captioning (Vision)
이미지를 여러 region으로 나누어 각각 설명

### ProteinDense
단백질을 여러 domain으로 분해하여 각각 설명하고, 이를 통합하여 전체 기능을 생성

- Domain-level semantic modeling
- Prototype Regression 기반 alignment
- Domain caption을 LLM에 auxiliary input으로 주입
- Hierarchical caption fusion 구조

---

## Architecture

### 1. Domain Captioner

**Input**
- Protein sequence
- InterProScan 기반 domain segmentation

**Components**
- Protein Encoder: `ESM2–3B` (frozen)
- Domain Projector: `MLP + L2 Normalization`
- Prototype Regression (Many-to-One)
- LLaMA-based Domain Captioner (LoRA fine-tuning)

**Training Strategy**
- 여러 sequence variant → 하나의 기능 prototype으로 수렴
- Cosine similarity loss 기반 semantic alignment

---

### 2. Protein Captioner (Fusion Captioner)

**Role**
- Domain caption + Protein embedding 통합
- 최종 protein-level function description 생성

**Techniques**
- Contrastive Learning (Mean + Std pooling)
- Protein–Text alignment
- Instruction + Soft Prompting
- LoRA-based LLM fine-tuning

---

## Dataset

### Protein-Level
- UniProtKB / Swiss-Prot
- 약 20K protein subset

### Domain-Level
- InterPro database
- 25,854 domain instances
- 7,230 unique domain types

### Final Split

| Split | Count |
|-------|-------|
| Train | 234,179 |
| Eval  | 3,840 |
| Test  | 3,831 |
| **Total** | **241,850** |

---

## Evaluation Metrics

- Exact Match
- BLEU-2 / BLEU-4
- ROUGE-1 / ROUGE-2 / ROUGE-L
- BERTScore (RoBERTa)
- BERTScore (BioBERT)

---

## Results

### Protein Captioning Performance

| Model | Exact | BLEU-4 | ROUGE-L | BioBERTScore |
|-------|--------|--------|----------|--------------|
| Baseline (Prot2Text) | 10.3 | 14.5 | 23.8 | 76.0 |
| Ours (Generated Domain) | 14.4 | 20.4 | 30.3 | 79.8 |
| Ours (GT Domain) | 18.5 | 26.3 | 40.1 | 84.0 |

Domain information consistently improves performance over baseline.

---

## Ablation Study

| Domain Caption | Exact | BLEU-4 |
|---------------|-------|--------|
| None | 10.3 | 14.5 |
| Generated | 14.4 | 20.4 |
| GT | 18.5 | 26.3 |
| Random | 3.0 | 5.7 |

Domain caption quality directly impacts prediction performance.

---

## Qualitative Analysis

**Case 1 – Mechanism Correction**  
Hydrolysis → Phosphorylation correct recovery via domain hint

**Case 2 – Functional Direction Correction**  
Activator → Repressor 방향성 복원

**Case 3 – Hallucination Mitigation**  
잘못된 motor protein 예측 → 정확한 complex membership 복원

---

## Contributions

- Domain-aware dense captioning framework 제안
- Prototype Regression 기반 semantic alignment
- Domain-to-Protein hierarchical fusion 구조 설계
- Domain 품질과 예측 성능 간 통계적 상관성 검증
- Hallucination 감소 및 기능 방향성 오류 완화

---

## Conclusion

ProteinDense는 단백질 기능 예측을  
sequence-level captioning에서  
domain-aware dense captioning으로 확장한 구조적 접근입니다.

Domain 정보를 보조 입력으로 활용함으로써:

- 성능을 일관되게 향상
- 기능 방향성 오류 감소
- hallucination 완화
- 복잡 기능 설명에 대한 표현력 강화
