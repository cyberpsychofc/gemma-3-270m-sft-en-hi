# Gemma-3-270M-SFT-en-hi

##  Project Overview  
Gemma-3-270M-SFT-en-hi is fine-tuning project which aims to make compact large language models to perform translation from English to Hindi, prioritizing efficiency and domain focus over scale.

**Base Model**: Gemma 3 270M

## Language Pairs
English → Hindi (en → hi)

## Training Data
The **IIT Bombay English-Hindi** corpus contains parallel corpus for English-Hindi as well as monolingual Hindi corpus collected from a variety of existing sources and corpora developed at the Center for Indian Language Technology, IIT Bombay over the years. This page describes the corpus. This corpus has been used at the Workshop on Asian Language Translation Shared Task since 2016. 

## Benchmark Results (Work In Progress)
Achieves an mean token accuracy of **93.05%** in English to Hindi translation task.

### English → Hindi
| Model | BLEU | 
|-------|------|
| **Gemma-3-270M-SFT-en-hi** | **34.33** |
| NLLB-200 Baseline | 32.8 |

## Intended Use
The model is intended for:
- General-purpose translation from English to Hindi.
- Run efficiently on resource-constrained machines.
- Educational purposes and language learning

## Acknowledgements
- Base Model: [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m)
- Training & Evaluation Dataset: [cfilt/iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi)