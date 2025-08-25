# Romanian Pragmatics Benchmark (LLMs)

This repository contains CSV datasets and evaluation scripts used in:
Towards a Quantitative Assessment of Pragmatic Competence in Generative Large Language Models (LLMs) in Romanian, University of Bucharest, 2025.

## Files
- data/items_all.csv — full benchmark
- data/items_politeness.csv — politeness subset
- data/items_indirectness.csv — indirectness subset
- data/items_implicature.csv — implicature (GCI/PCI) subset
- APIcall+evaluate.py -ntegration of the the API script with evalution
- code/evaluate.py — accuracy & macro-recall evaluation
- metadata/datasheet.md — dataset documentation

## Columns
Context | Option_A | Option_B | Gold_Label | Phenomenon | Domain | Role_Direction
- `Gold_Label` ∈ {A, B}
- `Phenomenon` ∈ {politeness, indirectness, implicature_gci, implicature_pci}

## License
Data: CC BY 4.0. Code: MIT.
