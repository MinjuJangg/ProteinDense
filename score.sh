export CUDA_VISIBLE_DEVICES=1


python scripts/benchmark.py \
  --read_generation_dir /home/minju/Prot2Text-V2/1215/ \
  --read_file_identifier 20000_generation.json \
  --evaluate_exact_match true \
  --evaluate_bleu true \
  --evaluate_rouge true \
  --evaluate_bert_score false \
  --verbose true
