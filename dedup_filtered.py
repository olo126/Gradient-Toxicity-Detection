import argparse
import json
import numpy as np
import os

from datasets import load_dataset

def main(args):
  l = []
  total_data=0
  for i in range(100):
    data_file = os.path.join(args.data_path, f"val_{i}/top_p{args.top_p}.jsonl")
    data = load_dataset("json", data_files=data_file)["train"]
    total_data += len(data)
    for i in range(len(data)):
      if data[i] not in l:
        l.append(data[i])
    print(len(l))
  print(len(l))
  print(total_data)
  with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(l, f, ensure_ascii=False, indent=4)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path')
  parser.add_argument('--output_path')
  parser.add_argument('--top_p')
  args = parser.parse_args()
  main(args)