import argparse
import json
import numpy as np

from datasets import load_dataset

def main(args):
  data = load_dataset("json", data_files=args.data_path)["train"]
  indices = list(np.random.randint(low = len(data),size=50))
  indices.sort()
  sample_data = [data[int(i)]['text'] for i in indices]
  with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(sample_data, f, ensure_ascii=False, indent=4)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path')
  parser.add_argument('--output_path')
  args = parser.parse_args()
  main(args)