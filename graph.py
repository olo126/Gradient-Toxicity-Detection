import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

results_dir = 'results/Llama_8B/selected_data_separate'
task = 'toxic'
for i in range(100):
  scores_df = pd.read_csv(os.path.join(results_dir, task, f"val_{i}/sorted.csv"))
  plot = sns.displot(scores_df, x=" score", kind="kde")
  fig = plot._figure
  fig.savefig(os.path.join(results_dir, task, f"val_{i}/smooth.png"))
  plt.close()
  plot2 = sns.displot(scores_df, x=" score", bins=50)
  fig2 = plot2._figure
  fig2.savefig(os.path.join(results_dir, task, f"val_{i}/histogram.png"))
  plt.close()