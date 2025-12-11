# !pip install datasets # 因为我们要从hf中下载数据，所以

from datasets import load_dataset

# ds = load_dataset("gunnybd01/Financial_Services_News_smr")      
# 指定路径
ds = load_dataset("gunnybd01/Financial_Services_News_smr",cache_dir="/Users/isaac/financeTuning/data/sft/download")