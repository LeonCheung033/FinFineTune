# 导入必要的库
import json                                    # 用于处理JSON格式数据的读写
import os                                      # 用于操作系统相关功能，如文件路径操作
from datasets import load_from_disk, Dataset  # Hugging Face datasets库，用于加载和处理数据集
from tqdm import tqdm                          # 用于显示进度条，让用户知道处理进度

# 指定数据目录和输出文件路径
data_dir = "/root/autodl-tmp/data"                                          # 数据存储的根目录
output_file = os.path.join(data_dir, "filtered_financial_news_5k.jsonl")   # 过滤后数据的输出文件路径

try:
    # 第一种方法：使用 Hugging Face 的方法直接读取 Arrow 文件
    # Arrow是一种高效的列式数据存储格式，Hugging Face datasets经常使用这种格式
    from datasets import Dataset  # 重新导入Dataset类（虽然上面已经导入，但这里明确表示要使用）
    import pyarrow as pa          # PyArrow库，用于处理Arrow格式文件
    
    # 定义两个Arrow文件的完整路径
    # 这些文件是数据集的分片文件（split files），大数据集通常会分成多个文件存储
    arrow_file_1 = os.path.join(data_dir, "financial_services_news_smr-train-00000-of-00002.arrow")  # 第一个分片文件
    arrow_file_2 = os.path.join(data_dir, "financial_services_news_smr-train-00001-of-00002.arrow")  # 第二个分片文件
    
    # 使用 datasets 库的特殊方法加载数据
    print("尝试使用 datasets.Dataset.from_file 加载数据...")
    
    # 尝试直接读取每个分片文件
    datasets = []  # 创建一个空列表，用于存储成功加载的数据集
    
    # 检查第一个Arrow文件是否存在，如果存在则尝试加载
    if os.path.exists(arrow_file_1):
        try:
            ds1 = Dataset.from_file(arrow_file_1)  # 从Arrow文件创建Dataset对象
            datasets.append(ds1)                   # 将成功加载的数据集添加到列表中
            print(f"成功加载文件: {arrow_file_1}")
        except Exception as e:
            # 如果加载失败，打印错误信息但不停止程序
            print(f"加载 {arrow_file_1} 失败: {e}")
    
    # 检查第二个Arrow文件是否存在，如果存在则尝试加载
    if os.path.exists(arrow_file_2):
        try:
            ds2 = Dataset.from_file(arrow_file_2)  # 从Arrow文件创建Dataset对象
            datasets.append(ds2)                   # 将成功加载的数据集添加到列表中
            print(f"成功加载文件: {arrow_file_2}")
        except Exception as e:
            # 如果加载失败，打印错误信息但不停止程序
            print(f"加载 {arrow_file_2} 失败: {e}")
    
    # 如果成功加载了至少一个数据集文件
    if datasets:
        # 合并数据集（如果有多个分片文件）
        if len(datasets) > 1:
            # 如果有多个数据集，使用concatenate_datasets函数将它们合并成一个
            from datasets import concatenate_datasets
            dataset = concatenate_datasets(datasets)
        else:
            # 如果只有一个数据集，直接使用它
            dataset = datasets[0]
            
        # 开始过滤数据并写入到输出文件
        total_count = len(dataset)      # 获取数据集的总记录数
        filtered_count = 0              # 初始化过滤后的记录计数器
        
        # 打开输出文件，准备写入过滤后的数据
        # 'w'表示写入模式，'encoding="utf-8"'确保正确处理中文等Unicode字符
        with open(output_file, "w", encoding="utf-8") as f:
            # 遍历数据集中的每一条记录，tqdm提供进度条显示
            for item in tqdm(dataset):
                # 过滤条件：只保留文章长度小于5000字符的记录
                if len(item["Article"]) < 5000:
                    # 创建一个新的字典，只包含我们需要的字段
                    filtered_item = {
                        "Article": item["Article"],    # 文章内容
                        "Summary": item["Summary"]     # 文章摘要
                    }
                    # 将过滤后的记录转换为JSON格式并写入文件
                    # ensure_ascii=False确保中文字符正常显示，而不是转义序列
                    f.write(json.dumps(filtered_item, ensure_ascii=False) + "\n")
                    filtered_count += 1  # 增加过滤后记录的计数
        
        # 打印处理结果的统计信息
        print(f"总数据量: {total_count}")
        print(f"过滤后的数据量: {filtered_count}")
        print(f"数据已保存到 {output_file}")
    else:
        # 如果没有成功加载任何数据集文件，抛出异常
        raise Exception("无法加载任何数据集文件")

except Exception as e:
    # 如果第一种方法失败，打印错误信息并尝试第二种方法
    print(f"使用 Dataset.from_file 方法失败: {e}")
    
    # 第二种方法：如果上述方法失败，尝试使用 memory_mapped_arrow 格式
    try:
        print("\n尝试另一种方法 - 直接读取内存映射的 arrow 文件...")
        # 导入datasets库的内部函数，用于连接多个数据集
        from datasets.arrow_dataset import _concatenate_map_style_datasets
        
        # 定义一个函数来尝试读取 Arrow 文件
        def try_load_arrow(file_path):
            """
            尝试使用底层方法读取Arrow文件
            参数: file_path - Arrow文件的完整路径
            返回: Dataset对象或None（如果失败）
            """
            try:
                # 使用 dataset 库的底层方法尝试读取
                from datasets.arrow_dataset import ArrowReader
                reader = ArrowReader(file_path)  # 创建Arrow文件读取器
                dataset = reader.read()          # 读取数据集
                return dataset
            except Exception as e:
                # 如果读取失败，打印错误信息并返回None
                print(f"无法读取 Arrow 文件 {file_path}: {e}")
                return None
        
        # 尝试读取所有的 Arrow 文件
        datasets = []  # 创建空列表存储成功加载的数据集
        # 遍历所有可能的Arrow文件名
        for file_name in ["financial_services_news_smr-train-00000-of-00002.arrow", 
                         "financial_services_news_smr-train-00001-of-00002.arrow"]:
            file_path = os.path.join(data_dir, file_name)  # 构建完整文件路径
            if os.path.exists(file_path):                  # 检查文件是否存在
                ds = try_load_arrow(file_path)             # 尝试加载文件
                if ds is not None:                         # 如果加载成功
                    datasets.append(ds)                    # 添加到数据集列表
        
        # 如果成功加载了至少一个数据集
        if datasets:
            # 合并数据集（如果有多个）
            if len(datasets) > 1:
                # 使用内部函数合并多个数据集
                dataset = _concatenate_map_style_datasets(datasets)
            else:
                # 如果只有一个数据集，直接使用
                dataset = datasets[0]
                
            # 这里应该有数据过滤和写入的代码，但原代码中用注释省略了
            # 实际应用中，这里会包含与上面相同的数据处理逻辑
        else:
            # 如果没有成功加载任何Arrow文件，抛出异常
            raise Exception("无法加载任何 Arrow 文件")
            
    except Exception as e2:
        # 如果第二种方法也失败，打印错误信息并尝试第三种方法
        print(f"使用备选方法也失败了: {e2}")
        
        # 第三种方法：最后的方案，尝试使用原始 Hugging Face API 重新下载
        print("\n尝试使用 Hugging Face API 重新加载数据集...")
        try:
            # 导入load_dataset函数，用于从Hugging Face Hub下载数据集
            from datasets import load_dataset
            
            # 直接从 Hugging Face Hub 重新下载和加载数据集
            # "gunnybd01/Financial_Services_News_smr" 是数据集在Hugging Face上的标识符
            dataset = load_dataset("gunnybd01/Financial_Services_News_smr")
            
            # 检查数据集是否包含训练集分割
            if "train" in dataset:
                train_dataset = dataset["train"]           # 获取训练集部分
                total_count = len(train_dataset)           # 获取训练集的总记录数
                print(f"直接从 Hugging Face 加载成功! 数据集大小: {total_count}")
                
                # 过滤并写入数据（与第一种方法相同的逻辑）
                filtered_count = 0  # 初始化过滤后记录计数器
                
                # 打开输出文件准备写入
                with open(output_file, "w", encoding="utf-8") as f:
                    # 遍历训练集中的每条记录
                    for item in tqdm(train_dataset):
                        # 过滤条件：文章长度小于14000字符（注意这里的阈值与第一种方法不同）
                        if len(item["Article"]) < 14000:
                            # 创建包含所需字段的新字典
                            filtered_item = {
                                "Article": item["Article"],    # 文章内容
                                "Summary": item["Summary"]     # 文章摘要
                            }
                            # 将记录转换为JSON格式并写入文件
                            f.write(json.dumps(filtered_item, ensure_ascii=False) + "\n")
                            filtered_count += 1  # 增加过滤后记录计数
                
                # 打印最终的处理统计信息
                print(f"总数据量: {total_count}")
                print(f"过滤后的数据量: {filtered_count}")
                print(f"数据已保存到 {output_file}")
            
        except Exception as e3:
            # 如果所有方法都失败了，打印最终的错误信息
            print(f"所有尝试都失败了: {e3}")
            
            # 给用户提供一些建议和可能的解决方案
            print("\n建议尝试以下方法:")
            print("1. 检查 dataset_info.json 文件中的详细内容，看是否有关于如何加载数据的提示")
            print("2. 尝试在环境中更新 datasets 库: pip install -U datasets")
            print("3. 尝试在全新环境中重新下载并加载数据集")
            print("4. 或者直接联系数据集作者询问正确的加载方式")