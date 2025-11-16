"""
Data preparation script for AgentFlow
"""

import os
import sys
import shutil
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """准备数据集"""
    print("Preparing datasets for AgentFlow...")
    
    # 创建数据目录
    data_dir = project_root / "src" / "data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"Data directories created:")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")
    
    # 检查是否有原始数据文件
    aime24_path = data_dir / "aime24.parquet"
    combined_train_path = train_dir / "combined_train.parquet"
    
    if not aime24_path.exists():
        print(f"\nAIME24 dataset not found at: {aime24_path}")
        print("Please run the data preparation scripts first:")
        print("  python aime24_data.py")
        print("  python get_train_data.py")
        return
    
    # 复制AIME24数据到验证目录
    val_aime24_path = val_dir / "aime24.parquet"
    if not val_aime24_path.exists():
        print(f"\nCopying AIME24 data to validation directory...")
        shutil.copy2(aime24_path, val_aime24_path)
        print(f"Copied to: {val_aime24_path}")
    
    # 检查训练数据
    if not combined_train_path.exists():
        print(f"\nCombined training data not found at: {combined_train_path}")
        print("Please run: python get_train_data.py")
        return
    
    print(f"\nDataset preparation completed!")
    print(f"Training data: {combined_train_path}")
    print(f"Validation data: {val_aime24_path}")
    
    # 显示数据集信息
    try:
        import pandas as pd
        
        # 读取训练数据
        train_df = pd.read_parquet(combined_train_path)
        print(f"\nTraining dataset info:")
        print(f"  Samples: {len(train_df)}")
        print(f"  Columns: {list(train_df.columns)}")
        
        # 读取验证数据
        val_df = pd.read_parquet(val_aime24_path)
        print(f"\nValidation dataset info:")
        print(f"  Samples: {len(val_df)}")
        print(f"  Columns: {list(val_df.columns)}")
        
        # 显示样本
        print(f"\nSample from training data:")
        sample = train_df.iloc[0]
        for col in ['question', 'result']:
            if col in sample:
                print(f"  {col}: {str(sample[col])[:100]}...")
        
    except Exception as e:
        print(f"Error reading dataset info: {e}")


if __name__ == "__main__":
    main()