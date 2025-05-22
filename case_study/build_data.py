import pandas as pd
from pathlib import Path
from loguru import logger
from molvs import standardize_smiles
from sklearn.model_selection import ShuffleSplit


def _standardize_smiles(smiles):
    try:
        return standardize_smiles(smiles)
    except Exception as e:
        logger.warning(f"Error standardizing SMILES: {e}")
        return smiles


def preprocess_data(src_data_path, task):
    """读取并预处理数据：标准化 smiles、筛选列、去除缺失。"""
    logger.info(f"Reading data from {src_data_path}...")
    df = pd.read_csv(src_data_path)

    expected_cols = {"smiles", "solvent", task}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    # 标准化 smiles 和 solvent
    logger.info("Standardizing SMILES and solvent...")
    for col in ["smiles", "solvent"]:
        df[col] = df[col].apply(_standardize_smiles)

    # 筛选需要的列，去除缺失值
    df = df[list(expected_cols)].dropna().reset_index(drop=True)
    logger.info(f"Data shape after preprocessing: {df.shape}")
    return df


def split_data(df):
    """将数据集划分为 train, valid, test。"""
    logger.info("Splitting data into train/valid/test...")

    df = df.copy()
    df["split"] = "train"

    splitter1 = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(splitter1.split(df))
    df.loc[temp_idx, "split"] = "temp"

    temp_df = df[df["split"] == "temp"].copy()
    splitter2 = ShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(splitter2.split(temp_df))

    temp_df.loc[temp_df.index[val_idx], "split"] = "valid"
    temp_df.loc[temp_df.index[test_idx], "split"] = "test"

    df.update(temp_df)
    df = df[df["split"] != "temp"].reset_index(drop=True)

    split_counts = df["split"].value_counts().to_dict()
    logger.info(f"Split counts: {split_counts}")

    return df


def build_data(src_data_path, dst_data_dir, task):
    """主流程函数，处理并保存任务数据。"""
    dst_data_dir.mkdir(parents=True, exist_ok=True)

    df = preprocess_data(src_data_path, task)
    df = split_data(df)

    save_path = dst_data_dir / task / f"{task}.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.success(f"Data saved to {save_path}")


if __name__ == "__main__":
    src_data_path = Path("case.csv")  # 输入文件
    dst_data_dir = Path("datasets")   # 保存目录

    tasks = [
        "absorption",
        "emission",
        "log_molar_absorptivity",
        "quantum_yield"
    ]

    for task in tasks:
        try:
            build_data(src_data_path, dst_data_dir, task)
        except Exception as e:
            logger.error(f"Failed to process task '{task}': {e}")
