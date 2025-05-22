import lightgbm as lgb
import pandas as pd
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from src.data.finetune_dataset import MoleculeDataset
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from loguru import logger
import sys
import joblib
from sklearn.metrics import mean_squared_error, r2_score

logger.remove()
logger.add(sys.stdout, level="DEBUG")

def get_model(model_name: str, random_state: int = 42, verbose: bool = True):
    """
    Returns the specified regression model based on the model_name.

    :param model_name: Model name as string ('rf', 'svr', 'lightgbm', 'gbrt')
    :param random_state: Random state for reproducibility
    :param verbose: Whether to display detailed logs for the model (for models that support it)

    :return: A model instance
    """
    if model_name == "rf":
        return RandomForestRegressor(random_state=random_state, verbose=verbose, n_estimators=500, n_jobs=4)
    elif model_name == "svr":
        return SVR()  # SVR doesn't take random_state directly
    elif model_name == "lightgbm":
        return lgb.LGBMRegressor(random_state=random_state, n_estimators=500, n_jobs=4)
    elif model_name == "gbrt":
        return GradientBoostingRegressor(
            random_state=random_state,
            verbose=verbose,
            n_estimators=500,
        )
    else:
        raise ValueError(f"Model {model_name} not found")


def run_single_task(
    data_dir: str,
    split_method: str,
    dataset_name: str,
    fold: int,
    task_name: str,
    save_dir: str,
    model_name: str,
    model_save_dir: str,
):
    logger.info(f"Loading datasets (train and validation)")
    train_dataset = MoleculeDataset(
        root_path=data_dir,
        dataset=task_name,
        dataset_type="regression",
        split_name="splits",
        split="train",
    )

    val_dataset = MoleculeDataset(
        root_path=data_dir,
        dataset=task_name,
        dataset_type="regression",
        split_name="splits",
        split="val",
    )

    # Extract features and labels from the train dataset
    smiles_list, solvent_list, graphs, fps, mds, sds, labels = map(
        list, zip(*train_dataset)
    )
    features = np.hstack([np.vstack(fps), np.vstack(mds), np.vstack(sds)])
    labels = np.hstack(labels)
    assert features.shape[0] == labels.shape[0]
    logger.debug(f"Train shape: {features.shape}, {labels.shape}")

    # Train the model
    logger.info(f"Training {model_name} model")
    model = get_model(model_name)
    model.fit(features, labels)
    save_path = model_save_dir / f"{task_name}.pkl"
    joblib.dump(model, save_path)
    logger.info(f"Model saved to {save_path}")

    # Make predictions on the training set and validation set
    preds_train = model.predict(features)

    # Extract features and labels from the validation dataset
    (
        val_smiles_list,
        val_solvent_list,
        val_graphs,
        val_fps,
        val_mds,
        val_sds,
        val_labels,
    ) = map(list, zip(*val_dataset))

    val_features = np.hstack(
        [np.vstack(val_fps), np.vstack(val_mds), np.vstack(val_sds)]
    )
    val_labels = np.hstack(val_labels)
    logger.debug(f"Validation shape: {val_features.shape}, {val_labels.shape}")
    assert val_features.shape[0] == val_labels.shape[0]

    preds_val = model.predict(val_features)

    # Calculate evaluation metrics
    train_rmse = np.sqrt(mean_squared_error(labels, preds_train))
    val_rmse = np.sqrt(mean_squared_error(val_labels, preds_val))
    train_r2 = r2_score(labels, preds_train)
    val_r2 = r2_score(val_labels, preds_val)

    logger.info(f"Train RMSE: {train_rmse}, Train R²: {train_r2}")
    logger.info(f"Validation RMSE: {val_rmse}, Validation R²: {val_r2}")

    train_results = {
        "smiles": smiles_list,
        "solvent": solvent_list,
        "label": labels,
        "predictions": preds_train,
    }
    val_results = {
        "smiles": val_smiles_list,
        "solvent": val_solvent_list,
        "label": val_labels,
        "predictions": preds_val,
    }

    df_train = pd.DataFrame(train_results)
    df_val = pd.DataFrame(val_results)

    df_train.to_csv(save_dir / "train.csv", index=False)
    df_val.to_csv(save_dir / "valid.csv", index=False)


if __name__ == "__main__":
    for fold in range(5):
        split_method = "scaffold"
        data_dir = "datasets"
        dataset_name = "xanthene"  # "consolidation", "cyanine", "xanthene"
        task_name = "log_molar_absorptivity"  # "absorption" "emission", "quantum_yield", "log_molar_absorptivity"
        model_name = "svr"  # "rf", "svr", "lightgbm", "gbrt"

        save_dir = Path(
            f"results_{model_name}/{split_method}/{dataset_name}_fold{fold}/{task_name}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        model_save_dir = Path(f"models/{model_name}/{split_method}/{dataset_name}_fold{fold}")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        task_dir = Path(data_dir) / split_method / f"{dataset_name}_fold{fold}"
        run_single_task(
            data_dir=task_dir,
            split_method=split_method,
            dataset_name=dataset_name,
            fold=fold,
            task_name=task_name,
            save_dir=save_dir,
            model_name=model_name,
            model_save_dir=model_save_dir,
        )
