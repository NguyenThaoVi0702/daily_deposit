import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
import logging
from typing import Dict, Any, List
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import mean_absolute_error

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

n_jobs = 4

def split_data_to_dev_and_test(
    m1_features: pd.DataFrame,
    m2_features: pd.DataFrame,
    params: Dict[str, Any]
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Splits full feature sets into development and test (holdout) sets."""
    holdout_days = params["holdout_days"]
    
    m1_features.index = pd.to_datetime(m1_features.index)
    m2_features.index = pd.to_datetime(m2_features.index)

    cutoff_date = m1_features.index.max() - pd.Timedelta(days=holdout_days)
    log.info(f"Splitting data. Development set ends on {cutoff_date.date()}.")

    m1_dev = m1_features[m1_features.index <= cutoff_date]
    m1_test = m1_features[m1_features.index > cutoff_date]

    m2_dev = m2_features[m2_features.index <= cutoff_date]
    m2_test = m2_features[m2_features.index > cutoff_date]

    log.info(f"M1 dev/test split: {m1_dev.shape}/{m1_test.shape}")
    log.info(f"M2 dev/test split: {m2_dev.shape}/{m2_test.shape}")
    
    return m1_dev, m1_test, m2_dev, m2_test


def split_features_and_target(df_features: pd.DataFrame, params: Dict[str, Any]) -> (pd.DataFrame, pd.Series, str): # Note the changed return type hint
    """Splits the feature data into X and y, and gets the feature list as a string."""
    target_col = params["target_col"]
    balance_col = params["balance_col"]
    
    feature_cols = [col for col in df_features.columns if col not in [target_col, balance_col]]
    X = df_features[feature_cols]
    y = df_features[target_col]
    
    log.info(f"Data split complete. X shape: {X.shape}, y shape: {y.shape}")
    
    features_as_string = "\n".join(feature_cols)
    log.info(f"{len(feature_cols)} features saved.")
    
    return X, y, features_as_string 


def _objective_unified(trial, X_train, y_train, n_splits, pred_type):
    """Internal objective function for Optuna, not a node."""

    if pred_type == 'mean':
        base_params = {'objective': 'regression_l2', 'metric': 'mae', 'random_state': 42, 'n_jobs': n_jobs, 'verbose': -1, 'boosting_type': 'gbdt'}
        tuned_params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1200, step=100), 'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100), 'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95), 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7), 'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
    else:
        alpha = pred_type
        base_params = {'objective': 'quantile', 'alpha': alpha, 'metric': 'quantile', 'random_state': 42, 'n_jobs': n_jobs, 'verbose': -1, 'boosting_type': 'gbdt'}
        if alpha == 0.5:
            tuned_params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 80), 'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9), 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
            }
        else:
            tuned_params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500, step=100), 'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 30, 150), 'min_child_samples': trial.suggest_int('min_child_samples', 5, 25),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.95), 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 20.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 20.0, log=True),
            }
    params = {**base_params, **tuned_params}
    eval_metric = 'mae' if pred_type == 'mean' else 'quantile'
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        if len(train_idx) == 0 or len(val_idx) == 0: continue
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        metric_to_watch = "l1" if pred_type == 'mean' else "quantile"

        pruning_callback = LightGBMPruningCallback( 
            trial, 
            metric_to_watch, 
            valid_name="valid_0"
        )
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  eval_metric=eval_metric,
                  callbacks=[pruning_callback])

        score = model.best_score_['valid_0']['l1'] if pred_type == 'mean' else model.best_score_['valid_0']['quantile']
        scores.append(score)

        trial.report(score, i)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores) if scores else float('inf')


def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, pred_type: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Tunes hyperparameters for a given prediction type using Optuna."""
    log.info(f"--- Tuning hyperparameters for target: {pred_type} ---")
    
    cutoff_idx = int(len(X) * params["initial_train_ratio_for_optuna"])
    X_optuna, y_optuna = X.iloc[:cutoff_idx], y.iloc[:cutoff_idx]

    sampler = optuna.samplers.TPESampler(seed=params["optuna_seed"])
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(
        lambda trial: _objective_unified(trial, X_optuna, y_optuna, params["n_splits_cv_hyperopt"], pred_type),
        n_trials=params["optuna_n_trials"],
        show_progress_bar=False  
    )
    
    log.info(f"Best trial for {pred_type}: {study.best_trial.value}")
    return study.best_params


def train_model(X: pd.DataFrame, y: pd.Series, best_params: Dict[str, Any], pred_type: Any) -> lgb.LGBMRegressor:
    """Trains a final LightGBM model with the best parameters."""
    log.info(f"--- Training final model for target: {pred_type} ---")
    
    if pred_type == 'mean':
        final_params = {'objective': 'regression_l2', 'metric': 'mae', 'random_state': 42, 'n_jobs': n_jobs, 'verbose': -1, 'boosting_type': 'gbdt'}
    else:
        final_params = {'objective': 'quantile', 'alpha': pred_type, 'metric': 'quantile', 'random_state': 42, 'n_jobs': n_jobs, 'verbose': -1, 'boosting_type': 'gbdt'}
    
    final_params.update(best_params)
    
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X, y)
    
    log.info(f"Final model for {pred_type} trained.")
    return model


def create_oos_predictions(X: pd.DataFrame, y: pd.Series, model: lgb.LGBMRegressor, pred_type: Any, params: Dict[str, Any]) -> pd.Series:
    """Creates Out-of-Sample (OOS) predictions using the trained model."""
    log.info(f"--- Creating OOS predictions for target: {pred_type} ---")
    
    predictions_list = []
    tscv_oos = TimeSeriesSplit(n_splits=params["n_splits_cv_oos"])
    
    for train_idx, val_idx in tscv_oos.split(X):
        if len(train_idx) == 0 or len(val_idx) == 0: continue
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        
        # Re-train the model on each fold to simulate a real-world scenario
        model_fold = lgb.LGBMRegressor(**model.get_params())
        model_fold.fit(X_train, y_train)
        
        preds = model_fold.predict(X_val)
        predictions_list.append(pd.Series(preds, index=X_val.index))

    df_oos_predictions = pd.concat(predictions_list).sort_index()
    df_oos_predictions = df_oos_predictions[~df_oos_predictions.index.duplicated(keep='last')]
    
    col_name = f'm1_predicted_q{pred_type}' if isinstance(pred_type, float) else f'm1_predicted_{pred_type}'
    df_oos_predictions.name = col_name
    
    log.info(f"OOS predictions for {pred_type} created.")
    return df_oos_predictions


def combine_predictions(*predictions: pd.Series) -> pd.DataFrame:
    """Combines all OOS prediction Series into a single DataFrame."""
    log.info("--- Combining all OOS predictions ---")
    df = pd.concat(predictions, axis=1).sort_index()
    df = df.reset_index().rename(columns={'index': 'cob_dt'})
    return df



# --- Model 2 ---

def create_model_2_training_data(
    m1_oos_predictions: pd.DataFrame,
    daily_actuals: pd.DataFrame,
    intraday_features: pd.DataFrame,
    params: Dict[str, Any]
) -> pd.DataFrame:
    """Combines all necessary data sources to create the base table for M2 training."""
    log.info("--- [M2] Creating base data for Model 2 training ---")
    
    # Prepare daily actuals
    df_actuals = daily_actuals[["cob_dt", params["actual_m1_target_col"]]].copy()
    df_actuals = df_actuals.rename(columns={params["actual_m1_target_col"]: 'actual_m1_target'})
    df_actuals['cob_dt'] = pd.to_datetime(df_actuals['cob_dt'])
    df_actuals = df_actuals.set_index('cob_dt').sort_index()

    # Prepare M1 predictions
    df_m1_preds = m1_oos_predictions.copy()
    df_m1_preds['cob_dt'] = pd.to_datetime(df_m1_preds['cob_dt'])
    df_m1_preds = df_m1_preds.set_index('cob_dt').sort_index()

    # Join actuals and predictions
    df_for_m2 = df_actuals.join(df_m1_preds, how='inner')
    
    # Join intraday features
    if intraday_features is not None and not intraday_features.empty:
        df_intraday = intraday_features.reset_index()
        df_intraday['cob_dt'] = pd.to_datetime(df_intraday['cob_dt'])
        df_intraday = df_intraday.set_index('cob_dt')
        df_for_m2 = df_for_m2.join(df_intraday, how='left')
        log.info(f"[M2] Joined {df_intraday.shape[1]} intraday features.")

    # Fill any NaNs that might result from the join
    df_for_m2.fillna(0, inplace=True)
    return df_for_m2


def create_model_2_common_features(df: pd.DataFrame) -> (pd.DataFrame, str):
    """Creates features that are common to all M2 corrector models."""
    log.info("--- [M2] Creating common features for M2 ---")
    df_features = df.copy()
    
    # Feature describing M1's uncertainty
    df_features['m1_pred_interval_width'] = df_features['m1_predicted_q0.9'] - df_features['m1_predicted_q0.1']
    
    cols_to_exclude = [col for col in df_features.columns if 'actual' in col or 'm1_predicted' in col]
    common_feature_cols = [col for col in df_features.columns if col not in cols_to_exclude]
    
    X_common = df_features[common_feature_cols]
    features_as_string = "\n".join(common_feature_cols)
    log.info(f"[M2] Created {len(common_feature_cols)} common features.")
    
    return X_common, features_as_string


def create_error_target_for_model_2(df: pd.DataFrame, target_type: str) -> pd.Series:
    """Calculates the specific error that the M2 model needs to predict."""
    log.info(f"--- [M2] Creating error target for '{target_type}' ---")
    if target_type == 'median':
        m1_prediction_col = 'm1_predicted_q0.5'
    elif target_type == 'mean':
        m1_prediction_col = 'm1_predicted_mean'
    else:
        raise ValueError(f"Unknown target type for M2: {target_type}")

    target_col_m2 = df['actual_m1_target'] - df[m1_prediction_col]
    target_col_m2.name = f'error_{target_type}'
    return target_col_m2


def _objective_m2(trial, X_train, y_train, n_splits_cv, objective_type):
    """Internal objective function for M2 hyperparameter tuning."""
    params = {
        'objective': objective_type, 'metric': 'mae', 'random_state': 42, 'n_jobs': n_jobs,
        'verbose': -1, 'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 80),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
    }
    tscv = TimeSeriesSplit(n_splits=n_splits_cv)
    mae_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        if len(train_idx) == 0 or len(val_idx) == 0: continue
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                  eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                  eval_metric='mae', callbacks=[lgb.early_stopping(20, verbose=False)])
        preds = model.predict(X_train.iloc[val_idx])
        mae_scores.append(mean_absolute_error(y_train.iloc[val_idx], preds))
    return np.mean(mae_scores) if mae_scores else float('inf')


def tune_hyperparameters_m2(X: pd.DataFrame, y: pd.Series, target_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Tunes hyperparameters for an M2 corrector model."""
    log.info(f"--- [M2] Tuning hyperparameters for '{target_type}' corrector ---")
    cutoff_idx = int(len(X) * params["train_ratio"])
    X_optuna, y_optuna = X.iloc[:cutoff_idx], y.iloc[:cutoff_idx]
    
    objective_name = 'regression_l1' if target_type == 'median' else 'regression_l2'

    sampler = optuna.samplers.TPESampler(seed=params["optuna_seed"])
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(
        lambda trial: _objective_m2(trial, X_optuna, y_optuna, params["n_splits_cv_hyperopt"], objective_name),
        n_trials=params["optuna_n_trials"]
    )
    return study.best_params


def train_corrector_model_m2(X: pd.DataFrame, y: pd.Series, best_params: Dict[str, Any], target_type: str) -> lgb.LGBMRegressor:
    """Trains the final M2 error corrector model."""
    log.info(f"--- [M2] Training final corrector model for '{target_type}' ---")
    objective_name = 'regression_l1' if target_type == 'median' else 'regression_l2'
    
    final_params = {
        **best_params,
        'objective': objective_name,
        'random_state': 42, 'n_jobs': n_jobs, 'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X, y)
    return model
