from datetime import datetime, timedelta

import numpy as np
from sklearn.model_selection import KFold
from pytorch_tabnet.tab_model import TabNetRegressor
from torch.optim.lr_scheduler import ReduceLROnPlateau

SEED = 42

if __name__ == '__main__':
    train_dataset = np.load(
        '/workspace/20_MoA/tabnet/dataset/YearPredictionMSD_tr.npz')
    test_dataset = np.load(
        '/workspace/20_MoA/tabnet/dataset/YearPredictionMSD_te.npz')
    X_train_org, y_train_org, X_test, y_test = (
        train_dataset['features'], train_dataset['labels'].reshape(-1, 1),
        test_dataset['features'], test_dataset['labels'].reshape(-1, 1)
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for _fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_org, y_train_org)):
        print(f"SEED [{SEED}] FOLD [{_fold:02d}]")
        X_train, y_train, X_valid, y_valid = (
            X_train_org[train_idx], y_train_org[train_idx],
            X_train_org[valid_idx], y_train_org[valid_idx]
        )
        model = TabNetRegressor(
            n_d=128,
            n_a=32,
            n_steps=1,
            gamma=1.0,
            n_independent=1,
            n_shared=1,
            cat_emb_dim=0,
            # scheduler_fn=ReduceLROnPlateau,
            epsilon=1e-15,
            seed=SEED,
            verbose=1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=[('VALID')],
            eval_metric=['mse'],
            max_epochs=10000,
            patience=1000
        )
        
        model.save_model(f'./saved_models/{SEED}_{_fold}_{(datetime.utcnow() + timedelta(hours=9)).strftime("%Y%m%d")}')
        
        
