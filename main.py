import torch
import numpy as np
import os
import time

# 論文で定義されたモジュールをインポート
import config
import data_loader
import preprocessor
import dae_model
import ddqn_model

# 論文 4.3.2 比較用の既存手法
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# PyTorchの決定論的動作（再現性のため）
torch.manual_seed(42)
np.random.seed(42)

# --- 定数 ---
DAE_MODEL_PATH = "dae_model.pth"
DDQN_MODEL_PATH = "ddqn_model.pth"
ENCODED_DIM = 64 # DAEの圧縮次元

def phase_1_data_loading():
    """
    フェーズ1: データのロードと分割 (論文 4.1, 4.2.1 (1))
    """
    print("\n--- [フェーズ1 開始] ---")
    
    # 1. データのロード
    known_df, unknown_df = data_loader.load_and_split_data(
        'data/KDDTrain+.txt', 
        'data/KDDTest+.txt',
        config.COLUMNS,
        config.KNOWN_CLASSES,
        config.UNKNOWN_CLASSES,
        config.DATA_SAMPLE_FRACTION
    )
    
    print(f"既知のデータセットサイズ: {len(known_df)} 件")
    print(f"未知のデータセットサイズ: {len(unknown_df)} 件")
    print("\n--- [フェーズ1 完了] ---")
    return known_df, unknown_df

def phase_2_preprocessing(known_df, unknown_df):
    """
    フェーズ2: 前処理 (論文 3.2, 4.2.1 (2), (3))
    (注: サンプリング(4)は、DAEの後(フェーズ4)と既存手法(フェーズ6)に移動)
    """
    print("\n--- [フェーズ2 開始] ---")
    
    # 1. ラベルをカテゴリ名に変換 (論文 4.2.1 (3))
    print("ステップ 2.1: ラベルをカテゴリに変換中...")
    y_known_str, y_unknown_str, X_known, X_unknown = \
        data_loader.transform_labels(known_df, unknown_df, config.ATTACK_CATEGORY_MAP)
    
    print("\n既知データセットのカテゴリ分布 (処理前):")
    print(y_known_str.value_counts().sort_index())
    
    # 2. ラベルを数値にエンコード
    y_known_numeric, y_unknown_numeric, label_encoder = \
        preprocessor.create_label_encoder(y_known_str, y_unknown_str)

    # 3. 特徴量の標準化とOne-Hotエンコード (論文 4.2.1 (2))
    print("\nステップ 2.2: 特徴量のエンコーディングと標準化...")
    feature_pipeline = preprocessor.create_feature_preprocessor()
    
    print("既知のデータ (X_known) を使って前処理パイプラインを学習 (fit)...")
    feature_pipeline.fit(X_known)
    
    print("学習したパイプラインで X_known と X_unknown を変換 (transform)...")
    X_known_processed = feature_pipeline.transform(X_known)
    X_unknown_processed = feature_pipeline.transform(X_unknown)
    
    processed_cols = X_known_processed.shape[1]
    print(f"前処理後の既知データの形状 (行, 列): {X_known_processed.shape}")
    print(f"前処理後の未知データの形状 (行, 列): {X_unknown_processed.shape}")
    
    print("\n--- [フェーズ2 (サンプリング前処理) 完了] ---")
    
    return (X_known_processed, y_known_numeric, 
            X_unknown_processed, y_unknown_numeric, 
            label_encoder, processed_cols)

def phase_3_dae_learning(X_known_processed, input_dim, device):
    """
    フェーズ3: DAEモデルの学習 (論文 3.3, 4.2.2)
    (注: サンプリング前の「本物の」データ(X_known_processed)で学習する)
    """
    print("\n--- [フェーズ3 開始 (DAEモデルの学習)] ---")
    
    if os.path.exists(DAE_MODEL_PATH):
        print(f"{DAE_MODEL_PATH} を発見。学習済みモデルをロードします。")
        trained_dae = dae_model.DenoisingAutoencoder(input_dim, ENCODED_DIM).to(device)
        try:
            trained_dae.load_state_dict(torch.load(DAE_MODEL_PATH, map_location=device))
        except RuntimeError as e:
            print(f"[エラー] {DAE_MODEL_PATH} のロードに失敗しました。モデルの構造が変更された可能性があります。")
            print(f"エラー詳細: {e}")
            print(f"古い {DAE_MODEL_PATH} を削除して、再学習します。")
            os.remove(DAE_MODEL_PATH)
            # 再帰呼び出しで学習からやり直し
            return phase_3_dae_learning(X_known_processed, input_dim, device)
    else:
        print(f"{DAE_MODEL_PATH} が見つかりません。DAEの学習を開始します...")
        trained_dae = dae_model.train_dae(
            X_known_processed, # サンプリング前のデータ (14万件)
            input_dim=input_dim,
            device=device,
            epochs=config.DAE_EPOCHS,
            noise_factor=config.DAE_NOISE_FACTOR
        )
        print(f"学習完了。モデルを {DAE_MODEL_PATH} に保存します。")
        torch.save(trained_dae.state_dict(), DAE_MODEL_PATH)
        
    print("\n--- [フェーズ3 完了 (DAEモデル準備OK)] ---")
    return trained_dae.eval()

def phase_4_ddqn_learning(trained_dae, X_known_processed, y_known_numeric, device):
    """
    フェーズ4: DAE特徴量抽出し、SMOTE+ENNでサンプリングし、DDQNを学習 (論文 3.4, 4.2.3)
    """
    print("\n--- [フェーズ4 開始 (特徴量抽出 & DDQN学習)] ---")
    
    # 1. DAEで特徴量抽出 (122次元 -> 64次元)
    print("DAEエンコーダで「既知データ」の特徴量を抽出中...")
    with torch.no_grad():
        known_features_tensor = torch.from_numpy(X_known_processed.astype(np.float32)).to(device)
        known_features = trained_dae.encoder(known_features_tensor).cpu().numpy()
        
    print(f"  DAE特徴量 形状: {known_features.shape}")

    # 2. サンプリング (SMOTE + ENN) (論文 4.2.1 (4))
    # (DAEで抽出した 64次元 の特徴量をサンプリング)
    print("\nステップ 4.1: DAE特徴量のサンプリング (SMOTE + ENN) を実行中...")
    known_features_resampled, y_known_resampled = preprocessor.apply_resampling(
        known_features,
        y_known_numeric
    )

    # 3. DDQNモデルの学習 (論文 4.2.3)
    print("\nステップ 4.2: DDQNモデルの学習...")
    if os.path.exists(DDQN_MODEL_PATH):
        print(f"{DDQN_MODEL_PATH} を発見。学習済みモデルをロードします。")
        trained_ddqn = ddqn_model.DDQN_Network(ENCODED_DIM, config.NUM_CLASSES).to(device)
        trained_ddqn.load_state_dict(torch.load(DDQN_MODEL_PATH, map_location=device))
    else:
        print(f"{DDQN_MODEL_PATH} が見つかりません。DDQNの学習を開始します...")
        
        # DDQN学習のため、サンプリング後のデータをシャッフル (論文 4.2.3)
        print("DDQN学習のため、サンプリング後の既知データをシャッフル中...")
        indices = np.random.permutation(len(known_features_resampled))
        X_ddqn_train = known_features_resampled[indices]
        y_ddqn_train = y_known_resampled[indices]
        
        trained_ddqn = ddqn_model.train_ddqn(
            X_ddqn_train,
            y_ddqn_train,
            input_dim=ENCODED_DIM,
            # ★★★ 修正箇所 ★★★
            # 'output_dim' を 'num_classes' に変更
            num_classes=config.NUM_CLASSES, 
            # ★★★★★★★★★★★★
            device=device,
            epochs=config.DDQN_EPOCHS,
            gamma=config.DDQN_GAMMA,
            learning_rate=config.DDQN_LR
        )
        print(f"学習完了。モデルを {DDQN_MODEL_PATH} に保存します。")
        torch.save(trained_ddqn.state_dict(), DDQN_MODEL_PATH)

    print("\n--- [フェーズ4 完了 (DDQNモデル準備OK)] ---")
    return trained_ddqn.eval()

def phase_5_evaluate_proposed(trained_dae, trained_ddqn, X_unknown_processed, y_unknown_numeric, label_encoder, device):
    """
    フェーズ5: 提案手法 (DAE+DDQN) の予測と評価 (論文 4.2.4, 4.3.1)
    """
    print("\n--- [フェーズ5 開始 (提案手法 DAE+DDQN の予測と評価)] ---")
    
    # 1. 未知データの特徴量抽出
    with torch.no_grad():
        unknown_features_tensor = torch.from_numpy(X_unknown_processed.astype(np.float32)).to(device)
        unknown_features = trained_dae.encoder(unknown_features_tensor) # (7536, 64)
    
    # 2. DDQNで予測
    with torch.no_grad():
        q_values = trained_ddqn(unknown_features) # (7536, 5)
        y_pred_numeric = torch.argmax(q_values, dim=1).cpu().numpy()

    # 3. 評価
    print("\n--- 提案手法 (DAE+DDQN) の予測性能 ---")
    
    # 論文[表6]の「マイクロ平均」 (Accuracy)
    accuracy = accuracy_score(y_unknown_numeric, y_pred_numeric)
    print(f"\nマイクロ平均 (正解率): {accuracy*100:.2f}%")
    paper_result = config.PAPER_RESULTS.get("提案手法(DAE+DDQN)", "N/A")
    print(f" (論文 [表6] \"提案手法(DAE+DDQN)\": {paper_result})")

    # 論文[図6]のカテゴリ別評価
    print("\nカテゴリ別 適合率 (Precision), 再現率 (Recall), F値:")
    report = classification_report(
        y_unknown_numeric, 
        y_pred_numeric,
        target_names=label_encoder.classes_,
        labels=np.arange(len(label_encoder.classes_)), # 5クラス強制表示
        digits=3,
        zero_division=0
    )
    print(report)
    print("\n--- [フェーズ5 完了] ---")

def run_comparison_models(X_known_processed, y_known_numeric, X_unknown_processed, y_unknown_numeric, label_encoder):
    """
    フェーズ6: 既存手法との比較 (論文 4.3.2)
    (注: 122次元の元データをサンプリングして学習)
    """
    print("\n--- [フェーズ6: 既存手法との比較 (論文 4.3.2)] ---")

    print("\nステップ 6.1: 既存手法用のサンプリング (SMOTE + ENN) を実行中...")
    # (122次元の元データをサンプリング)
    X_known_resampled_122, y_known_resampled_122 = preprocessor.apply_resampling(
        X_known_processed,
        y_known_numeric
    )
    
    # 比較モデルの定義 (論文で言及されているもの)
    models = {
        "ニューラルネットワーク (MLP)": MLPClassifier(random_state=42, max_iter=200, hidden_layer_sizes=(100,)),
        "サポートベクターマシン (LinearSVC)": LinearSVC(random_state=42, max_iter=5000, dual="auto"),
        "ランダムフォレスト": RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    }

    for name, model in models.items():
        print(f"\n--- 既存手法: {name} の学習・評価 ---")
        print(f"{name} の学習中... (時間がかかる場合があります)")
        start_time = time.time()
        
        # 1. 122次元のサンプリング済みデータで学習
        model.fit(X_known_resampled_122, y_known_resampled_122)
        
        print(f"学習完了 (所要時間: {time.time() - start_time:.1f} 秒)")
        print("未知の攻撃を予測中...")
        
        # 2. 122次元の未知データで予測
        y_pred_numeric = model.predict(X_unknown_processed)
        
        # 3. 評価
        accuracy = accuracy_score(y_unknown_numeric, y_pred_numeric)
        print(f"\n{name} のマイクロ平均 (正解率): {accuracy*100:.2f}%")
        paper_key = name.split(" ")[0] # "ニューラルネットワーク"
        paper_result = config.PAPER_RESULTS.get(paper_key, "N/A")
        print(f" (論文 [表6] {paper_key}: {paper_result})")
        
        report = classification_report(
            y_unknown_numeric, 
            y_pred_numeric,
            target_names=label_encoder.classes_,
            labels=np.arange(len(label_encoder.classes_)),
            digits=3,
            zero_division=0
        )
        print(report)

    print("\n--- [フェーズ6 完了] ---")


def main():
    """
    メイン実行関数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # --- フェーズ1: データロード ---
    known_df, unknown_df = phase_1_data_loading()

    # --- フェーズ2: 前処理 (サンプリング除く) ---
    (X_known_processed, y_known_numeric, 
     X_unknown_processed, y_unknown_numeric, 
     label_encoder, input_dim) = phase_2_preprocessing(known_df, unknown_df)

    # --- フェーズ3: DAE学習 (サンプリング前のデータを使用) ---
    trained_dae = phase_3_dae_learning(X_known_processed, input_dim, device)

    # --- フェーズ4: DDQN学習 (DAE特徴量をサンプリングして使用) ---
    trained_ddqn = phase_4_ddqn_learning(
        trained_dae, 
        X_known_processed, # DAEエンコーダへの入力
        y_known_numeric,   # サンプリング対象のラベル
        device
    )

    # --- フェーズ5: 提案手法の評価 ---
    phase_5_evaluate_proposed(
        trained_dae, 
        trained_ddqn, 
        X_unknown_processed, 
        y_unknown_numeric, 
        label_encoder, 
        device
    )

    # --- フェーズ6: 既存手法との比較 (122次元データをサンプリングして使用) ---
    run_comparison_models(
        X_known_processed, 
        y_known_numeric, 
        X_unknown_processed, 
        y_unknown_numeric, 
        label_encoder
    )

    print("\n--- [全フェーズ 完了] ---")

if __name__ == "__main__":
    
    # --- config.py に学習パラメータを追加 ---
    # (main.py の実行時に config モジュールに動的に追加)
    
    # 論文 表4 (DAE)
    config.DAE_EPOCHS = 300
    config.DAE_NOISE_FACTOR = 0.1
    
    # 論文 表5 (DDQN)
    config.DDQN_EPOCHS = 10
    config.DDQN_GAMMA = 0.01
    config.DDQN_LR = 0.0001
    
    # 論文 図4 (DDQN)
    config.DDQN_BATCH_SIZE = 512 # 論文に記載はないが、学習効率のため設定
    
    # ラベルエンコーダから取得
    config.NUM_CLASSES = 5 
    
    main()

