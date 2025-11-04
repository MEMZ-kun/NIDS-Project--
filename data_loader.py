import pandas as pd
import config

def load_and_split_data(train_file, test_file, columns, known_classes, unknown_classes, sample_fraction=1.0):
    """
    フェーズ1: データをロードし、「既知」と「未知」に分割する
    (論文 4.1, 4.2.1 (1))
    """
    try:
        train_df = pd.read_csv(train_file, header=None, names=columns)
        test_df = pd.read_csv(test_file, header=None, names=columns)
    except FileNotFoundError as e:
        print(f"\n[エラー] データファイルが見つかりません: {e.filename}")
        print("wget コマンドで KDDTrain+.txt と KDDTest+.txt をダウンロードしてください。")
        exit()
    
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df = all_df.drop('difficulty', axis=1)
    print(f"全データ読み込み完了: {len(all_df)} 件")

    # 開発用にサンプリング (デフォルトは 1.0 = 100%)
    if sample_fraction < 1.0:
        print(f"[開発モード] データセットを {sample_fraction*100:.0f}% に削減します...")
        all_df = all_df.sample(frac=sample_fraction, random_state=42)
        print(f"削減後のデータサイズ: {len(all_df)} 件")

    # --- 4. データセットの分割 ---
    known_df = all_df[all_df['attack_class'].isin(known_classes)].copy()
    unknown_df = all_df[all_df['attack_class'].isin(unknown_classes)].copy()

    return known_df, unknown_df

def transform_labels(known_df, unknown_df, attack_map):
    """
    フェーズ2.1: ラベルをクラス名からカテゴリ名に変換
    (論文 4.2.1 (3))
    """
    known_df['attack_category'] = known_df['attack_class'].map(attack_map)
    unknown_df['attack_category'] = unknown_df['attack_class'].map(attack_map)

    # 特徴量(X)とラベル(y)に分離
    X_known = known_df.drop(['attack_class', 'attack_category'], axis=1)
    y_known_str = known_df['attack_category']
    
    X_unknown = unknown_df.drop(['attack_class', 'attack_category'], axis=1)
    y_unknown_str = unknown_df['attack_category']
    
    return y_known_str, y_unknown_str, X_known, X_unknown

