import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import warnings
import config

# SklearnのFutureWarningを非表示にする
warnings.filterwarnings('ignore', category=FutureWarning)

def create_label_encoder(y_known_str, y_unknown_str):
    """
    カテゴリ名のラベル('Normal', 'DoS'...)を
    数値(0, 1...)に変換するエンコーダを作成し、変換する
    """
    print("ラベルを 5 個の数値にエンコードしました。")
    label_encoder = LabelEncoder()
    
    # 既知と未知のすべてのカテゴリ名を学習させる
    all_labels = pd.concat([y_known_str, y_unknown_str]).unique()
    label_encoder.fit(all_labels)
    
    print(f"カテゴリ名: {label_encoder.classes_}")
    print(f"対応する数値: {label_encoder.transform(label_encoder.classes_)}")
    
    # 変換
    y_known_numeric = label_encoder.transform(y_known_str)
    y_unknown_numeric = label_encoder.transform(y_unknown_str)
    
    return y_known_numeric, y_unknown_numeric, label_encoder

def create_feature_preprocessor():
    """
    論文 4.2.1 (2) に基づく前処理パイプラインを定義
    - 数値特徴量: 標準化 (StandardScaler)
    - カテゴリ特徴量: One-Hotエンコーディング
    """
    
    # 数値特徴量の処理
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # カテゴリ特徴量の処理
    categorical_transformer = Pipeline(steps=[
        # sparse_output=False にして NumPy 配列を直接返す
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])

    # ColumnTransformer で処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, config.NUMERICAL_FEATURES),
            ('cat', categorical_transformer, config.CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
    return preprocessor

def apply_resampling(X, y):
    """
    論文 4.2.1 (4) に基づくサンプリング (SMOTE + ENN) を実行する
    
    引数:
    X (np.array): 変換前の特徴量
    y (np.array): 変換前のラベル
    
    戻り値:
    X_resampled (np.array): サンプリング後の特徴量
    y_resampled (np.array): サンプリング後のラベル
    """
    if not config.RUN_RESAMPLING:
        print("サンプリング (SMOTE + ENN) はスキップされました。")
        return X, y
        
    print(f"サンプリング前のデータ形状: {X.shape}")
    print("サンプリング前のラベル分布 (数値):")
    print(pd.Series(y).value_counts().sort_index())

    # 1. SMOTE (k_neighborsは最小クラス数より小さい必要がある)
    min_class_size = pd.Series(y).value_counts().min()
    # 最小クラス数が1の場合(U2Rの未知データなど)、k_neighbors=1 に設定できないため調整
    k_neighbors = min(5, min_class_size - 1)
    if k_neighbors < 1:
        print(f"警告: 最小クラス数が {min_class_size} のため、SMOTEが実行できません。k_neighbors=1 に設定します。")
        k_neighbors = 1
        
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

    # 2. ENN
    if config.RUN_ENN:
        print("ENN (アンダーサンプリング) を有効にして実行します。(時間がかかります)")
        enn = EditedNearestNeighbours(sampling_strategy='all')
        resampling_pipeline = ImbPipeline([
            ('smote', smote),
            ('enn', enn)
        ])
    else:
        print("ENN (アンダーサンプリング) はスキップされました。")
        resampling_pipeline = ImbPipeline([('smote', smote)])

    # 3. サンプリング実行
    try:
        X_resampled, y_resampled = resampling_pipeline.fit_resample(X, y)
    except ValueError as e:
        print(f"\n[エラー] サンプリング中にエラーが発生しました: {e}")
        print("config.py の RUN_RESAMPLING=False を試してください。")
        exit()

    print(f"\nサンプリング後のデータ形式: {type(X_resampled)}")
    print(f"サンプリング後のデータ形状: {X_resampled.shape}")
    print("サンプリング後のラベル分布 (数値):")
    print(pd.Series(y_resampled).value_counts().sort_index())
    
    return X_resampled, y_resampled

