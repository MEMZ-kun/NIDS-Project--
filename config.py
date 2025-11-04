"""
カラム名、クラス定義、ハイパーパラメータなどの静的な設定ファイル
"""

# --- 高速化のための設定 ---

# True にすると、論文[4.2.1 (4)]のENN(アンダーサンプリング)を実行します。
# (注: SMOTEと合わせて実行すると、完了までに数十分以上かかる場合があります)
RUN_ENN = True

# True にすると、SMOTE/ENNを実行します (論文通りの設定)
# False にすると、SMOTE/ENNをスキップし、高速に動作確認できます (精度は低下します)
RUN_RESAMPLING = True

# 1.0 未満に設定すると、学習データ(known_df)をこの割合に削減して高速化します
# (例: 0.2 = 20%のデータのみ使用)
# 論文通りの結果を得るには 1.0 に設定します
DATA_SAMPLE_FRACTION = 1.0


# --- NSL-KDD カラム定義 ---
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'attack_class', # これがラベル（正解）です
    'difficulty' # この列は今回使用しません
]

# --- 論文 表3: 「既知」と「未知」のクラス定義 ---
KNOWN_CLASSES = [
    'normal',
    'neptune', 'smurf', 'back', 'apache2', 'processtable', 'mailbomb', 'pod',
    'worm', 'udpstorm',
    'satan', 'portsweep', 'mscan', 'saint',
    'warezmaster', 'warezclient', 'snmpguess', 'snmpgetattack', 'httptunnel',
    'multihop', 'named', 'sendmail', 'ftp_write', 'xlock', 'phf', 'xsnoop', 'spy',
    'buffer_overflow', 'ps', 'xterm', 'loadmodule', 'sqlattack', 'rootkit'
]

UNKNOWN_CLASSES = [
    'teardrop', 'land',
    'ipsweep', 'nmap',
    'guess_passwd', 'imap',
    'perl'
]

# --- 論文 表2 & 表3: クラスからカテゴリへのマッピング ---
ATTACK_CATEGORY_MAP = {
    'normal': 'Normal',
    
    # DoS
    'neptune': 'DoS', 'smurf': 'DoS', 'back': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'mailbomb': 'DoS', 'pod': 'DoS', 'worm': 'DoS',
    'udpstorm': 'DoS', 'teardrop': 'DoS', 'land': 'DoS',
    
    # Probe
    'satan': 'Probe', 'portsweep': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    'ipsweep': 'Probe', 'nmap': 'Probe',
    
    # R2L
    'warezmaster': 'R2L', 'warezclient': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'multihop': 'R2L',
    'named': 'R2L', 'sendmail': 'R2L', 'ftp_write': 'R2L', 'xlock': 'R2L',
    'phf': 'R2L', 'xsnoop': 'R2L', 'spy': 'R2L', 'guess_passwd': 'R2L',
    'imap': 'R2L',
    
    # U2R
    'buffer_overflow': 'U2R', 'ps': 'U2R', 'xterm': 'U2R', 'loadmodule': 'U2R',
    'sqlattack': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R'
}


# --- 論文 4.2.1: 前処理の対象カラム ---
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

# --- 数値特徴量 (COLUMNSから上記とラベルを除外) ---
NUMERICAL_FEATURES = [
    col for col in COLUMNS 
    if col not in CATEGORICAL_FEATURES 
    and col not in ['attack_class', 'difficulty']
]

# --- 論文 表6: 比較用の結果 ---
PAPER_RESULTS = {
    "提案手法(DAE+DDQN)": 0.657,
    "DDQNのみ": 0.537,
    "ニューラルネットワーク": 0.551,
    "サポートベクターマシン": 0.597,
    "ランダムフォレスト": 0.572
}

