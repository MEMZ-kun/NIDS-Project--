import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import config

class DDQN_Network(nn.Module):
    """
    論文 3.4節, 4.2.3節 DDQN
    DAEで圧縮された64次元の特徴量を入力とし、5クラスのQ値を出力する
    """
    def __init__(self, input_dim, output_dim):
        super(DDQN_Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim) # 5クラス (Normal, DoS, ...) のQ値
        )

    def forward(self, x):
        return self.network(x)

def train_ddqn(X_train, y_train, input_dim, num_classes, device, epochs, gamma, learning_rate):
    """
    DDQNモデルの学習 (論文 図5, 表5)
    
    引数:
    X_train (np.array): DAE特徴量 (シャッフル済み)
    y_train (np.array): ラベル (シャッフル済み)
    input_dim (int): 入力次元 (DAEの圧縮次元, 64)
    num_classes (int): 出力次元 (5)
    """
    print(f"\nDDQN学習開始 (デバイス: {device})")
    
    # 1. ネットワークの定義 (Q_current と Q_target)
    q_network = DDQN_Network(input_dim, num_classes).to(device)
    target_network = DDQN_Network(input_dim, num_classes).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval() # ターゲットネットワークは評価モード

    # 論文(表5)指定: Adam
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    # 論文(表5)指定: huberloss
    criterion = nn.HuberLoss()

    # 2. データセットの準備 (論文 図4)
    # 状態 s_t, 行動 a_t, 次の状態 s_{t+1} の組を作成
    X_s_t = torch.tensor(X_train[:-1], dtype=torch.float32)
    X_s_t_plus_1 = torch.tensor(X_train[1:], dtype=torch.float32)
    y_a_t = torch.tensor(y_train[:-1], dtype=torch.int64)
    
    dataset = TensorDataset(X_s_t, y_a_t, X_s_t_plus_1)
    train_loader = DataLoader(dataset, batch_size=config.DDQN_BATCH_SIZE, shuffle=False) # 事前にシャッフル済み

    # 3. 学習ループ (論文 図5)
    # (論文(表5)指定: 10試行)
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for s_t, a_t, s_t_plus_1 in train_loader:
            s_t = s_t.to(device)
            a_t = a_t.to(device)
            s_t_plus_1 = s_t_plus_1.to(device)
            
            q_network.train()
            optimizer.zero_grad()

            # --- 論文 図5 (1)~(9) の実装 ---
            
            # (1) Q_current が状態 s_t での全行動のQ値 Q(s_t, {a}) を予測
            all_q_values_t = q_network(s_t) # (batch_size, 5)
            
            # (2) Policy (argmax) で s_t での最善の行動 a_t' を選択
            a_t_prime = torch.argmax(all_q_values_t, dim=1) # (batch_size,)

            # (3) 報酬 r_t を計算
            # 予測した行動(a_t') と 正解の行動(a_t) が一致すれば報酬 1, 異なれば 0
            r_t = (a_t_prime == a_t).float() # (batch_size,)
            
            # (4) Q_current が s_{t+1} での全行動のQ値 Q(s_{t+1}, {a}) を予測
            all_q_values_t_plus_1_current = q_network(s_t_plus_1)
            
            # (5) Policy (argmax) で s_{t+1} での最善の行動 a_{t+1}' を選択 (Q_currentネットワークで)
            a_t_plus_1_prime = torch.argmax(all_q_values_t_plus_1_current, dim=1)

            # (6) Q_current から、予測した行動 a_t' のQ値 (q_t') を取得
            # ★★★ 論文 図5(6) に基づく修正 ★★★
            q_t = all_q_values_t.gather(1, a_t_prime.unsqueeze(1)).squeeze(1)

            # (7) Q_target から、a_{t+1}' のQ値 (q'_{t+1}) を取得
            with torch.no_grad():
                all_q_values_t_plus_1_target = target_network(s_t_plus_1)
                q_t_plus_1 = all_q_values_t_plus_1_target.gather(1, a_t_plus_1_prime.unsqueeze(1)).squeeze(1)
            
            # (8) 実際の価値 q_ref を計算 (q_ref = r_t + γ * q'_{t+1})
            q_ref = r_t + (gamma * q_t_plus_1)
            
            # (9) 損失を計算し、Q_current を更新
            loss = criterion(q_t, q_ref)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # Q_target を Q_current の重みで更新 (論文 図5 の最後)
        target_network.load_state_dict(q_network.state_dict())
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"DDQN試行 [{epoch + 1}/{epochs}], 損失 (Loss): {avg_loss:.6f}")

    print("DDQN学習完了。")
    return q_network.eval() # 学習済みの Q_current ネットワークを返す

