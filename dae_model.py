import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Denoising Autoencoder (DAE)
class DenoisingAutoencoder(nn.Module):
    """
    論文 3.3節, 4.2.2節 DAE
    122次元の入力 -> 128次元 -> 64次元 (エンコーダ)
    64次元 -> 128次元 -> 122次元 (デコーダ)
    """
    def __init__(self, input_dim, encoding_dim):
        super(DenoisingAutoencoder, self).__init__()
        
        # エンコーダ
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim), # 圧縮された特徴量
            nn.ReLU()
        )
        
        # デコーダ
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim) # 元の次元に復元
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_dae(X_train_processed, input_dim, device, epochs, noise_factor, batch_size=256):
    """
    DAEモデルの学習 (論文 表4)
    (注: サンプリング前の「本物の」データ(X_train_processed)で学習する)
    """
    print(f"\nDAE学習開始 (デバイス: {device})")
    print(f"DAE学習データ形状: {X_train_processed.shape} (SMOTE適用前)")
    
    # DAEモデルと設定
    model = DenoisingAutoencoder(input_dim, 64).to(device)
    
    # 論文(表4)指定: adadelta
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    # 論文(表4)指定: mse
    criterion = nn.MSELoss()

    # データローダーの準備 (DAEは教師なし学習だが、DataLoaderでバッチ処理する)
    # ラベルは使わないのでダミー (torch.zeros) を設定
    dataset = TensorDataset(
        torch.from_numpy(X_train_processed.astype(np.float32)), 
        torch.zeros(len(X_train_processed))
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 学習ループ
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            
            # 論文(4.2.2)に基づき、入力にノイズを追加
            noise = torch.randn_like(inputs) * noise_factor
            noisy_inputs = inputs + noise
            
            # ★★★ 修正箇所 ★★★
            # noisy_inputs = noisy_inputs.clamp(0., 1.) 
            # ↑ この行を削除。標準化データにはclampは使わない。
            # ★★★★★★★★★★★★
            
            # 1. 順伝播 (ノイズ付き入力 -> 復元出力)
            outputs = model(noisy_inputs)
            
            # 2. 損失計算 (復元出力 と ノイズなし入力 を比較)
            loss = criterion(outputs, inputs)
            
            # 3. 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 50 == 0 or (epoch + 1) == 1:
            print(f"エポック [{epoch + 1}/{epochs}], 損失 (Loss): {avg_loss:.6f}")

    print("DAE学習完了。")
    return model.eval() # 学習済みのモデルを返す

