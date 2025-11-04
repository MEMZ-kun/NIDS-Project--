# NIDS（機械学習を用いた未知の攻撃検知手法）

このプロジェクトは、情報処理学会論文誌 Vol.62 No.12
「機械学習を用いたNIDSにおける未知の攻撃検知手法の提案」の実装を再現したものです。

論文の手法に基づき、DAE（ノイズ除去オートエンコーダ）による特徴量抽出と、
DDQN（深層強化学習）による分類を組み合わせ、NSL-KDDデータセットに含まれる
「未知の攻撃」を分別・検知します。

動作環境

OS: Ubuntu 24.04.3 LTS (ARM 版)

パッケージ管理: Conda 25.7.0

Python: 3.11

1. 環境構築

本プロジェクトは、Conda を使用した仮想環境での実行を推奨します。

ステップ 1: Conda 仮想環境の作成

ターミナルで nids_project ディレクトリの親ディレクトリ等で実行します。

'nids_attack' という名前で Python 3.11 の環境を作成
conda create -n nids_attack python=3.11

作成した環境をアクティベート
conda activate nids_attack


ステップ 2: 必須ライブラリのインストール

(nids_attack) 環境がアクティブな状態で実行します。

1. 基本ライブラリのインストール (Conda)

conda install -c conda-forge pandas numpy scikit-learn imbalanced-learn


2. PyTorch（深層学習ライブラリ）のインストール (pip)

ARM 版 Ubuntu では、Conda 標準チャネルの PyTorch で互換性問題が起きるため pip を使用します。

pip install torch torchvision torchaudio


3. インストールの確認

python -c "import torch; import pandas; import sklearn; import imblearn; print('\n✅ ライブラリの準備が完了しました！')"


2. データセットの準備

実行には NSL-KDD データセットが必要です。
nids_project ディレクトリ内で以下のコマンドを実行し、データファイルをダウンロードしてください。

（注：data ディレクトリを作成し、その下にダウンロードすることを推奨します。
.gitignore により data/ フォルダはアップロード対象外です。）

mkdir data
cd data

KDDTrain+.txt (学習データセット)
wget https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt

KDDTest+.txt (テストデータセット)
wget https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt

cd ..


(もし data ディレクトリに保存した場合は、data_loader.py 内のファイルパス "KDDTrain+.txt" を "data/KDDTrain+.txt" に変更してください。)

3. 実行方法

すべての .py ファイルが nids_project ディレクトリにあることを確認し、以下を実行します。

(nids_attack) 環境がアクティブであることを確認
conda activate nids_attack

メインプログラムを実行
python main.py


再学習（モデルの削除）

dae_model.py や ddqn_model.py を変更した場合、学習済みの古い .pth ファイルが
エラーの原因となることがあります。以下で削除してから再実行してください。

DAE と DDQN の両方を再学習する場合
rm dae_model.pth
rm ddqn_model.pth

再実行
python main.py

-補足-
私自身が実行した結果は「kekka.txt」に記載しましたが、環境等の要因により大きく変動する可能性があります。
また、プログラミング内容の一部が論文内で提示されている手法を満たしてない可能性があります。
