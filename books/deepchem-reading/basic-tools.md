---
title: "Ch1: Basic Tools — 3行で溶解度予測モデルを動かす"
free: true
---


# アイデア

化学・生命科学向けのOSS [DeepChem](https://github.com/deepchem/deepchem) を、公式チュートリアルを確認しながら読んでいく。
<br>
今回は Chapter 1 ([The_Basic_Tools_of_the_Deep_Life_Sciences.ipynb](https://github.com/deepchem/deepchem/blob/master/examples/tutorials/The_Basic_Tools_of_the_Deep_Life_Sciences.ipynb)) 

### 学ぶこと

- DeepChem の3分クッキング: `load_delaney → GraphConvModel → fit → evaluate` だけで Delaney溶解度データセット (ESOL) に対する グラフ畳み込みネットワーク (GCN) が学習・評価できる
- SMILES → RDKit Mol → GraphConvFeaturizer → 原子特徴行列+隣接行列、というパイプラインの確認


# そもそも

DeepChem は 2017年から続く化学・生命科学向けの ML フレームワークらしく、MIT License で誰でも使える ([github.com/deepchem/deepchem](https://github.com/deepchem/deepchem))。

特徴は「化学の前処理をライブラリ側に寄せて、ユーザは `dc.models.GraphConvModel(...)` を呼ぶだけで生命科学の良さそうなモデルを試せる」ところ。

公式チュートリアルは73本あるが、本シリーズはまず Ch.1 を単独で動かすところから始める。


# 試す

## インストール

公式チュートリアル ([The_Basic_Tools_of_the_Deep_Life_Sciences.ipynb](https://github.com/deepchem/deepchem/blob/master/examples/tutorials/The_Basic_Tools_of_the_Deep_Life_Sciences.ipynb)) は ipynb なので、本シリーズも **Colab / Jupyter で開いて 1 セル目から実行する**前提で進める。最初のセルにこれを貼る:

## 動作確認: 3行クッキング

DeepChem が他のライブラリと違うのは「3行で生命科学のSOTAっぽいモデルが回る」を売りにしていること。

```python
import deepchem as dc

# データセットのロード (MoleculeNetから)
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

# モデル定義
model = dc.models.GraphConvModel(n_tasks=1, mode='regression', dropout=0.2)

# 学習
model.fit(train_dataset, nb_epoch=100)

# 評価
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
print("Test R²:", model.evaluate(test_dataset, [metric], transformers))
```

これだけで Delaney溶解度データセット (ESOL, 1128分子 + LogS) に対する グラフ畳み込みネットワーク (GCN) が学習・評価できる。


# Chapter 1: Basic Tools of the Deep Life Sciences

## 1. `load_delaney(featurizer='GraphConv')` の中身

呼び出した瞬間に裏で起きていること、を 3 ステップに分解する。


**① 分子記述文字列からグラフ表現に変換 — 特徴量化**

SMILES を RDKit という化学計算ライブラリで分子オブジェクトに変換し、続いて特徴量化器が「原子ごとの特徴ベクトル (元素・結合数・電荷・芳香族かどうか、など)」と「どの原子とどの原子が繋がっているかの隣接情報」のペアを返す。これがそのままグラフ畳み込みネットワーク (GCN) の入力になる形。ユーザは分子記述文字列も RDKit も直接は触らない。

② 学習用 / 検証用 / テスト用に 80:10:10 で分割 — 既定は「骨格分割

データ読み込み関数の既定の分割方法は、よくある「ランダム分割」ではなく「骨格分割」。違いはざっくりこういうこと:

- **骨格分割**: 分子の**中心の骨格構造** 
<br>置換基を取り除いた中央の環システム。発案者の名前から「ベミス・マーコ骨格」と呼ばれるもので先にグループ分けし、骨格ごとまとめて学習用かテスト用に振り分ける。たとえば「学習用ではステロイド系の骨格を学習 → テスト用では別系統の骨格で予測」のような状況になるので、「学習で見たことのない骨格にも当てられるか」を測る厳しめの評価**になる


**③ 目的変数 (予測したい値) を正規化する**

回帰タスクなので、戻り値の3番目に「正規化変換器」が入る。学習用データの水溶解度の値から平均 μ・標準偏差 σ を計算し、すべての分割の目的変数を `(値 − μ) / σ` に変換する (平均 0・分散 1 にそろえる)。これでモデル学習の収束が安定する。

---

つまりユーザは分子記述文字列を一度も自分の手で触らずに、本来なら 100 行以上書く「データ取得 → パース → 特徴量化 → 骨格分割 → 目的変数の正規化」が関数 1 つで呼べる。

## 2. `GraphConvModel(...)` の中身

呼び出した瞬間に 「分子グラフを直接受け取って数値を予測するニューラルネットワーク」が組み立てられる。元になったのは Duvenaud らの neural fingerprints (2015) という論文で、画像処理の CNN (畳み込みニューラルネット) と同じ発想を分子グラフに広げたモデル。
論文: [04_papers/pdfs/04_neural_fingerprints_2015_duvenaud.pdf](https://arxiv.org/abs/1509.09292)

引数の意味:

**`n_tasks=1` — 同時に何種類の値を予測するか

「**1分子から、何種類の数値を一度に当てたいか**」を決める引数。今回は「水溶解度」1 種類だけ予測したいので1。

**`mode='regression'` — 予測したいのが「数値」か「カテゴリ」か**

予測対象が**実数値**なのか**ラベル (カテゴリ)** なのかを指定する。値は基本 2 種類:

**`dropout=0.2` — モデルに「丸暗記」させない工夫**

「モデルがデータの細かい癖まで丸暗記してしまう (= 過学習)」のを防ぐ仕組み。

## 3. `model.fit(...)` のバックエンド

呼ぶと実際に学習が始まる。


# 雑感

`load_delaney → GraphConvModel → fit → evaluate` の3行クッキングで、SMILES → 溶解度予測まで一直線で到達できる。次回以降は `dc.Dataset` の中身や Splitter の選び方などをやってみたい

---

> Code adapted from [DeepChem](https://github.com/deepchem/deepchem) (MIT License, Copyright 2017 PandeLab). Original tutorial: [The_Basic_Tools_of_the_Deep_Life_Sciences.ipynb](https://github.com/deepchem/deepchem/blob/master/examples/tutorials/The_Basic_Tools_of_the_Deep_Life_Sciences.ipynb).
