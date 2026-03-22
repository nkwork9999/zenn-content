---
title: "NeurIPS Polymer 2025 1位解法の追試で8位相当を達成した話"
emoji: "🧪"
type: "tech"
topics: ["kaggle", "machinelearning", "python", "chemistry", "neurips"]
published: false
---

## はじめに

NeurIPS - Open Polymer Prediction 2025は、ポリマーの化学構造（SMILES）から5つの物性値を予測するKaggleコンペティションです。2,240チームが参加し、賞金総額$50,000。

コンペ終了後に1位のJames Day氏のwriteupを読み、その解法を追試（reproduction）しました。結果、Late Submissionで**Private LB 0.08180**を達成。これは2,240チーム中**実質8位相当**のスコアです。

> **注意:** この記事は1位解法の再現実験です。解法の設計は全てJames Day氏の功績です。目的は再現性の検証と、動くコードベースをコミュニティに提供することです。

**リポジトリ:** [GitHub - NeurIP2025_mytrial_following_1st_solution](https://github.com/nkwork9999/NeurIP2025_mytrial_following_1st_solution)

## コンペ概要

### タスク

ポリマーのSMILES表記から、分子動力学シミュレーションで計算された5つの物性値を予測します。

| Property | 説明 | 単位 |
|----------|------|------|
| Tg | ガラス転移温度 | °C |
| FFV | 自由体積分率 | - |
| Tc | 熱伝導率 | W/(m·K) |
| Density | 密度 | g/cm³ |
| Rg | 慣性半径 | Å |

### スパースラベル問題

このコンペの最大の特徴は、**ラベルが非常にスパース**なことです。全てのポリマーに5つ全ての物性値があるわけではなく、各物性ごとに利用可能なサンプル数が大きく異なります。

### 評価指標

weighted Mean Absolute Error（wMAE）。各物性の値域で正規化し、サンプル数が少ない物性に高い重みを与える設計です。つまり、**データが少ない物性ほど重要**という設計になっています。

## 1位解法の分析

### James Day氏のアプローチ

1位のJames Day氏の解法は、以下の3つのモデルのアンサンブルでした。

1. **ModernBERT-base** — SMILESをテキストとして入力し、回帰ヘッドで物性値を予測
2. **AutoGluon** — RDKit記述子 + Morgan fingerprintを特徴量とするテーブルモデル
3. **Uni-Mol 2** — 分子3D構造を考慮した事前学習モデル

最も意外だったのは、**コード用に事前学習されたModernBERT-baseが、化学特化モデルを上回った**という点です。SMILESは見た目がプログラミングコードに似ており（括弧、記号、繰り返しパターン）、コード用モデルのトークナイザーがSMILESの構造をうまく捉えるようです。

### 2位・3位の知見

- **2位 Ezra氏:** Tgの単位を摂氏から華氏に変換するだけでスコアが激変。ExtraTreesRegressorというシンプルなモデルで高順位
- **3位 hongyu Guo氏:** GATv2Conv（6層）+ Morgan fingerprint。線形回帰によるポストホック校正

## 再現コードの設計判断

### なぜCodeBERTa-small？

1位は`ModernBERT-base`（125Mパラメータ）を使っていましたが、追試では**CodeBERTa-small**（84Mパラメータ）を使いました。

理由:
- Google Colabの無料枠（L4 GPU）で7時間以内に学習を完了させたかった
- 「コード用モデルがSMILESに効く」という仮説の検証には最小モデルで十分
- 結果的に、最小モデルでも1位の90%の性能が出ることがわかった

### パイプライン構成

```
SMILES → CodeBERTa-small → 回帰ヘッド → 5物性値
                                    ↕ (アンサンブル)
SMILES → RDKit特徴量 → AutoGluon → 5物性値
```

各ターゲットについて5-Fold CVで学習し、推論時にはTTA（Test Time Augmentation）を30回適用します。

## Random SMILES拡張

この手法の**最も重要なテクニック**がRandom SMILES拡張です。

### SMILESの非一意性

同じ分子でも、原子の走査順序を変えることで**何百通りものSMILES表現**が可能です。

```
例: エタノール
  Canonical: CCO
  Random 1:  OCC
  Random 2:  C(O)C
```

RDKitの`MolToSmiles(mol, canonical=False, doRandom=True)`を使うと、ランダムなSMILES表記を生成できます。

### 学習時

各エポックで同じ分子を異なるSMILES表記で入力します（10倍拡張）。これにより、モデルは**表記の違いに依存しない分子の本質的な特徴**を学習します。

```python
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_len=128, aug_factor=10):
        self.smiles = smiles_list
        self.labels = labels
        self.aug = aug_factor

    def __len__(self):
        return len(self.smiles) * self.aug

    def __getitem__(self, idx):
        ri = idx % len(self.smiles)
        smi = random_smiles(self.smiles[ri])  # 毎回ランダムに変換
        ...
```

### 推論時（TTA）

テスト時も30回のランダムSMILES変換を行い、**予測値のmedian**を最終予測とします。meanではなくmedianを使うのは、外れ値の影響を抑えるためです。

```python
def predict_tta(model, tokenizer, smiles_list, n_tta=30):
    all_preds = []
    for _ in range(n_tta):
        aug = [random_smiles(s) for s in smiles_list]
        preds = model.predict(aug)
        all_preds.append(preds)
    return np.median(all_preds, axis=0)  # medianで集約
```

## Tg分布シフト後処理

1位解法のもう一つの重要なテクニックが、Tg（ガラス転移温度）の分布シフト補正です。

trainとtestのTg分布にシフトがあることが知られており、予測後に`std × 0.5644`を加算することでスコアが改善します。

```python
tg_std = submission['Tg'].std()
tg_shift = tg_std * 0.5644
submission['Tg'] += tg_shift
```

この係数`0.5644`はOOF上でグリッドサーチして求めた値です。

## Direct Match

テストデータのSMILESが学習データに完全一致する場合、モデルの予測ではなく学習データの値をそのまま使います。シンプルですが確実にスコアを改善します。

```python
for target in TARGETS:
    for _, row in test.iterrows():
        match = train.loc[(train['SMILES'] == row['SMILES']) & (train[target].notna()), target]
        if len(match) > 0:
            submission.loc[submission['id'] == row['id'], target] = match.values[0]
```

## Colab rdkitインストール地獄

追試で最も苦労したのは、実はモデルではなく**環境構築**でした。

Google Colab上でrdkitをインストールすると、pandas/numpyのバージョン不整合が発生します。

```bash
# これだけだとpandasが壊れる
!pip install rdkit

# 解決策: pandas/numpyを再インストール
!pip install --force-reinstall "pandas<2.4" "numpy<2.4"

# そしてランタイム再起動が必須
```

この問題は2026年3月時点でもまだ存在しています。rdkitのConda依存が原因です。

## 結果

### スコア

Late Submissionで**Private LB 0.08180**を達成しました。

| 順位 | チーム | スコア |
|------|--------|--------|
| 1位 | James Day | 0.07536 |
| 2位 | Ezra | 0.07722 |
| 3位 | Ghy HUST CS | 0.07820 |
| 7位 | CoderGirlM | 0.08144 |
| → | **この追試** | **0.08180** |
| 8位 | Dmitry Uarov | 0.08271 |

### アンサンブル重み

```
Tg:      w_bert=0.97  → BERTがほぼ全て
FFV:     w_bert=0.94
Tc:      w_bert=0.91
Density: w_bert=0.95
Rg:      w_bert=0.93
```

AutoGluonの寄与はほとんどありませんでした（w_bert > 0.91）。CodeBERTa単体でほぼ8位相当の性能が出ています。

## 1位との差分と改善余地

| 要素 | 1位 | この追試 |
|------|-----|----------|
| 言語モデル | CodeBERT (125M) | CodeBERTa-small (84M) |
| アンサンブル | BERT + AutoGluon + Uni-Mol 2 | BERT + AutoGluon |
| 疑似ラベル | あり | なし |
| Tg後処理 | あり | あり |
| スコア | 0.07536 | 0.08180 |

### 改善すれば届きそうなポイント

| 施策 | 推定改善 |
|------|---------|
| CodeBERT (125M) に変更 | -0.002〜0.005 |
| CatBoost追加 | -0.001〜0.002 |
| 疑似ラベル事前学習 | -0.004 |
| TTA 30→50 | -0.001 |
| 2位のTg華氏変換トリック | -0.01+ |

## 学び

### 1. コード用モデルがSMILESに効く

これが最大の発見です。化学特化のモデルを使わなくても、コード用に事前学習されたモデルでSMILES回帰ができる。SMILESの構造（括弧のネスト、記号の規則性）がプログラミング言語に類似しているためと考えられます。

### 2. Random SMILES拡張は強力

学習データ10倍拡張 + TTA30回のmedian集約は、モデル変更よりも効果が大きい可能性があります。SMILESの非一意性を逆手に取った、化学ML特有のテクニックです。

### 3. 最小モデルで十分な性能

84Mパラメータの最小モデルで、1位の90%の性能が出ます。計算リソースの制約がある場合でも、上位に食い込む手法が構築可能です。

### 4. 後処理の重要性

Tg分布シフト補正とDirect Matchは、モデルを一切変えずにスコアを改善するテクニックです。コンペでは最後の一押しが順位を分けるため、こうした後処理の工夫が重要です。

## おわりに

上位解法の「まとめ記事」は多いですが、実際にコードを書いて動かして検証する「追試」は少ないです。writeupだけでは伝わらない実装の詳細（環境構築の罠、ハイパーパラメータの感度、アンサンブル重みの実態）を、追試を通じて共有できればと思います。

コードは全てGitHubで公開しています:
https://github.com/nkwork9999/NeurIP2025_mytrial_following_1st_solution
