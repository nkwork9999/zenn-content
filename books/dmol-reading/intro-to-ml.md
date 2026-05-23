---
title: "Ch2: Introduction to Machine Learning — Delaney溶解度で線形回帰"
free: true
---

:::message
本章は [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/) で公開されている [dmol-book](https://dmol.pub/) (Andrew White, "Deep Learning for Molecules and Materials") を題材にした学習ノートです。本章の二次著作物部分も同じく CC BY-NC 3.0 で公開します。
:::


# アイデア

今回は Chapter 2 (Introduction to Machine Learning) の全範囲。


### 学ぶこと

- 機械学習の三要素 (特徴量 / ラベリング / モデル) を化学データに当てはめる
- AqSolDB (水溶解度データセット) の中身を一通り見る
- 線形回帰 を JAX で: 損失 (MSE) → 勾配降下 → 最終予測
- 標準化と小分け学習の併用
- k-means をやってみる
- エルボープロット


# そもそも

Ch1 はテンソル操作だけだったが、Ch2 で初めて「データを使ってモデルを学習させる」が出てくる。

ML教科書の典型 (iris / MNIST) を使わず最初から 化学例 = Delaney溶解度で始めているのがdmolの特色。

# 試す

dmol-bookの各ページには冒頭にロケットアイコンがあり、Google Colab で該当 ipynb をそのまま開いて実行できる仕組み。Ch2 は pandas + numpy + JAX + sklearn を使う。


## データ: AqSolDB (Delaney 溶解度データセット)

dmol Ch2 が使う `curated-solubility-dataset.csv` の正体は、Sorkun ら (2019) が公開した **AqSolDB** ([Scientific Data, doi:10.1038/s41597-019-0151-1](https://doi.org/10.1038/s41597-019-0151-1)) という水溶解度データセット。Delaney (2004) を含む9つの公開データを統合・整備したもので、**9982化合物** を収録。本家リポジトリは [mcsorkun/AqSolDB](https://github.com/mcsorkun/AqSolDB)。dmol-book は同じ CSV を自前リポジトリに置いている。

形は **9982行 × 26列**。中身は大きく 3 種類に分かれる。

### A. メタデータ・分子表現 (5列)

| 列名 | 型 | 内容 | サンプル値 |
|---|---|---|---|
| `ID` | str | データ点の一意ID (9982通り) | `A-3`, `A-4`, `A-5` |
| `Name` | str | 化合物名 (9893通り、重複あり) | `4-chlorobenzaldehyde` |
| `InChI` | str | InChI文字列 (構造の標準表記) | `InChI=1S/C7H5ClO/...` |
| `InChIKey` | str | InChIをハッシュ化した固定長キー | `AVPYQKSLYISFPO-UHFFFAOYSA-N` |
| `SMILES` | str | SMILES文字列 | `Clc1ccc(C=O)cc1` |

これらは **分子の同定情報** で、モデルの入力にはしない。「どの分子なのか」を後から追跡するためのキー。

### B. ラベル + データ品質情報 (4列)

| 列名 | 型 | 内容 | 範囲・備考 |
|---|---|---|---|
| `Solubility` | float | **★これが目的変数** = LogS (= log10 で表した水溶解度 mol/L)。例: -2.89 なら約 1.3 mmol/L | min=-13.17 (ほぼ溶けない), max=+2.14 (大量に溶ける), 平均-2.89 |
| `SD` | float | ソース間の食い違いの大きさ。0 = ソース全員が同じ値を言っている、値が大きいほど信頼性が低い | 0〜3.87 |
| `Ocurrences` | int | 元ソース群でこの分子の溶解度が合計何回測られているか | 1〜38 (大半は1) |
| `Group` | str | 「測定回数」と「測定値の一致度」で品質を分けたランク (G1〜G5、下記) | G1=7746, G3=1182, G5=636, G2=235, G4=183 |

なぜ `Solubility` が log10 なのか: 水への溶けやすさは分子によって **16桁** も違う (-13 〜 +2)。生の値を予測しようとすると桁数違いの数値が混在して扱いにくいので、log で揃えて回帰しやすい形にしている。化学・薬学では LogS / LogP のように log系の指標が定番。

**Group ラベルの定義** (Sorkun 2019 論文より):

| Group | ソース数 | SDによる一致度 | 件数 | 信頼度 |
|---|---|---|---|---|
| G1 | 1個 (単一ソース) | 検証不能 | 7746 | コメント不可 (1ソースしかないので比較できない) |
| G2 | 2個 | SD > 0.5 (食い違い大) | 235 | 低 |
| G3 | 2個 | SD ≤ 0.5 (一致) | 1182 | 中 |
| G4 | 3個以上 | SD > 0.5 (食い違い大) | 183 | 低 |
| G5 | 3個以上 | SD ≤ 0.5 (一致) | 636 | **最も高い** |

信頼度順は **G5 > G3 > G4 > G2 > (G1は別格で評価不能)**。全データの78%は G1 で「検証不可」 という状況なので、本気で品質を絞るなら G3+G5 (= 約1800件) だけで学習する戦略もある。dmol-book はこの絞り込みはせず、9982件全部を使う。

### C. RDKit 記述子 (17列、これが特徴量)

| 列名 | 型 | 内容 | 範囲 (min〜max) |
|---|---|---|---|
| `MolWt` | float | 分子量 (g/mol) | 9.01〜5299.46 |
| `MolLogP` | float | Wildman-Crippen 法による LogP 予測値 | -40.87〜+68.54 |
| `MolMR` | float | モル屈折率 | 0〜1419.35 |
| `HeavyAtomCount` | float | 水素を除いた重原子の数 | 1〜388 |
| `NumHAcceptors` | float | 水素結合アクセプター数 | 0〜86 |
| `NumHDonors` | float | 水素結合ドナー数 | 0〜26 |
| `NumHeteroatoms` | float | ヘテロ原子数 (C/H以外) | 0〜89 |
| `NumRotatableBonds` | float | 回転可能結合数 | 0〜141 |
| `NumValenceElectrons` | float | 価電子数 | 0〜2012 |
| `NumAromaticRings` | float | 芳香環数 | 0〜35 |
| `NumSaturatedRings` | float | 飽和環数 | 0〜30 |
| `NumAliphaticRings` | float | 脂肪族環数 | 0〜30 |
| `RingCount` | float | 全環数 | 0〜36 |
| `TPSA` | float | トポロジカル極性表面積 (Å²) | 0〜1214.34 |
| `LabuteASA` | float | Labute van der Waals 表面積 | 7.50〜2230.69 |
| `BalabanJ` | float | Balaban J トポロジ指数 | 0〜7.52 |
| `BertzCT` | float | Bertz 構造複雑性指数 | 0〜20720.27 |

つまり **A は同定情報**(モデル対象外)、**B はラベル + 品質メタ**、**C の 17 列が線形回帰の入力特徴量** という構成。

ここで注目したいのは **C の17列の範囲がバラバラ** な点。`BalabanJ` は 0〜7 だが `BertzCT` は 0〜20720。同じ単位ではないし、桁数も違う。後で §5 で「**Standardize しないと学習率を上げられない**」 という現象を見るが、その原因がここに既に埋まっている。


# Chapter 2: Introduction to Machine Learning

## 1. The Ingredients — 機械学習の3つの材料

dmol 本文の定義はとても素直:

- **Features** $\{\vec{x}_i\}$ : $N$ 個の $D$ 次元ベクトル
- **Labels** $\{y_i\}$ : $N$ 個の整数 or 実数
- **Model** $\hat{f}(\vec{x})$ : フィットさせる関数

これを AqSolDB の話に当てはめると:

| 役割 | dmol記法 | AqSolDB | 形 |
|---|---|---|---|
| Features | $\vec{x}_i$ | i 番目の分子の RDKit 記述子17個 | `(17,)` |
| Labels | $y_i$ | i 番目の分子の LogS | scalar |
| 全データ | $\{\vec{x}_i, y_i\}$ | 全9982分子 | features `(9982, 17)`, labels `(9982,)` |
| Model | $\hat{f}$ | 「17個の数値を入れたら LogS を返す関数」 | これから決める |

学習 = データを使って Model のパラメータを調整し、Loss (損失) を下げる、それだけ。

> dmolの脇注: 「教師あり学習はこの三要素の特定パターン」「ラベルが連続値なら回帰、離散値なら分類」と、機械学習の分類学そのものを features / labels の型で定義し直している。すっきりした見方。

## 2. データ準備

```python
features_start_at = list(soldata.columns).index("MolWt")
feature_names = soldata.columns[features_start_at:]
features = soldata.loc[:, feature_names].values   # (N, D)
labels = soldata.Solubility.values                # (N,)
print("features.shape =", features.shape)
print("labels.shape   =", labels.shape)
```

実機の出力:

```
features.shape = (9982, 17)
labels.shape   = (9982,)
```

ラベルの分布を 5 数要約 (`describe`) で確認:

```python
print(soldata.Solubility.describe())
```

実機の出力:

```
count    9982.000000
mean       -2.889909
std         2.368154
min       -13.171900
25%        -4.326325
50%        -2.618173
75%        -1.209735
max         2.137682
```

LogS は -13 〜 +2 に分布。平均が -2.9 (= 約 1 mmol/L)、std=2.4 で広い。**左に長く尾を引く** 形 (= 極端に溶けない分子が少数いる)。

### dmol が強調するEDA作法

> EDA を「特徴選択の根拠」にするなら、必ず **train/test split を先にやる**。test set を覗いてしまうと汚染される。

これは Ch3 (Regression & Model Assessment) で深く扱われる話だが、Ch2 では一旦 random分割 + 評価まで通して、後から「実はこの分割は甘い」と種明かしする構成になっている。

## 3. 線形モデルを書く (JAX)

最初のモデルは **線形回帰**。「17 個の特徴を、それぞれ重みをつけて足し合わせる」だけ。

$$\hat{y} = \vec{w} \cdot \vec{x} + b = \sum_{d=1}^{17} w_d \cdot x_d + b$$

| 記号 | 意味 |
|---|---|
| $\vec{w}$ | 17 個の重み (各特徴の貢献度) |
| $b$ | バイアス (切片、全体のオフセット) |
| $\hat{y}$ | 予測した LogS |

この $\vec{w}$ と $b$ を、データに合うように調整するのが学習。

### Loss (損失) = どれくらい間違えているかの数値

dmol が最初に使うのは **MSE (mean squared error)**:

$$\text{Loss} = \frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

予測 $\hat{y}_i$ が正解 $y_i$ から離れていればいるほど大きくなる、自乗で罰則を強める形。**0 に近いほど良い予測**。

```python
import jax
import jax.numpy as jnp

def linear_model(x, w, b):
    return jnp.dot(x, w) + b

def loss(y, labels):
    return jnp.mean((y - labels) ** 2)

def loss_wrapper(w, b, data):
    features, labels = data
    return loss(linear_model(features, w, b), labels)

# wとbの両方に対する勾配関数を1行で作る (JAX採用の決め手)
loss_grad = jax.grad(loss_wrapper, (0, 1))
```

### なぜ JAX なのか

`jax.grad(loss_wrapper, (0, 1))` の `(0, 1)` は **「第0引数 (w) と第1引数 (b) について微分する」** の意味。numpyだとMSE を $w$ や $b$ で偏微分する式を自分で書く必要があるが、JAX なら **コードから自動で勾配を計算してくれる (autodiff)**。

| 書き方 | 手間 | 拡張性 |
|---|---|---|
| numpy で手計算 | 偏微分を自分で導出 → コード化 | モデルを変えるたびに書き直し |
| JAX `jax.grad` | 関数を渡すだけ | モデルを変えても勾配計算は自動 |

線形回帰程度なら手計算でも書けるが、Ch6 (Standard Layers) 以降の NN 本体では autodiff 必須。**ここで JAX に慣れておくのが伏線**。

> dmolの脇注: JAX 採用の理由は autodiff だけではなく、`@jax.jit` (関数を XLA でコンパイルして GPU/TPU で走らせる) との相性も含めて、後章の deep learning に直接繋がるため。

## 4. 勾配降下 — 生 features、 batchingなし

学習の本体は **勾配降下法**。

$$\vec{w}_{\text{new}} = \vec{w}_{\text{old}} - \eta \cdot \frac{\partial L}{\partial \vec{w}}$$

意訳: 「**Loss が増える方向の逆に、少しだけ進む**」 を繰り返す。$\eta$ は **学習率** (1ステップでどれくらい進むか)。

```python
N, D = features.shape
np.random.seed(0)
w = np.random.normal(size=D)
b = 0.0
eta = 1e-6

for i in range(10):
    grad = loss_grad(w, b, (features, labels))
    w -= eta * grad[0]
    b -= eta * grad[1]

print(f"final loss = {float(loss_wrapper(w, b, (features, labels))):.4f}")
```

実機の出力:

```
final loss = 10832.5908
```

ラベルが -13 〜 +2 で std=2.37 なので、**完全に外しても loss = std² ≈ 5.6 程度** で済むはず。それが 10832 になっているのは**絶望的に大きい**。

### なぜこんなに大きいのか

理由は2つ:

1. **学習率 η が小さすぎる** (1e-6) のでほとんど進んでいない (10ステップしか回していない)
2. **features のスケールが揃っていない** ので、勾配の各成分の大きさが桁違いになる

特に 2 が効く。`BertzCT` は範囲 0〜20720、`BalabanJ` は 0〜7。同じ重み更新式で扱うと、`BertzCT` の方向にだけ巨大な勾配が乗って、他の特徴は無視されたまま進む。η を上げると `BertzCT` 方向で発散し、η を下げると他の方向が動かない、というジレンマ。

これを解決するのが次節。

## 5. Batching + Standardize — 両方揃って初めて効く

dmol の観察は鋭くて、**両方やる**:

1. **Batching**: 全データで 1 回更新するのではなく、32 件ずつのバッチで何回も更新 (= 更新回数を増やす)
2. **Standardize**: 各列を 平均 0・分散 1 にする (= スケールを揃える)

### Standardize とは何か

各列について:

$$x_{\text{new}} = \frac{x - \text{mean}(x)}{\text{std}(x)}$$

意訳: 「**各列を、平均を引いて標準偏差で割る**」。結果として全列が「平均 0 を中心に、std=1 で広がる」共通スケールに揃う。`BertzCT` の 0〜20720 も、`BalabanJ` の 0〜7 も、変換後はだいたい -3〜+3 に収まる。

```
変換前:                    変換後:
BertzCT     : 0  20720    BertzCT     : -1.2  +2.8
MolWt       : 9   5299    MolWt       : -0.8  +1.5
BalabanJ    : 0   7.5     BalabanJ    : -2.1  +1.9
NumRings    : 0   36      NumRings    : -1.0  +3.2
   ↑ 列ごとに桁違い         ↑ 全列が同じレンジに揃う
```

### なぜ batching と standardize はセットで効くか

ためしに **「Standardize なしで batching だけ入れる」** (η=1e-6, batch_size=32) を実行すると、loss は **1.05e14 まで発散** する。

理由: バッチが小さいと、勾配がバッチ内のサンプルの偏り (= 偶然 `BertzCT` が大きい分子が集まった等) に敏感になる。スケール差が揃ってないと、その偏りが増幅されて発散する。逆に Standardize さえしてあれば、スケールが揃うので η を上げても安定する。

```python
fmean = features.mean(axis=0)
fstd = features.std(axis=0)
std_features = (features - fmean) / fstd

np.random.seed(0)
w = np.random.normal(scale=0.1, size=D)
b = 0.0
eta = 1e-2          # 1e-6 から1万倍に上げられる
batch_size = 32

for i in range(N // batch_size):
    bx = std_features[i * batch_size : (i + 1) * batch_size]
    by = labels[i * batch_size : (i + 1) * batch_size]
    grad = loss_grad(w, b, (bx, by))
    w -= eta * grad[0]
    b -= eta * grad[1]

predicted = linear_model(std_features, w, b)
print(f"final loss = {float(loss(predicted, labels)):.4f}")
print(f"correlation coef = {float(np.corrcoef(labels, np.array(predicted))[0, 1]):.4f}")
```

実機の出力:

```
final loss = 3.6924
correlation coef = 0.6597
```

loss が 10832 → **3.7まで一気に下がった** (約 3000 倍改善)。相関係数 0.66 は「**悪くないが great ではない**」。dmol 本文も同じトーンで「ここから先は非線形にしないと頭打ち」と書いている。

### dmol筆者の脇注: 評価の甘さ

> ここで「ScaffoldSplit を使わないとこの相関 0.66 は過大評価」のような話が来てもいいが、それは Ch3 (Regression & Model Assessment) の範囲。dmol はあえて Ch2 では random 分割 + 評価まで通して、後から「実はこの分割は甘い」と種明かしする構成。

### ハマりどころ: Standardize の統計量は train で取る

実用上は **train データの mean / std を計算し、それで test も標準化する** のが鉄則。test の統計量を覗くと「test の情報が学習に漏れる (data leakage)」。Ch2 のコードは全データで standardize しているが、これは Ch3 で修正される。


## 6. 教師なし学習: k-means

ここから先は **教師なし** (= ラベルなしで「似たデータをまとめる」)。

化学的な動機: 9982 分子を何種類かのグループ (例: 「水によく溶ける」「ほぼ溶けない」「特殊な構造をもつ」など) に自動で分けたい。ラベルを使わずに、**特徴量 17 次元だけで** グループ分けする。

```python
import sklearn.cluster

kmeans = sklearn.cluster.KMeans(n_clusters=4, random_state=0, n_init=10)
kmeans.fit(std_features)

print("inertia (k=4) =", round(kmeans.inertia_, 2))
print("cluster sizes:", np.bincount(kmeans.labels_))
```

実機の出力:

```
inertia (k=4) = 94162.44
cluster sizes: [6012 3103  280  587]
```

### inertia とは

**inertia = 各点から、その点が属するクラスタ中心までの距離の二乗、を全点で合計した値**。

意訳: 「**クラスタ内で点がどれくらいまとまっているか**」を測る量。値が小さいほどクラスタ内が密、つまりまとまりが良い。クラスタ数 k を増やせば必ず減る (極端には N 個のクラスタにすれば inertia=0)。

### クラスタサイズの偏り

`[6012, 3103, 280, 587]` ← 半分以上が 1 クラスタ (6012/9982 = 60%) に入っている。「**自然な分け方ではなく、k=4 を強引に当てはめた結果**」 を示唆。280 や 587 の小さなクラスタは外れ値的な特殊分子の塊 (例: 極端に大きい高分子、フッ素やヨウ素を大量に持つ分子など) になりがち。

## 7. Elbow plot — クラスタ数の選び方

k を増やすと inertia は単調に下がるが、**「あるところを超えるとそれ以上 k を増やしても急に改善しない」 = エルボー** が見えるはず。そこが「自然なクラスタ数」と解釈する。

```python
for k in range(2, 10):
    km = sklearn.cluster.KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(std_features[::50])     # dmol本文と同じく、50個に1個でサブサンプル
    print(f"k={k}: inertia={round(km.inertia_, 2)}")
```

実機の出力:

```
k=2: inertia=1864.05
k=3: inertia=1485.02
k=4: inertia=1284.79
k=5: inertia=1134.18
k=6: inertia=992.63
k=7: inertia=896.49
k=8: inertia=837.78
k=9: inertia=769.6
```

「ガクッと折れる」明確なエルボーは **無く、緩やかに下がる**。dmol 本文も「3? 4? 6? 7?」 と曖昧で、結局 4 を選ぶ。

### 教訓: クラスタリングは正解がない

| 評価方法 | できること | できないこと |
|---|---|---|
| inertia | クラスタ内の密度 | 「k=4 が正解」かどうかは判定不能 |
| Elbow plot | 候補の k を絞る | 化学的な意味があるかは別途検証 |
| 専門知識 | 「水溶解度の典型グループは何種類?」と問い直す | 数値だけからは導けない |

**クラスタリングは複数の k で意味のある解釈が立つ方を選ぶ、というのが現実**。教師あり学習 (回帰・分類) と違って正解ラベルが無いので、検証は人間の解釈に頼ることになる。


# まとめ

Ch2 の旅程を 1 行で:

| 段階 | 数値 | 学び |
|---|---|---|
| データ確認 | 9982 × 17 + LogS | AqSolDB の中身 + Group ラベル |
| 線形モデル + 生 features | loss = 10832 | スケール差で進めない |
| Standardize + batching | **loss = 3.69, 相関 0.66** | 両方そろって初めて学習が回る |
| k-means (k=4) | inertia=94162 | クラスタが不均衡 (60% が 1 クラスタ) |
| Elbow plot | 明確なエルボー無し | クラスタリングは正解が無い |

次の Chapter 3 (Regression & Model Assessment) は、ここで出した「**相関 0.66**」が本当に正しい評価なのかを疑い直す章。過学習・正則化・分割評価・Scaffold split と進んでいく。化学固有の落とし穴 (ScaffoldSplit / 測定グループ別 CV) はそこで本格的に出てくる。


# 雑感

- dmol の「化学例から始める」は最初の数章だと地味だが、**後章の GNN・Equivariant・拡散モデルが化学例で一気に駆け上がるための布石**になっている
- Ch2 の「**Standardize しないと η が上げられない**」は numpy / sklearn を書く全員が踏むハマりで、化学に限らず実用上の知見
- JAX は線形回帰程度なら numpy とほぼ同じ書き方で済むが、**Ch6 (Standard Layers) 以降の NN 本体で autodiff が効いてくる**
- k-means のクラスタサイズが偏ったとき、「k が悪い」「特徴量が悪い」「データに自然なクラスタが存在しない」のどれかを判定するのは難しい。**教師なしは検証が辛い** という感覚を最初に味わうのが Ch2 の隠れた価値


# 参考リンク

- dmol-book 全文公開: https://dmol.pub/
- GitHub (ipynb): https://github.com/whitead/dmol-book
- 論文版 (LiveCoMS): https://doi.org/10.33011/livecoms.3.1.1499
- AqSolDB 論文: https://doi.org/10.1038/s41597-019-0151-1
- AqSolDB 本家リポジトリ: https://github.com/mcsorkun/AqSolDB
- 本記事のソース (公式 dmol-book): [ml/introduction.ipynb](https://github.com/whitead/dmol-book/blob/main/ml/introduction.ipynb)
- 前章: [Ch1 Tensors and Shapes](https://zenn.dev/nkwork9999/books/dmol-reading/viewer/tensors-and-shapes)
- 次章: [Ch3 Regression & Model Assessment](https://zenn.dev/nkwork9999/books/dmol-reading/viewer/regression)
