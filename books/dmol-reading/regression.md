---
title: "Ch3: Regression & Model Assessment — 過学習・正則化・Scaffold Split で線形回帰を本気で評価"
free: true
---

:::message
本章は [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/) で公開されている [dmol-book](https://dmol.pub/) (Andrew White, "Deep Learning for Molecules and Materials") を題材にした学習ノートです。本章の二次著作物部分も同じく CC BY-NC 3.0 で公開します。
:::


# アイデア


前回 [Ch2: Introduction to Machine Learning](https://zenn.dev/nkwork9999/books/dmol-reading/viewer/intro-to-ml) で AqSolDB 全9982化合物 + 17記述子を Standardize + JAX で線形回帰し、相関係数 0.66 までは取れた。<br>
ただし その評価はランダムに split しただけで、過学習や化学固有の汚染を見ていない。Ch3 はその評価をやり直す章。

### 学ぶこと

- train/test split で **過学習 (overfitting)** を初めて目で見る
- 過学習を 3つの要因 (ノイズ / 余剰特徴 / 訓練しすぎ) に分解する
- **Bias-Variance 分解** で「テスト誤差の下限」を式で理解する
- **L2 正則化 (Ridge)** で 7次多項式の test loss が 1744 → 6.9 に下がる
- **k-fold CV** が全データでは安定、N=25 だと完全に崩壊する
- **LOCOCV** (測定グループ別 leave-one-class-out) で 1.5x ペナルティが出る
- **Scaffold split** (Bemis-Murcko) で random split より test MSE が **2.3 倍** に跳ね上がる ← これが化学屋として一番見たいやつ


# そもそも

Ch3 のメッセージは1行で言うと: **「Ch2 の評価は甘い、正しく評価しろ」**。

dmol 本文は前章で出した「相関0.66」をいきなり信用せず、

1. そもそも overfitting しているか?
2. overfitting を bias と variance で読み解けるか?
3. 化学データで random split は信頼できるか?

の順で詰めていく。最後 (§6) で出てくる **Scaffold Split** が、Ch2 の予告どおり化学固有のハマりどころ。


# 試す

## §1. train25 + test25 で過学習を可視化

dmol が最初にやるのは「9982件あるのに、わざと 50 件だけ使う」。理由: 大量データだと overfit が起きにくく、学習曲線の test loss が綺麗に折れない。

```python
import numpy as np, pandas as pd, jax, jax.numpy as jnp

soldata = pd.read_csv(".../curated-solubility-dataset.csv")
feature_names = list(soldata.columns[list(soldata.columns).index("MolWt"):])

sample = soldata.sample(50, random_state=0)
train, test = sample.iloc[:25].copy(), sample.iloc[25:].copy()

# ★ train の統計で test を標準化する (data leakage を避ける)
mean, std = train[feature_names].mean(), train[feature_names].std()
train[feature_names] = (train[feature_names] - mean) / std
test[feature_names]  = (test[feature_names]  - mean) / std

x, y           = train[feature_names].values, train["Solubility"].values
test_x, test_y = test[feature_names].values,  test["Solubility"].values

@jax.jit
def loss(w, b, x, y): return jnp.mean((y - jnp.dot(x, w) - b) ** 2)
loss_grad = jax.grad(loss, (0, 1))

np.random.seed(0)
w, b, eta = np.random.normal(size=x.shape[1]), 0.0, 0.05
tr_hist, te_hist = [], []
for i in range(2000):
    g = loss_grad(w, b, x, y)
    w -= eta * g[0]; b -= eta * g[1]
    tr_hist.append(float(loss(w, b, x, y)))
    te_hist.append(float(loss(w, b, test_x, test_y)))
```

実機の出力:

```
final train loss = 0.872
final test  loss = 1.310
min   test  loss = 1.077  at step 782
```

学習が進むと **train loss は下がり続け、test loss はステップ782で底を打って上昇** する。これが overfitting の標準的な姿。「test loss が一度下がってまた上がり始めたら停止」がいわゆる **early stopping** の根拠。

🛠 **dmol原文の落とし穴**: 原文コードは `test[feature_names] -= train[...].mean()` の後で `train` の `mean()` を計算しているので、test が train の標準化前統計で歪む。上記は順番を入れ替えて修正している。pandas の `SettingWithCopyWarning` も `.copy()` で消える。

## §2. 合成多項式で「過学習の3要因」を切り分ける

dmol の鋭いところは、**過学習の発生メカニズムを実データではなく合成データで切り分ける**こと。

真の関数: $f(x) = x^3 - x^2 + x - 1$。x ∈ [-3, 3] を20点取り、中央10点を test、両端10点を train にする (= 外挿問題)。

4パターンの実機結果:

| 条件 | 特徴 | 訓練誤差 | テスト誤差 | 観察 |
|---|---|---|---|---|
| ノイズ無し・完璧特徴 | $[x^3, x^2, x, 1]$ | **0.000** | **0.000** | 完璧に再現 |
| **ノイズ有り**・完璧特徴 | $[x^3, x^2, x, 1]$ | 10.228 | 22.715 | ノイズ分テストが悪化、まだ overfit ではない |
| ノイズ有り・**余剰特徴** | $[x^6 \ldots x^0]$ (7個) | 1.745 | **1744.142** | 訓練が下がる代わりに **テストが1000倍悪化** ← 典型的 overfit |
| ノイズ有り・**間違った特徴** | $[x^2, x, e^{-x^2}, \cos x, 1]$ | 46.422 | 116879 | 補外領域で発散、これも overfit の一種 |

> **過学習が起きるための必要条件**: (a) ラベルにノイズがある、(b) 余分 or 不適切な特徴がある、(c) 訓練が収束している。3つ揃って初めて起きる。

合成データで切り分けたから断言できるのが嬉しい。実データでは交絡しまくって因果が見えない。

## §3. L2 正則化 (Ridge) で 1744 → 6.9 に下げる

Ch3 §2 の「ノイズ有り・余剰特徴 (7個)」を救う。L2 正則化 = 重みの大きさを罰則項にする:

$$L = \frac{1}{N}\sum_i (y_i - \hat{f}(\vec{x}_i))^2 + \lambda \sum_k w_k^2$$

閉形式解は $\vec{w} = (X^TX + \lambda I)^{-1} X^T y$。実機で λ を振った結果:

| λ | train_loss | test_loss | \|w\|₂ |
|---|---|---|---|
| 0.0 (= 普通の最小二乗) | 1.745 | **1744.142** | 73.9 |
| 0.01 | 3.052 | 288.4 | 32.5 |
| 0.1  | 5.281 | 35.5  | 10.1 |
| 1.0  | 7.954 | 17.9  | 3.9  |
| 10.0 | 11.281 | **6.876** | 0.9 |

λ=10 で **test loss が 1744 → 6.9 と約250倍改善**。代わりに train loss は 1.7 → 11 と悪化していて、これがまさに「bias を増やして variance を減らした」教科書的なトレードオフ。

> L1 (= Lasso) は同じ式の $\sum |w_k|$ 版で、**一部の重みが厳密に 0 になる** (= 特徴選択効果)。dmol は「予測精度が欲しいなら L2、特徴選択したいなら L1。ただし L1 の解は train データ次第でブレやすい」と Frank Harrell の言葉を引いている。

## §4. k-fold CV: 全データでは安定、N=25 では崩壊

Bias-Variance 分解の「分散」項は、**train/test の切り方そのもの**に対する分散も含む。だから 1回 split しただけの test loss は推定値として弱い。k-fold で複数取って平均する。

```python
def kfold_loss(X, y, k, seed=0):
    Xc = np.hstack([X, np.ones((len(X), 1))])  # ★ intercept列を追加
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), k)
    errs = []
    for i in range(k):
        te, tr = folds[i], np.concatenate([folds[j] for j in range(k) if j != i])
        w, *_ = np.linalg.lstsq(Xc[tr], y[tr], rcond=None)
        errs.append(float(np.mean((y[te] - Xc[te] @ w)**2)))
    return np.mean(errs), np.std(errs)
```

実機の結果:

| データ規模 | k | MSE | std |
|---|---|---|---|
| 全データ (N=9982) | 5  | 2.791 | 0.204 |
| 全データ | 10 | 2.789 | 0.281 |
| 全データ | 20 | 2.797 | 0.394 |
| **N=25** | 2  | 13.07 | 0.68 |
| N=25 | 5  | 51.28 | 49.62 |
| N=25 | 10 | 21.06 | 22.70 |
| N=25 | 25 (=LOOCV) | 24.34 | **61.67** |

全データでは k を変えても MSE はほぼ 2.79 で安定。**N=25 では MSE が k によって 13 〜 51 まで暴れ、std も平均並みに大きい**。dmol が「小データだからこそ k-fold (とその上の Jacknife+) が要る」と強調する理由が一発で出る。

🛠 **ハマりどころ**: `np.linalg.lstsq(X, y)` だけだと **intercept (バイアス項) が無い**。Solubility の平均は -2.89 なので、intercept 無しだと予測が常に 0 寄りに引っ張られて MSE が 11 付近に張り付く (今回も最初それで出した数値が dmol の値と合わなかった)。dmol が `linear_model = w@x + b` と b を持っているのと同じく、`[X | 1]` の列を増やすか sklearn の `LinearRegression(fit_intercept=True)` を使う。

## §5. LOCOCV: 測定グループを丸ごと test に追い出すと?

AqSolDB の `Group` 列 (Ch2 で説明した G1〜G5、ソース数 + SD で品質ランク付け) はそれぞれ別の測定元・別の品質特性を持つ。LOCOCV では **1グループまるごと test に追い出して、残りで学習** する。

実機の結果:

| left out | n_test | test MSE |
|---|---|---|
| G1 (単一ソース、最大派閥) | 7,746 | 3.230 |
| G2 (2ソース・食い違い大) | 235   | 6.372 |
| G3 (2ソース・一致) | 1,182 | 2.077 |
| G4 (3+ソース・食い違い大) | 183 | 6.327 |
| G5 (3+ソース・一致、最高品質) | 636 | 2.557 |
| **mean** | | **4.113** |

参考: 普通の k=5 CV の MSE = 2.791。

**LOCOCV のほうが 1.47 倍悪い**。これは「測定グループ別に偏りがあって、片方を test に追い出すと普通の random CV では見えない誤差が出る」ことを示す。特に **G2 と G4 (= SD が大きい = もともと値が食い違っている)** だけ MSE が 2倍以上に跳ねている。data quality の偏りがそのまま test 誤差に出る、という当たり前だが見落とされがちな事実。

## §6. Scaffold Split: 化学屋として一番見たい数字

ここが Ch3 のクライマックス。**random split は化学的に意味のある test 集合を作らない**。なぜなら近い分子 (= 同じ骨格、置換基だけ違う) が train と test に混在しがちで、モデルは「骨格を覚えればよい」状態になりがち。

代わりに **Bemis-Murcko scaffold** で分子を骨格ごとに分類し、希少な骨格をまとめて test に入れる。これが **MoleculeNet 系のベンチマークでよく使われる scaffold split**。

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

scaffolds = defaultdict(list)
for i, smi in enumerate(soldata["SMILES"]):
    mol = Chem.MolFromSmiles(smi)
    sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    scaffolds[sc].append(i)

# 希少骨格から順に test に追加、20% に達したら止める
items = sorted(scaffolds.items(), key=lambda t: len(t[1]))
test_idx, target = [], int(len(soldata) * 0.2)
for sc, ix in items:
    if len(test_idx) >= target: break
    test_idx.extend(ix)
train_idx = sorted(set(range(len(soldata))) - set(test_idx))
```

実機の結果:

```
unique scaffolds = 1948
scaffold split → train 7986 / test 1996
scaffold split test MSE = 6.311
random   split test MSE = 2.736   (同じサイズで比較)
scaffold/random ratio   = 2.31x
```

**Scaffold split は random split の 2.3倍悪い**。Ch2 の相関0.66 はランダム分割だから出た数字で、scaffold split に切り替えると評価は一気に厳しくなる。

骨格分布の中身も興味深い:

```
最頻骨格 (train行き):
  ''                  n=2940  ← 環無し化合物 (脂肪族鎖)
  'c1ccccc1'          n=1753  ← 単純なベンゼン環
  'c1ccc(-c2ccccc2)cc1'  n=219   ← ビフェニル

希少骨格 (test行き):
  'O=C1Nc2cccc3cccc1c23'              n=1   ← フェナントリジノン系
  'c1cc(N(CC2CO2)CC2CO2)ccc1Cc1...'   n=1   ← ジエポキシ
  ...
```

scaffold split は **「ベンゼン由来の常識を覚えたモデルが、見たことない骨格で何ができるか」** を試している。化学のリアルな課題 (新規骨格でのスクリーニング) を模した評価。


# まとめ

Ch3 のメッセージを実機の数値で再確認:

| 段階 | 操作 | test MSE | 学び |
|---|---|---|---|
| Ch2 | random 9982件、評価ナシ | (相関 0.66) | ベースライン |
| Ch3 §1 | 25/25 split で学習曲線見る | 1.31 | 過学習が見える |
| Ch3 §2 | 合成データで切り分け | 1744 | ノイズ + 余剰特徴で爆発 |
| Ch3 §3 | L2 で抑制 | 6.9 | 250倍改善、bias↑variance↓ |
| Ch3 §4 | k-fold で安定推定 | 2.79 | std=0.2 で信頼できる |
| Ch3 §5 | LOCOCV (品質グループ別) | 4.11 | random の 1.5x |
| Ch3 §6 | **Scaffold split** | 6.31 | **random の 2.3x ← 本命** |

次の Chapter 4 (Classification) は同じデータを連続値ではなく可溶/不溶のラベルにして、softmax / cross-entropy / ROC-AUC に入る。CN系 (Ch5 Kernel Learning, Ch6 Deep Learning Intro) への橋渡し。


# 雑感

- dmol の構成、Ch2→Ch3 の流れが綺麗。Ch2 で「random split + 相関0.66」を出した瞬間に「これ甘いよね?」と思った読者をちゃんと拾いに来る
- **Scaffold split の 2.3x は知らないと事故る数字**。論文で random split のスコアだけ報告するのが NG な理由がここで腹落ちする
- L2 = closed form で一発、というのが地味に良い。Ch4 以降 NN に入ると closed form 解は無くなって gradient descent + early stopping だけが残るので、線形でやれるうちにやっておく価値
- LOCOCV を「自分のデータの自然なグルーピング」で回すのは応用が広い。NIR コンペで言えば「樹種別」「タイムスタンプ別」がそのまま LOCOCV のクラスになる ([Ch2 と同じく 07_sensor/nir 連動ネタ](../../07_sensor/nir/))
- dmol コードの落とし穴2件 (test標準化の順序、intercept 無しの lstsq) は、教科書ノートをそのまま動かすと数字が合わなくて困る典型。実機で動かして数字を取らないと気付かない


# 参考リンク

- dmol-book 全文公開: https://dmol.pub/
- Chapter 3 (Regression & Model Assessment): https://dmol.pub/ml/regression.html
- GitHub (ipynb): https://github.com/whitead/dmol-book/blob/main/ml/regression.ipynb
- 論文版 (LiveCoMS): https://doi.org/10.33011/livecoms.3.1.1499
- AqSolDB論文: https://doi.org/10.1038/s41597-019-0151-1
- Bemis-Murcko 原論文 (1996): https://doi.org/10.1021/jm9602928
- MoleculeNet (scaffold split を benchmark 化): https://doi.org/10.1039/C7SC02664A
- Jacknife+ (Barber 2019): https://arxiv.org/abs/1905.02928
- 本記事のソース (公式 dmol-book): [ml/regression.ipynb](https://github.com/whitead/dmol-book/blob/main/ml/regression.ipynb)
- 前章: [Ch2 Introduction to Machine Learning](https://zenn.dev/nkwork9999/books/dmol-reading/viewer/intro-to-ml)
