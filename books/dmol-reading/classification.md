---
title: "Ch4: Classification — FDA臨床試験データで分類器を作る"
free: true
---

:::message
本章は [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/) で公開されている [dmol-book](https://dmol.pub/) (Andrew White, "Deep Learning for Molecules and Materials") を題材にした学習ノートです。本章の二次著作物部分も同じく CC BY-NC 3.0 で公開します。
:::

# アイデア

前回 [Ch3: Regression & Model Assessment](https://zenn.dev/nkwork9999/books/dmol-reading/viewer/regression) では、AqSolDB の溶解度を題材に「回帰モデルをどう評価するか」を見た。<br>
今回は Chapter 4 (Classification)。連続値を当てる回帰ではなく、**クラスを当てる分類** を扱う。

題材は MoleculeNet に入っている Clintox データセット。薬剤候補の分子が並んでいて、それぞれに「FDA 承認薬かどうか」のラベルが付いている。

入力として使うのは `SMILES` という分子構造を文字列で表したもの。そこから RDKit/Mordred で分子の性質を表す数値特徴量を作り、最終的に `FDA_APPROVED`、つまり「承認薬なら 1、そうでなければ 0」を予測する。

### 学ぶこと

- 二値分類・多クラス分類・マルチラベル分類の違い
- SMILES 文字列を RDKit の分子オブジェクトに変換し、Mordred で分子記述子を作る流れ
- パーセプトロンのように 0/1 を直接返す分類器が、JAX の勾配降下と相性が悪い理由
- sigmoid と交差エントロピーで作る二値分類器
- クラス不均衡があると accuracy（正解率）が簡単に高く見えてしまうこと
- 偽陽性・偽陰性と、しきい値調整の考え方
- ROC 曲線と ROC AUC の読み方
- クラス不均衡に対する重み付けと、その限界


# そもそも

分類は「ラベルがカテゴリ」の教師あり学習。

dmol は最初に分類タスクを3つに分けている。

| 種類 | ラベル例 | 意味 |
|---|---|---|
| binary classification | `FDA承認 = 1 / 非承認 = 0` | 1つのクラスに属するかどうか |
| multi-class classification | `赤 / 緑 / 橙` | 複数クラスのうち、必ず1つだけに属する |
| multi-label classification | `THFに溶ける / 水に溶けない / クロロホルムに溶ける` | 複数ラベルを同時に持てる |

化学・材料では binary classification がかなり多い。例えば:

- 活性あり / なし
- 毒性あり / なし
- 合成できる / できない
- 膜を通る / 通らない
- 結晶化する / しない

今回の Clintox も binary classification。

```text
y = 1 : FDA承認薬
y = 0 : 非承認 / 臨床試験失敗側
```

分類は 0/1 のラベルを当てる問題だが、モデルの中ではいきなり 0 か 1 を出す必要はない。<br>
まずは「どれくらい承認薬っぽいか」を表す連続値の `score` を出して、その `score` をしきい値で区切れば分類になる。

```text
score = w・x + b

score > 0 なら class 1
score <= 0 なら class 0
```

この例では `score = 0` が境目になる。この境目を **decision boundary** と呼ぶ。

`score` が 0 より大きければ class 1、つまり FDA 承認薬側。0 より小さければ class 0 側。さらに、`score` が大きいほど「class 1 らしさ」が強い、と見ることができる。


# 試す

## §1. データ: Clintox

dmol Ch4 で使うのは MoleculeNet の Clintox データセット。

CSV の形は手元の `dmol-book/data/clintox.csv.gz` だと:

```text
1484 rows x 3 columns
columns = smiles, FDA_APPROVED, CT_TOX
```

この章で使う目的変数は `FDA_APPROVED`。

```python
toxdata = pd.read_csv(
    "https://github.com/whitead/dmol-book/raw/main/data/clintox.csv.gz"
)
toxdata.head()
```

ラベル分布を見るとかなり偏っている。

```text
FDA_APPROVED = 1 : 1390件
FDA_APPROVED = 0 :   94件
```

割合にすると:

```text
positive = 93.7%
negative =  6.3%
```

ここがこの章の本質。<br>
**何も考えずに全部 1 と予測するだけで、accuracy は 93.7% になる。**

この時点で accuracy だけを見てはいけない匂いがする。

## §2. SMILES から Mordred 記述子を作る

Ch2/Ch3 の AqSolDB は、最初から RDKit 記述子17列が CSV に入っていた。<br>
今回は `smiles` しかないので、特徴量を自分で作る。

流れはこう。

```python
calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)
molecules = [rdkit.Chem.MolFromSmiles(smi) for smi in toxdata.smiles]
```

`MolFromSmiles` は SMILES 文字列を RDKit の分子オブジェクトに変換する。<br>
ただし、全 SMILES が必ず変換できるとは限らない。失敗すると `None` が返る。

```python
valid_mol_idx = [bool(m) for m in molecules]
valid_mols = [m for m in molecules if m]
```

ここで重要なのは、**分子を消したらラベルも同じ行だけ消す** こと。

```python
features = calc.pandas(valid_mols).astype(float)
labels = toxdata[valid_mol_idx].FDA_APPROVED
```

`valid_mol_idx` を保存しておかないと、特徴量とラベルの行対応がズレる。<br>
これは地味だけどかなり危ない事故ポイント。

その後、Ch2 と同じように標準化する。

```python
features -= features.mean()
features /= features.std()

# std=0 の特徴は NaN になるので落とす
features.dropna(inplace=True, axis=1)
```

Mordred は大量の記述子を作る。便利だが、全部が有効とは限らない。定数列、欠損列、極端な外れ値は普通に出るので、標準化後に NaN 列を落とす処理が入っている。


# Chapter 4: Classification

## §3. 0/1 を直接返す分類器: パーセプトロン

最初の分類器はパーセプトロン。

```python
def perceptron(x, w, b):
    v = jnp.dot(x, w) + b
    y = jnp.where(v > 0, jnp.ones_like(v), jnp.zeros_like(v))
    return y
```

やっていることはかなり素直。

入力の特徴量を重み付きで足し合わせ、最後に `b` で少し位置をずらす。そうして作った値が 0 より大きければ `1`、0 以下なら `0` を返す。

つまりパーセプトロンは「この分子は承認薬である確率が 0.73 です」のような中間的な値を返さない。最初から 0/1 で返す。

このように、確率ではなくラベルを直接返す分類器を hard classifier と呼ぶ。

分類ラベルが 0/1 なら、予測が当たっているか外れているかは簡単に数えられる。<br>
正解が `1` なのに `0` と予測したら間違い。正解が `0` なのに `1` と予測しても間違い。つまり「どれだけ 0/1 を間違えたか」を損失として見ればよさそうに思える。

ところが、このやり方を JAX の勾配降下で学習しようとすると問題が出る。

勾配が全部 0 になってしまう。

理由は、パーセプトロンの出力が「なめらかに変わる値」ではなく、0 を境に急に切り替わる値だから。

例えば score が 0.2 でも 0.3 でも 10 でも、出力は全部 `1`。score が少し変わっても、0 の境目をまたがない限り、予測結果はまったく変わらない。

勾配降下は「少し動かしたら損失がどちらに変わるか」を手がかりに学習する。ところが、少し動かしても出力が変わらないなら、改善方向を見つけられない。だから勾配が 0 になってしまう。

パーセプトロンには専用の学習手順があるが、dmol はそこへ進まず、もっと現代的な **smooth な分類器** に切り替える。

この章で一番大事な直感はここだと思う。

```text
hard に 0/1 を出すモデルは、人間にはわかりやすい。
でも勾配降下で学習したいなら、途中は soft な確率で持つ方がよい。
```

## §4. sigmoid: 0/1 ではなく「確率」を出す

パーセプトロンの問題は、最初から `0` か `1` に切ってしまうことだった。そこで dmol は、hard threshold の代わりに sigmoid を使う。

sigmoid は、どんな実数でも 0〜1 の範囲に押し込む関数。score が 0 なら 0.5、score が大きくなるほど 1 に近づき、小さくなるほど 0 に近づく。

```text
score = 0      -> 確率 0.5
score > 0      -> 確率 0.5 より大きい
score < 0      -> 確率 0.5 より小さい
score が大きい -> 1 に近づく
score が小さい -> 0 に近づく
```

これでモデルは、いきなり「承認薬 / 非承認」と決め打ちしない。まず「承認薬っぽさ」を確率として出す。

この形が logistic regression。名前には regression と入っているが、やっていることは二値分類。内部では `score` を線形モデルで作り、それを sigmoid で確率に変換している。

Ch2 の線形回帰と比べると、変わったのは最後だけ。

```text
線形回帰: 数値をそのまま出す
分類: 数値を sigmoid に通して確率にする
```

この「途中はなめらかな確率で持つ」が大事。出力が連続的に変わるので、JAX の勾配降下で普通に学習できる。

## §5. 分類の損失: cross entropy

回帰では MSE を使った。予測値と正解値のズレを二乗して、「どれくらい外したか」を測る。

分類では cross entropy を使う。こちらは「正解クラスにどれくらい高い確率を置けたか」を見る損失。

直感はかなり素直。

- 正解が `1` のとき、予測確率 0.99 ならほぼ正解
- 正解が `1` のとき、予測確率 0.50 なら迷っている
- 正解が `1` のとき、予測確率 0.01 なら自信満々に外している

最後のケースは重く罰せられる。分類では「間違えた」だけでなく、「どれくらい自信を持って間違えたか」も重要になる。

正解が `0` の場合は逆で、0 に近い確率を出せばよく、1 に近い確率を出すほど大きく罰せられる。

学習ループ自体は Ch2 の線形回帰とほぼ同じ。違うのは、モデル出力を sigmoid で確率にし、損失を MSE ではなく cross entropy にするところだけ。

```text
Ch2: 線形モデル + MSE
Ch4: sigmoid分類器 + cross entropy
```

この置き換えで、回帰で使った「勾配降下でパラメータを少しずつ動かす」という考え方が、そのまま分類にも使える。


# 分類指標

## §6. accuracy は危ない

まず出てくる指標は accuracy。予測クラスと正解クラスが一致した割合なので、意味はわかりやすい。

ただし Clintox では、ここが罠になる。

このデータでは `FDA_APPROVED=1` が 93.7%。つまり、何も考えずに全部 `1` と予測するだけで、accuracy は 93.7% になる。

```text
全部「承認薬」と言う
=> positive が多いので accuracy は高い
=> でも negative を見つける力はゼロ
```

これはかなり危ない。accuracy だけを見ると、モデルが賢いのか、ただ多数派に乗っているだけなのかが分からない。

Ch3 が「random split のスコアを疑え」だったなら、Ch4 はここで「accuracy を疑え」に切り替わる。

## §7. 偽陽性・偽陰性と threshold

分類の間違いは、ただの「外れ」ではなく2種類に分けて見る。

| 種類 | 意味 | 今回の文脈 |
|---|---|---|
| 偽陽性 (false positive) | `1` と予測したが、本当は `0` | 通ると思って臨床試験したら失敗 |
| 偽陰性 (false negative) | `0` と予測したが、本当は `1` | 本当は通る薬を捨ててしまう |

今回の文脈では、偽陽性のコストがかなり重い。臨床試験は高額なので、「通る」と予測して進めたのに失敗するのは痛い。

一方、偽陰性は「本当は良い薬を見逃す」こと。これも痛いが、コストの性質が違う。

ここで threshold が出てくる。モデルは確率を出すだけなので、最後にどこから `1` と呼ぶかを決める必要がある。

```text
threshold = 0.50
=> 50%以上なら positive

threshold = 0.90
=> 90%以上の自信があるときだけ positive

threshold = 0.99
=> 99%以上の自信があるときだけ positive
```

threshold を上げると、positive と言う回数が減る。なので偽陽性は減りやすい。<br>
その代わり、本当は positive だったものも捨てやすくなるので、偽陰性は増える。

ここが分類の実務ポイント。

```text
モデル学習 = 確率を出せるようにする
運用設計 = どの threshold で意思決定するか決める
```

同じモデルでも、threshold を変えるだけで挙動はかなり変わる。

## §8. ROC curve と ROC AUC

threshold を1つに固定すると、その一点での性能しか見えない。そこで threshold を少しずつ動かしながら、偽陽性と真陽性のバランスを見る。

これが ROC curve。

| 軸 | 意味 |
|---|---|
| x軸 | 偽陽性率。negative をどれだけ positive と誤判定したか |
| y軸 | 真陽性率。本当の positive をどれだけ拾えたか |

理想は左上。左に行くほど偽陽性が少なく、上に行くほど positive をよく拾えている。

ランダム分類器はだいたい対角線になる。良い分類器は対角線より上にふくらむ。

ROC curve の面積が ROC AUC。threshold を固定しない分類能力の要約、と考えるとよい。

accuracy より ROC AUC が好まれる場面が多いのは、threshold を1点に決め打ちせず、偽陽性・偽陰性のトレードオフを広く見られるから。

## §9. precision / recall / confusion matrix

dmol は軽く触れるだけだが、分類では他にもよく使う指標がある。

precision は「positive と予測したもののうち、本当に positive だった割合」。薬剤スクリーニングなら、提案した候補の当たり率に近い。

recall は「本当の positive のうち、どれだけ拾えたか」。見逃しをどれだけ減らせたかを見る指標。

confusion matrix は、予測クラスと真のクラスの対応表。accuracy は1つの数字に潰すので、どう間違えたかが消える。confusion matrix は、間違い方を残してくれる。

例えば溶解度を `insoluble / weakly soluble / soluble` の3クラスに分けたとき、「soluble を weakly soluble と間違えやすい」のような失敗パターンが見える。


# Class Imbalance

## §10. クラス不均衡は、いつ問題なのか

Clintox は positive が 93.7%。かなり偏っている。

ただし dmol は、ここで雑に「不均衡だから必ず補正」とは言わない。まず考えるべきは、test 時の分布も train と同じなのか、という点。

もし実運用でも positive が 94% くらいなら、その不均衡は現実を反映しているだけかもしれない。その場合、無理に 50:50 に直すと、逆に現実からズレる。

一方で、train のラベル分布と実運用のラベル分布が違うなら label shift。この場合は対策が必要になる。

医療・薬剤探索では、ラベル不均衡は普通に起きる。毒性ありは少ない。活性ありも少ない。失敗例が記録されにくいこともある。screening では positive だけが集まりがち、という偏りも出る。

つまり、クラス不均衡は「数字が偏っているから悪」ではない。まず、その偏りが現実の分布なのか、データ収集の都合で生まれたズレなのかを見る必要がある。

## §11. 対策1: over-sampling / under-sampling

1つ目の対策は、データ側をいじる方法。

minority class を増やすのが over-sampling。majority class を減らすのが under-sampling。

over-sampling は、少ないクラスのデータをコピーしたり、SMOTE のように合成サンプルを作ったりする。モデルや損失関数に依存しないので、どんな分類器にも使いやすい。

ただし、同じデータをコピーしすぎると過学習しやすい。

under-sampling は、多数派データを捨ててバランスを取る。シンプルだが、せっかく持っているデータを捨てることになる。

どちらも「ラベル分布をきれいにする」方法だが、きれいにした分布が本当に運用時の分布に近いのかは別問題。

## §12. 対策2: weighted cross entropy

もう1つは、loss 側で重みを付ける方法。

rare class を間違えたときの罰を重くする。dmol の例では、少数派である negative (`y=0`) を強く重み付けしている。

直感的には「貴重な negative を間違えたら、より大きく怒る」loss。

ただし dmol の実験では、weighted training は劇的には良くならない。ROC を比べると、一部の false positive rate 帯では改善するが、低 false positive rate 側ではむしろ悪くなる。

ここも実務的な話になる。

class imbalance への対処は、training weight だけではない。学習後に threshold を調整するだけでも、挙動はかなり制御できる。

モデルを作る前に「重みをどうするか」だけ考えるより、モデルが出す確率を見て、運用上どの threshold を使うかまで含めて設計する方がよい。

## §13. negative がゼロの screening

dmol は最後に positive-unlabeled learning にも触れる。

薬剤・ペプチド探索では、screening によって「当たったもの」だけが集まり、明確な negative が無いことがある。

```text
positive examples はある
negative examples はない
unlabeled examples は大量にある
```

この状況は普通の binary classification ではない。positive-unlabeled learning という別の問題設定になる。

この指摘はかなり化学MLっぽい。データセットに `0` が入っているからといって、それが本当に negative なのか、単に未測定なのかを確認しないといけない。


# Overfitting

Ch4 では分類器の導入を優先しているので、Ch3 のような本格的な CV や scaffold split はやらない。

ただし dmol は最後に、分類でも当然 Ch3 の話が必要だと釘を刺す。

今回の特徴量は Mordred で大量に作る。サンプル数は 1484。特徴量が多く、サンプル数がそこまで多くないので、普通に過学習しうる。

本気で評価するなら、train/test split、k-fold CV、scaffold split、L1/L2 正則化、ROC AUC、threshold tuning、class imbalance check まで必要になる。

特に化学データなら、Ch3 と同じく random split だけでは甘い。FDA approved っぽい分子骨格を train と test にまたいで覚えているだけ、という可能性があるので、scaffold split で見たい。


# 雑感

Ch4 は「分類器を作ろう」より、「分類評価で事故らないようにしよう」の章だった。

刺さった点は3つ。

1つ目。0/1 を直接返す分類器は、人間にはわかりやすいが、勾配降下には向かない。だから sigmoid のように、途中は soft な確率で持つ。この流れは、そのまま deep learning の設計思想につながる。

2つ目。accuracy はクラス不均衡で簡単に壊れる。Clintox は positive が 93.7% なので、全部 `1` と言うだけで高 accuracy。分類ではまずラベル分布を見る。次に、偽陽性と偽陰性を見る。

3つ目。threshold はモデルそのものというより、運用上の意思決定。薬剤候補をどれだけ攻めるか、どれだけ保守的にするかで threshold は変わる。モデルは確率を出すだけ。最後にどこで切るかは、コスト・リスク・目的で決まる。

回帰の Ch3 が「評価分割を疑え」なら、分類の Ch4 は「accuracy を疑え」。


# 参考リンク

- dmol-book 全文公開: https://dmol.pub/
- Chapter 4 (Classification): https://dmol.pub/ml/classification.html
- GitHub (ipynb): https://github.com/whitead/dmol-book/blob/main/ml/classification.ipynb
- MoleculeNet: https://doi.org/10.1039/C7SC02664A
- Mordred descriptor paper: https://doi.org/10.1186/s13321-018-0258-y
- SMOTE: https://doi.org/10.1613/jair.953
- Positive-unlabeled learning example: https://doi.org/10.1038/s41598-021-01965-5
- 前章: [Ch3 Regression & Model Assessment](https://zenn.dev/nkwork9999/books/dmol-reading/viewer/regression)
