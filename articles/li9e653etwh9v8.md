---
title: "画像の間違い探しをDuckDBでできる拡張機能を作った"
emoji: "🖼️"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["duckdb", "onnx", "embedding", "画像処理", "machinelearning"]
published: true
published_at: 2026-05-16 09:00
---


# アイデア

画像系のKaggleコンペを追っていて、「画像をベクトルにしてしまえば、簡易的な似ている / 違うの判定はDuckDBの中だけで全部できるのでは？」と思った。

ちなみにコンペとは以下のやつ。
 [Recod.ai/LUC Scientific Image Forgery Detection 1位解法 (vlad3996/forgeryscope)](https://github.com/vlad3996/forgeryscope) 

### できること
- 画像をDuckDBの中でそのままベクトル化できる。
- 異なる画像同士でどれだけ似ているかを数値で出力できる。(多少の位置ずれには対応)
- 画像の**どこが**違うかをパッチ単位で取って、赤く塗ったヒートマップ画像まで書き出せる。(精度は粗目)

例えばこんな感じ。「ほぼ同じだが一部だけ違う」2枚の画像（[vzat/comparing_images](https://github.com/vzat/comparing_images) の `test1.1.jpg` / `test1.2.jpg`、典型的な間違い探しペア）を渡すと:

| test1.1.jpg | test1.2.jpg |
|---|---|
| ![test1.1](https://raw.githubusercontent.com/vzat/comparing_images/master/images/test1.1.jpg) | ![test1.2](https://raw.githubusercontent.com/vzat/comparing_images/master/images/test1.2.jpg) |

`pic_diff_heatmap`一発で、違うパッチだけ赤く塗ったヒートマップ画像がDuckDBから出てくる:

| test1.1にヒートマップを重ねた結果 | test1.2にヒートマップを重ねた結果 |
|---|---|
| ![test1a_heatmap](/images/li9e653etwh9v8/heatmap_test1a.png) | ![test1b_heatmap](/images/li9e653etwh9v8/heatmap_test1b.png) |

実装の中身と他の画像での試行は本文後半で。


# そもそも

画像をベクトル（embedding）に変換できると何が嬉しいか？いまいちピンとこない人もいるかもしれないが要は「画像同士の似ている度合い」を数値として扱えるようになります。(専門の人に怒られそうな説明ですが...)
つまり画像のみで画像検索や類似画像探索処理ができるようになります。一応簡易的な異常検知もできるはず。

それらを可能にするのが画像からベクトルを作るためのモデル（ビジョンエンコーダ）です。
近年は以下のようなものがあるようです。

- **CLIP (2021, OpenAI)**: 画像と説明文のセットで学習させたモデル。
- **DINOv3 (2025, Meta)**: テキストを使わず画像だけで学習する系統の最新版。画像同士の細かい類似性を取るのが得意。
- **EUPE (2026, Meta)**: 軽量・実用に振ったモデル（Efficient Universal Perception Encoder）<br>ViT-T/16なら6Mパラメータ・192次元と非常に小さく、拡張バイナリに同梱できるサイズ。今回pic2vecが標準で持っているのもこれ。

EUPEという超軽量なサイズのモデルが出たのでこれを埋め込んでおけば、拡張機能でも使えるか？と思い作ってみたというのが経緯です。
<br>あとduckdbで結果を保存しておけば「間違っている画像のリスト」みたいな簡易画像分類表のようなものやなんとなくカテゴリ分けもできそうだなと思い作ってみました。


# 作ったもの

## pic2vec
https://github.com/nkwork9999/pic2vec
https://duckdb.org/community_extensions/extensions/pic2vec

記事で行うのは以下です。

1. **画像をベクトルにして画像全体の類似度を比較する。**
   → 依存ライブラリ不要で画像をベクトル化する。
2. **「どこが違うか」を取り出す**
   → 画像をN×Nのマス目に切って、マスごとの類似度を返す。またヒートマップで違うマスほど赤く塗ったPNG画像を書き出す（間違い探し）。
3. **「2枚の画像がどれくらい似ているか」をSQL関数で数値化できる**
   → ベクトルを2つ渡すと類似度スコアが返ってくる関数が複数用意されている。画像枚数が増えて全件比較が重くなってきたら、DuckDB公式の近傍検索拡張 `vss` に繋いで高速化もできる。

4. **結果をDuckDBに保存して使い回せる**
   → そのままテーブル列として持っておけるので、CSV/Parquet/DuckDBファイルに永続化しておけば、後から類似度クエリで「間違ってる画像リスト」を引き直したり、ざっくりカテゴリ分けしたり、増分でデータを足したりがSQLだけで回せる。



# 使ってみる

OpenCVベースの画像差分検出サンプル [vzat/comparing_images](https://github.com/vzat/comparing_images) の中に、「ほぼ同じだけど微妙に違うプリント基板(PCB)」のペアと、それを改ざんした`fakePCB`シリーズが置いてあるので、これをそのままGitHubのraw URLから読み込む。ローカルにダウンロードする必要はない。

使う画像はこちら（GitHubから直接埋め込み表示）:

| pcb1.jpg（基準PCB） | pcb2.jpg（似ているPCB ＝ 間違い探しの相手） |
|---|---|
| ![pcb1](https://raw.githubusercontent.com/vzat/comparing_images/master/images/pcb1.jpg) | ![pcb2](https://raw.githubusercontent.com/vzat/comparing_images/master/images/pcb2.jpg) |

| fakePCB3.jpg（改ざん版3） | fakePCB5.jpg（改ざん版5） |
|---|---|
| ![fakePCB3](https://raw.githubusercontent.com/vzat/comparing_images/master/images/fakePCB3.jpg) | ![fakePCB5](https://raw.githubusercontent.com/vzat/comparing_images/master/images/fakePCB5.jpg) |

| test1.1.jpg（全然違う画像、対照群） | test1.2.jpg（test1.1のペア） |
|---|---|
| ![test1.1](https://raw.githubusercontent.com/vzat/comparing_images/master/images/test1.1.jpg) | ![test1.2](https://raw.githubusercontent.com/vzat/comparing_images/master/images/test1.2.jpg) |

```sh
duckdb
```

```sql
INSTALL pic2vec FROM community;
LOAD pic2vec;
INSTALL httpfs; LOAD httpfs;   -- 公開画像URLを直接掴むため
```

## 1. 画像をベクトルにする

公開URLの画像を直接ベクトル化する。

```sql
WITH img AS (
  SELECT content
  FROM read_blob('https://raw.githubusercontent.com/vzat/comparing_images/master/images/pcb1.jpg')
)
SELECT pic_dim(pic_embed_blob(content))   AS dim,
       pic_embed_blob(content)[1:5]       AS first5
FROM img;
```

```
┌───────┬─────────────────────────────────────────────────────────────┐
│  dim  │                           first5                            │
├───────┼─────────────────────────────────────────────────────────────┤
│   192 │ [-1.1443413, -0.8578818, -0.61944926, 1.3491402, 0.5958941] │
└───────┴─────────────────────────────────────────────────────────────┘
```

`read_blob`がHTTPS URLからJPEGのバイト列を持ってきて、それを`pic_embed_blob`に渡すだけでベクトル化できている。

## 2. 「どこが違うか」を取り出す（pic_diff_patches + pic_diff_heatmap）

前節で出したのは画像1枚を192次元のベクトルに圧縮した結果で、画像全体の特徴を1本のベクトルに丸めたもの。これを2枚分とって距離を測れば「2枚がどれくらい似ているか」のスコアは出る（次の章でやる）が、それだけだと「どこがどう違うか」までは分からない。

そこで `pic_diff_patches` を使うと、画像をN×Nのマス目に切って、対応するマスごとの類似度を返してくれる。<br>さらに `pic_diff_heatmap` を使うと違うマスほど赤く塗ったPNG画像を書き出してくれる。

注意: この2つの関数は先にローカルへ落としてから渡す。

```sh
curl -O https://raw.githubusercontent.com/vzat/comparing_images/master/images/pcb1.jpg
curl -O https://raw.githubusercontent.com/vzat/comparing_images/master/images/pcb2.jpg
```

```sql
WITH d AS (
  SELECT unnest(pic_diff_patches('pcb1.jpg', 'pcb2.jpg')) AS p
)
SELECT (p).row AS row, (p).col AS col, round((p).sim, 3) AS sim
FROM d
ORDER BY sim ASC
LIMIT 5;
```

```
┌─────┬─────┬───────┐
│ row │ col │  sim  │
├─────┼─────┼───────┤
│   6 │   6 │ 0.783 │  ← 一番違う場所
│   8 │  10 │ 0.824 │
│   8 │   7 │ 0.848 │
│   8 │   6 │ 0.849 │
│   8 │   4 │ 0.849 │
└─────┴─────┴───────┘
```

`(row=6, col=6)`が一番違っていて、その周辺の8行目もまとめて低い。pcb1とpcb2の差分が**左下〜中央付近に集中している**ことがSQLだけで分かる。

ヒートマップも書き出してみる:

```sql
SELECT pic_diff_heatmap(
  'pcb1.jpg', 'pcb2.jpg',
  'pcb1_heatmap.png', 'pcb2_heatmap.png'
);
```

| pcb1.jpg にヒートマップを重ねた結果 | pcb2.jpg にヒートマップを重ねた結果 |
|---|---|
| ![pcb1_heatmap](/images/li9e653etwh9v8/heatmap_pcb1.png) | ![pcb2_heatmap](/images/li9e653etwh9v8/heatmap_pcb2.png) |

赤いマスが「もう片方と違う場所」 ＝ 間違い探しの答え。<br>pcb1とpcb2はもともと別撮りとはいえ「ほぼ同じ基板」なので、赤の出方は控えめ。ほぼ同じ画像なので位置ズレを拾った程度のように見える。若干ICの有無くらいは拾えているか？

### ペンの有無の「間違い探し」ペアでもう一回 (test1.1 vs test1.2)

vzatリポジトリには、もう一組 `test1.1.jpg` / `test1.2.jpg` という典型的な間違い探しペアも入っている。同じ被写体だが一部だけが書き換えられているタイプ。

```sh
curl -O https://raw.githubusercontent.com/vzat/comparing_images/master/images/test1.1.jpg
curl -O https://raw.githubusercontent.com/vzat/comparing_images/master/images/test1.2.jpg
```

```sql
SELECT pic_diff_heatmap(
  'test1.1.jpg', 'test1.2.jpg',
  'test1a_heatmap.png', 'test1b_heatmap.png'
);
```

| test1.1.jpg にヒートマップを重ねた結果 | test1.2.jpg にヒートマップを重ねた結果 |
|---|---|
| ![test1a_heatmap](/images/li9e653etwh9v8/heatmap_test1a.png) | ![test1b_heatmap](/images/li9e653etwh9v8/heatmap_test1b.png) |

embedding全体としてはほぼ同じ (`pic_similarity` で 0.9488) なのに、違うマスだけが赤く浮かび上がっている。これがそのまま「間違い探しの答え」になる。

冒頭の # アイデア で書いた「間違っている場所のヒートマップ」はこのパス。

## 3. 「2枚の画像がどれくらい似ているか」を数値化する

ここまでは2枚の画像をその場で比較していたが、今度は複数画像をテーブルに持って類似度を一気に出す。基準のpcb1のほか、「ほぼ同じ pcb2」「改ざん版 fakePCB3/5」「全然違う test1.1」を一気に取り込む。

```sql
CREATE OR REPLACE TABLE imgs AS
SELECT filename AS url,
       pic_embed_blob(content) AS vec
FROM read_blob([
  'https://raw.githubusercontent.com/vzat/comparing_images/master/images/pcb1.jpg',
  'https://raw.githubusercontent.com/vzat/comparing_images/master/images/pcb2.jpg',
  'https://raw.githubusercontent.com/vzat/comparing_images/master/images/fakePCB3.jpg',
  'https://raw.githubusercontent.com/vzat/comparing_images/master/images/fakePCB5.jpg',
  'https://raw.githubusercontent.com/vzat/comparing_images/master/images/test1.1.jpg'
]);
```

`read_blob`はURL配列もそのまま受けるので、公開画像のリストを直接テーブルにできる。中間ファイルも`requests`もいらない。

基準画像 pcb1 を「正解」として、各候補との類似度を並べる:

```sql
WITH q AS (
  SELECT pic_embed_blob(content) AS v
  FROM read_blob('https://raw.githubusercontent.com/vzat/comparing_images/master/images/pcb1.jpg')
)
SELECT regexp_extract(url, '[^/]+$') AS file,
       round(pic_similarity(vec, q.v)::DOUBLE, 4) AS sim
FROM imgs, q
ORDER BY sim DESC;
```

実測値:

```
┌──────────────┬────────┐
│     file     │  sim   │
├──────────────┼────────┤
│ pcb1.jpg     │ 1.0000 │  ← クエリと同じ画像、自己
│ pcb2.jpg     │ 0.9358 │  ← ほぼ同じ別撮り基板
│ test1.1.jpg  │ 0.5427 │  ← 全然違う画像、対照群
│ fakePCB3.jpg │ 0.4010 │  ← 改ざん版3(手書き)
│ fakePCB5.jpg │ 0.3643 │  ← 改ざん版5(手書き)
└──────────────┴────────┘
```

「同一画像なら1.0」「ほぼ同じ別撮りなら0.93台」「全然違う画像で0.54」「改ざん版は0.4台まで一気に落ちる」というのが1クエリで一発で見える。

ここで面白いのは、`fakePCB` の類似度 (0.4台) が、全然関係ない test1.1 (0.54) よりも低くなっていること。

| 基準 pcb1（Arduino実物写真） | fakePCB3 (sim=0.4010) | test1.1 (sim=0.5427) |
|---|---|---|
| ![pcb1](https://raw.githubusercontent.com/vzat/comparing_images/master/images/pcb1.jpg) | ![fakePCB3](https://raw.githubusercontent.com/vzat/comparing_images/master/images/fakePCB3.jpg) | ![test1.1](https://raw.githubusercontent.com/vzat/comparing_images/master/images/test1.1.jpg) |

実物を見るとカラクリは単純で、`fakePCB` 系は基板写真の加工版ではなく、**紙にボールペンで描いた手書きスケッチ**だった（基板の部品配置を模写したフェイク）。test1.1 は無関係な被写体（スマホ + 手帳ケース）だが、pcb1 と同じ「物体を真上から撮った実物写真」という大枠が共通しているので、EUPE から見ると pcb1 にやや近い。


### 位置ズレ画像はどう判定されるか

「同じ画像をちょっとだけ動かした版」だとどうなるか。pcb1.jpgを30px右・20px下にロール（`magick pcb1.jpg -roll +30+20 pcb1_shifted.jpg`）したものと、中央90%だけクロップしたもの（`pcb1_cropshift.jpg`）を試した:

| 基準 pcb1 | pcb1_shifted (sim=0.9751) | pcb1_cropshift (sim=0.9843) |
|---|---|---|
| ![pcb1](https://raw.githubusercontent.com/vzat/comparing_images/master/images/pcb1.jpg) | ![pcb1_shifted](/images/li9e653etwh9v8/pcb1_shifted.jpg) | ![pcb1_cropshift](/images/li9e653etwh9v8/pcb1_cropshift.jpg) |

```
┌────────────────────────┬────────┐
│        compared        │  sim   │
├────────────────────────┼────────┤
│ pcb1 vs pcb1_shifted   │ 0.9751 │
│ pcb1 vs pcb1_cropshift │ 0.9843 │
└────────────────────────┴────────┘
```

撮影誤差くらいの位置ズレは0.97〜0.98で「ほぼ同じ」と判定される。ベクトル化が「画像全体の意味」で見ているからで、ピクセル単位のdiffだと別画像扱いになるところを吸収してくれる。
ただしほぼ位置ズレ画像内での間違い探しは位置ズレの方も異なるものとして認識されてしまうのが先述した難点。

## 4. 結果をDuckDBに保存して使い回す

embeddingが乗ったテーブルはそのままParquetやDuckDBファイルに永続化できる。後から再読み込みすれば、何度でも類似度クエリを叩ける。

Parquet書き出し:

```sql
COPY imgs TO 'imgs.parquet' (FORMAT 'parquet');
```

別セッションで読み戻して、別画像との類似度をすぐに計算する:

```sql
LOAD pic2vec;
WITH q AS (
  SELECT pic_embed('/tmp/another.jpg') AS v
)
SELECT regexp_extract(url, '[^/]+$') AS file,
       round(pic_similarity(vec, q.v)::DOUBLE, 4) AS sim
FROM read_parquet('imgs.parquet'), q
ORDER BY sim DESC LIMIT 5;
```

新しい画像はINSERTで追記できる:

```sql
INSERT INTO imgs
SELECT filename AS url, pic_embed_blob(content) AS vec
FROM read_blob('https://example.com/new.jpg');
```

これで「画像のembedding表」を1ファイルとして持ち運べるようになる。簡易な画像分類リストや「過去に出た間違い探しの答え」を蓄積していくのもこの形でできる。


# 雑感

簡易的な画像の分類であればこのツールでローカルで完了しそうな感触は得た。ただ高度な異常検知となるとモデルでの学習が必要になりそうである。

また実行速度をM1 Mac CPUのみで測ったところ、`pic_embed` は1枚あたり以下くらい

| 入力 | 1枚あたり |
|---|---|
| 元 (645×540, 200KB) | **約 80ms** |
| 4K (3840×2160, 1.1MB) | **約 150ms** |

数千枚規模なら数分で回せる。8K/16Kになると JPEG decode + 224x224 へのリサイズが律速になってくるので、巨大画像は事前に縮小してから流すのがおすすめ。

なので手早く画像分類をして目検であからさまに違うものは弾くくらいのことはできるのではないか。



