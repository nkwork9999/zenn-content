---
title: "RDKitの実装を読みながら を Rust + DuckDB で化学データ用のツールを作っていく　#01 — Wildman-Crippen logP"
emoji: "🏃‍➡️"
type: "tech"
topics: ["duckdb", "rust", "rdkit", "chemistry", "cheminformatics"]
published: false
---


# アイデア

ケモインフォマティクスの定番ライブラリ **RDKit (C++ + Python)** を、1機能ずつ読み解き、参考にしながら **Rust + DuckDB extension** で化学データ用ツールを作っていくシリーズの 1 回目。

題材は **logP** — 化合物が「水と油のどちらに溶けやすいか」を1つの数値で表す指標。<br>
創薬で「経口で飲める薬っぽさ」を判定するものでも使う(らしい)。

計算法は数十種類あるが、今回は RDKit が標準で使っている 「Wildman-Crippen 法」 (1999) を移植した。


RDKit 本家の [`Crippen.cpp`](https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Descriptors/Crippen.cpp) (322 行) 

寄与値テーブル [`Crippen.txt`](https://github.com/rdkit/rdkit/blob/master/Data/Crippen.txt) 

これを元にRust で再実装し、DuckDB community extension `ducksmiles` から `SELECT logp_crippen('CCO')` のように SQL で呼べるところまで持っていった。

### この回でやったこと

- **読む**: RDKit の `Crippen.cpp` を読み解きアルゴリズムの確認。
- **考える**:
  - SMARTSは今回の用途に絞る (汎用なら数千行、限定なら 600 行)
  - 省略された水素原子は内部で自動展開
  - Rust 標準ライブラリだけで完結
- **作る**: 分子パターンを探す Rust コード (約600行) + 寄与値の対応表を移植 + DuckDB から呼べるようにする。


# そもそも

logP は化合物が脂質と水のどちらに溶けやすいかの指標で現状は以下のような選択肢がある。

- **RDKit (Python)**: 一行で実装可能。
- **OpenBabel**: 同等のことが可能、計算式は別系統
- **PubChem の Calculated XLogP3**: 分子ごとに API 叩く

DuckDB拡張では

- Python に取り出して計算 → 結果を戻す、という往復を防止する。
- そもそも「SQL の中で WHERE / GROUP BY と一緒に descriptor を扱える。


# 作ったもの

## ducksmiles

https://duckdb.org/community_extensions/extensions/ducksmiles
https://github.com/nkwork9999/duckSMILES

RDKit の `Descriptors.MolLogP` 相当を、DuckDB の SQL 関数として呼べるようにした。記事では:

1. SQL から logP を計算する (使い方)
2. 中身の実装解説 (RDKit から何を写経し、何を Rust 流に再設計したか + 数値検証)


# 使ってみる

```sql
INSTALL ducksmiles FROM community;
LOAD ducksmiles;
```

VALUES 句でいくつか SMILES を並べて、`logp_crippen()` を流す:

```sql
SELECT name,
       round(logp_crippen(smiles), 4) AS logp,
       mol_formula(smiles)            AS formula
FROM (VALUES
  ('methane',     'C'),
  ('water',       'O'),
  ('benzene',     'c1ccccc1'),
  ('aspirin',     'CC(=O)Oc1ccccc1C(=O)O'),
  ('caffeine',    'Cn1c(=O)c2c(ncn2C)n(C)c1=O'),
  ('cholesterol', 'CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C')
) AS t(name, smiles);
```

実機の出力:

```
┌─────────────┬─────────┬───────────┐
│    name     │  logp   │  formula  │
├─────────────┼─────────┼───────────┤
│ methane     │  0.6361 │ CH4       │
│ water       │ -0.8247 │ H2O       │
│ benzene     │  1.6866 │ C6H6      │
│ aspirin     │  1.3101 │ C9H8O4    │
│ caffeine    │ -1.0293 │ C8H10N4O2 │
│ cholesterol │  7.3887 │ C27H46O   │
└─────────────┴─────────┴───────────┘
```

普通の SQL 関数なので `WHERE logp_crippen(smiles) <= 5 AND mol_weight(smiles) <= 500` のようにフィルタにも使える (Lipinski's Rule of Five 的な drug-likeness 判定が SQL 一本で書ける)。


# 実装の中身

## 1. 読む

RDKit の `Crippen.cpp` (322 行) を読むと、実質的なアルゴリズムは 80 行程度しかない。本体ループ (要約):

```cpp
// RDKit (Crippen.cpp)
boost::dynamic_bitset<> atomNeeded(mol.getNumAtoms());
atomNeeded.set();

for (const auto &param : *params) {                    // 110 SMARTS パターンを順に
  SubstructMatch(mol, *(param.dp_pattern.get()), matches, false, true);
  for (const auto &match : matches) {
    int idx = match[0].second;
    if (atomNeeded[idx]) {                             // まだ未割当なら採用 (先勝ち)
      atomNeeded[idx] = 0;
      logpContribs[idx] = param.logp;
    }
  }
  if (atomNeeded.none()) break;                        // 全原子割当済みなら終了
}
```

要点:

1. 110 パターンは「狭い (具体的) → 広い (汎用)」の順で並んでいて、各原子について上から順に試し、最初に当てはまったものだけ採用する。

エタノール (`CCO`) の最初のC原子 (= H3個 + Cに繋がる、つまりメチル基) で何が起きるかを追うとイメージしやすい:

| 試行順 | パターン | 結果 |
|---|---|---|
| #1 | `[CH4]` (メタン = H4個) | ✗ このCはH3個なので不一致 |
| #2 | `[CH3]C` (メチル基) | ✓ マッチ! このCに 0.1441 を割当て、以降は他パターンを試さない |
| #3〜 | `[CH2](C)C` 等 | スキップ (もう割当済) |
| #108 | `[#6]` (何でもC) | スキップ (もし#2が無ければここで 0.08129 が割当てられていた) |

もし順番が逆 (広い→狭い) だったら、メチル基のCも末尾の汎用パターン `[#6]` で先に拾われて 0.08129 (汎用炭素値) になってしまい、本来の 0.1441 にならない。

2. 最後に `logpContribs` (= 各原子の寄与値の配列) を全原子分足し合わせると、その分子の logP になる。


## 2. 考える — 3 つの設計判断

写経できる部分は写経しつつ、Rust + DuckDB 用に...

### (a) SMARTS は今回の用途に絞る (汎用なら数千行、限定なら 600 行)

SMARTS は SMILES の拡張で、「分子の中の特定のパターン」を文字列で書ける記法。
- `[CH3]` = メチル基にマッチ
- `[#6]=O` = カルボニルにマッチ
- `[A;!#1]` = 「水素以外の任意原子」にマッチ

RDKit には何でも書けるSMARTSが入っているが、これを完全再現すると数千行になる。一方、Crippen の 110 パターンが要求する文法はずっと狭いのでそこだけ実装。


| 機能 | 対応 | 例 |
|---|---|---|
| 元素 (脂肪族 / 芳香族) | ✅ | `C`, `c`, `[Cl]` |
| 原子番号 | ✅ | `[#6]`, `[#1]` |
| H 数・接続数 | ✅ | `[CH3]`, `[CX4]` |
| 電荷 | ✅ | `[NH+0]`, `[O-]` |
| 論理演算 (AND `;` / OR `,` / NOT `!`) | ✅ | `[A;!#1]` |
| 結合次数 | ✅ | `=`, `#`, `:`, `-` |
| 分岐 | ✅ | `(...)` |


### (b) 暗黙水素は内部で自動展開する

SMILES では H 原子は省略されるのが普通。`CCO` は heavy 原子 3 個しか書かれていないが、実際の C2H6O は 6 個の H 原子を持つ。H 原子も他の原子 (C, O) と同じく「分子の構成原子」として明示的に持たせる必要がある。

RDKit の場合: logP 計算の前にユーザー側で `mol = AddHs(mol)` を明示的に呼んで H を付ける必要がある。

「H ありで処理したい場面」(配座生成・力場計算など) と「H なしで処理したい場面」(反応式の表示・部分構造検索など) を選べる。

ducksmiles の場合: SQL 関数として `SELECT logp_crippen('CCO')` で呼ばれるので、デフォルトでは関数の内側で水素補完を自動実行する (毎行ごとに「先に補完してから渡して」とユーザーに強要するのは現実的でないため)。
一方で ユーザーが事前に補完したい場合に備え、`add_hydrogens(smiles)` という独立関数を別途用意した:

```sql
-- デフォルト: H 省略形 SMILES を渡す → 内部で自動補完
SELECT logp_crippen('CCO');                       -- → -0.0014

-- 明示形 SMILES を渡す (RDKit の MolToSmiles(allHsExplicit=True) 風のコンパクト形)
SELECT logp_crippen('[CH3][CH2][OH]');            -- → -0.0014  (同じ結果)

-- ducksmiles の add_hydrogens() で展開してから渡す (H を完全に独立ノードにした冗長形)
SELECT add_hydrogens('CCO');
-- → [C]([C]([O][H])([H])[H])([H])([H])[H]
SELECT logp_crippen(add_hydrogens('CCO'));        -- → -0.0014  (同じ結果)
```

の 2 通りがあるが、ducksmiles はどちらの形でも parse できるようにしている。`logp_crippen` 内部の水素補完は idempotent (= すでに H が明示済みなら何もしない) なので、どの形を渡しても結果が壊れない。


エタノールの例: `SELECT logp_crippen('CCO')` の内部の流れ:

## 3. 作る — 600 行のコード + 寄与値表の移植 + DuckDB 関数登録

### 分子グラフは最小限の構造体に

RDKitは最小環集合・芳香族判定キャッシュなど化学操作全般のためのデータを一通り持つが、今回の実装では logP を計算するだけなのでもっと単純になっている。:

```rust
pub struct Atom { symbol: String, hydrogen: i32, charge: i32, aromatic: bool, ... }
pub enum BondOrder { Single, Double, Triple, Aromatic }
pub struct Bond { a: usize, b: usize, order: BondOrder }
pub struct Molecule { atoms: Vec<Atom>, bonds: Vec<Bond> }
```

### メインループは RDKit を 1 対 1 でrustへ

```rust
// ducksmiles (logp_crippen.rs)
let mut assigned = vec![false; n];
let mut contrib  = vec![0.0_f64; n];

for (pat, logp) in patterns() {
    for i in 0..n {
        if !assigned[i] && match_at(pat, &mol_h, i) {
            assigned[i] = true;
            contrib[i] = *logp;
        }
    }
    if assigned.iter().all(|&x| x) { break; }
}
contrib.iter().sum()
```

### 寄与値テーブル (`Crippen.txt`) を Rust の static 配列に移植

110 個の SMARTS と寄与値を配列として埋め込み、初回呼び出しでパース結果をキャッシュ。

### 関連ファイル

- [crates/smiles/src/parser.rs](https://github.com/nkwork9999/duckSMILES/blob/main/crates/smiles/src/parser.rs) — SMILES パーサ + `Bond` / `BondOrder` / `with_explicit_hydrogens`
- [crates/smiles/src/smarts.rs](https://github.com/nkwork9999/duckSMILES/blob/main/crates/smiles/src/smarts.rs) — 限定 SMARTS パーサ + マッチャ (約 600 行)
- [crates/smiles/src/logp_crippen.rs](https://github.com/nkwork9999/duckSMILES/blob/main/crates/smiles/src/logp_crippen.rs) — Crippen テーブル + メインループ
- [src/ducksmiles_extension.cpp](https://github.com/nkwork9999/duckSMILES/blob/main/src/ducksmiles_extension.cpp) — DuckDB ScalarFunction 登録


## 4. 検証 — RDKit Python と数値が一致するか

実装が信頼できるかは、本家 RDKit の値と照合するのが早い。

| 分子 | ducksmiles | RDKit (`Descriptors.MolLogP`) | 差 |
|---|---|---|---|
| methane | +0.6361 | +0.6361 | **完全一致** |
| water | -0.8247 | -0.8247 | **完全一致** |
| ethanol | -0.0014 | -0.0014 | **完全一致** |
| methanol | -0.3915 | -0.3915 | **完全一致** |
| benzene | +1.6866 | +1.6866 | **完全一致** |
| aspirin | +1.3101 | ~1.19 | 0.12 |
| cholesterol | +7.3887 | ~7.02 | 0.37 |
| caffeine | -1.0293 | ~-0.55 | 0.48 |

小分子は完全一致。複雑な薬物分子で 0.1〜0.5 log unit ずれてしまう。
これは Wildman-Crippen 法そのものの床精度 (σ=0.68) の範囲内？
原因は SMARTS の `;` (AND_LO) と `,` (OR) の優先順位の解釈差や、特殊な原子型 (アミド N、双性イオン) か...?


# 雑感

RDKit を読み解いてみると、Crippen のようなな記述子計算は実は素朴で、「SMARTS マッチして寄与値を足す」だけのループ に集約される。
次回は TPSA (極性表面積) を同じ流儀で移植する予定。「分子のどこが極性か」を面積で表現する記述子で、Crippen と同じ寄与値テーブル方式 ([Ertl 2000](https://doi.org/10.1021/jm000942e))。


---

### 参照

- 論文: S. A. Wildman, G. M. Crippen, *J. Chem. Inf. Comput. Sci.*, **39**, 868-873 (1999). DOI: [10.1021/ci990307l](https://doi.org/10.1021/ci990307l)
- RDKit 本家: [github.com/rdkit/rdkit](https://github.com/rdkit/rdkit) ([Crippen.cpp](https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Descriptors/Crippen.cpp) / [Crippen.txt](https://github.com/rdkit/rdkit/blob/master/Data/Crippen.txt))
- ducksmiles repo: [github.com/nkwork9999/duckSMILES](https://github.com/nkwork9999/duckSMILES)
- DuckDB community extensions: [duckdb.org/community_extensions/](https://duckdb.org/community_extensions/)
