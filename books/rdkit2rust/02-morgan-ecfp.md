---
title: "Ch2: Morgan / ECFP fingerprint — 層状 BFS + hash_combine"
free: true
---


# アイデア

[前回](https://github.com/nkwork9999/chemdatatravelers/blob/main/01_rdkit2rust/articles/01_logp.md) は logP (Wildman-Crippen 法) を SMARTS マッチで実装したが、今回は化学MLの主役級である **Morgan / ECFP fingerprint** に移る。

Morgan fingerprint は分子を「**周辺n結合以内の部分構造の集合**」に変換する。各部分構造をハッシュして固定長のビットベクタに詰めると、分子を 2048bit ぐらいの BLOB として扱えるようになる。
これがあれば

- 部分構造の重なりを **Tanimoto 類似度** で出して 「似てる分子探索」
- そのまま **ML モデルの入力**として使う (RDKitで一番使われる descriptor)
- DuckDB の `WHERE` で「特定の部分構造を持つ分子だけ」みたいなフィルタ

ができるようになる。今回は RDKit の `MorganGenerator.cpp` (596行) を読み解いて、Rust で再実装し、`SELECT morgan_fp_bits('CCO')` と書ける DuckDB スカラ関数として ducksmiles に組み込んだ。


# そもそも Morgan / ECFP とは

化学者の Morgan が 1965 年に提案した分子のキャノニカル化アルゴリズムを、Rogers と Hahn が 2010 年に再解釈し、現代の ECFP (Extended-Connectivity FingerPrint) として整備した。

イメージ:

1. **各原子に初期 ID を割り当てる**: 「炭素・2本結合・水素2個・電荷0・環の中」みたいな情報をハッシュした 32bit 整数
2. **半径 1 に広げる**: 各原子について、自分のIDと隣接原子のIDをまとめてハッシュ → 新しい ID
3. **半径 2 に広げる**: 半径 1 の結果を入力にもう一度ハッシュ → さらに広い周辺を表す ID
4. **これを `radius` 回繰り返す** (ECFP4 なら radius=2)
5. **出てきた ID 全部を `% 2048` してビットに立てる** → 2048bit の指紋

ECFP**4** の "4" は、原子1個から見て直径4結合分の情報が入ってる、という命名 (radius=2 × 2 = 4)。慣習的に ECFP4 = radius 2, ECFP6 = radius 3。


# 使ってみる

```sql
INSTALL ducksmiles FROM community;
LOAD ducksmiles;

-- デフォルト: ECFP4 = radius 2, 2048 bit → 256 byte の BLOB
SELECT name,
       octet_length(morgan_fp_bits(smiles))           AS bytes,
       bit_count(CAST(morgan_fp_bits(smiles) AS BIT)) AS pop
FROM (VALUES
  ('methane',  'C'),
  ('ethanol',  'CCO'),
  ('benzene',  'c1ccccc1'),
  ('aspirin',  'CC(=O)Oc1ccccc1C(=O)O'),
  ('caffeine', 'Cn1c(=O)c2c(ncn2C)n(C)c1=O')
) AS t(name, smiles);
```

実機の出力:

```
┌──────────┬───────┬─────┐
│   name   │ bytes │ pop │
├──────────┼───────┼─────┤
│ methane  │   256 │   1 │
│ ethanol  │   256 │   6 │
│ benzene  │   256 │   3 │
│ aspirin  │   256 │  26 │
│ caffeine │   256 │  28 │
└──────────┴───────┴─────┘
```

DuckDB の `bit_count()` は BIT 型を取るので、BLOB を `CAST(... AS BIT)` でビット列に変換してから popcount する。
立つビット位置は 2048 bit 全体に散らばるので、生の BLOB を `SELECT morgan_fp_bits(...)` で見るとほぼ `\x00` 連発、複雑な分子になるほどちらほら値が乗ってくる。

半径とビット幅を明示することもできる:

```sql
-- ECFP6 (radius=3) を 4096 bit で
SELECT bit_count(CAST(morgan_fp_bits('CC(=O)Oc1ccccc1C(=O)O', 3, 4096) AS BIT));
-- → 33
```


# 実装の中身

## 1. 読む

RDKit の Morgan 実装は **`MorganFingerprints.cpp` (158行) → `MorganGenerator.cpp` (596行) → `FingerprintGenerator.cpp` (基盤)** という3層構造。一番上の `MorganFingerprints.cpp` は API ラッパなので、本体は `MorganGenerator.cpp` の `MorganEnvGenerator::getEnvironments()` (291-503行) にある。

要旨を抜き出すとこう:

```cpp
// MorganGenerator.cpp getEnvironments() 抜粋
std::vector<OutputType> currentInvariants(...);  // 各原子の現在の不変量
std::vector<boost::dynamic_bitset<>> atomNeighborhoods(nAtoms, bonds);  // 各原子の「カバーした結合の集合」

// Round 0: 初期不変量をそのまま結果に詰める
for (i in 0..nAtoms) result.push_back(MorganAtomEnv(currentInvariants[i], i, 0, ...));

// Round 1..radius:
for (layer = 0; layer < radius; ++layer) {
  for (atomIdx in atomOrder) {
    if (deadAtoms[atomIdx]) continue;
    // 隣接の (bond_invariant, neighbor_invariant) を収集
    for (bond in atom.getBonds()) {
      roundAtomNeighborhoods[atomIdx][bond.idx] = 1;
      roundAtomNeighborhoods[atomIdx] |= atomNeighborhoods[bond.other];
      neighborhoodInvariants.push_back({bondInv, currentInvariants[other]});
    }
    sort(neighborhoodInvariants);
    // hash_combine で 1 個の uint32 に潰す
    uint32_t invar = layer;
    hash_combine(invar, currentInvariants[atomIdx]);
    for (pair in neighborhoodInvariants) hash_combine(invar, pair);
    nextLayerInvariants[atomIdx] = invar;
  }
  // 同じ近傍 (=結合集合) を既出ならスキップ、初出ならビットに採用
  for ((nbhd, invar, atomIdx) in this_round) {
    if (neighborhoods.insert(nbhd).second) result.push_back(...);
    else deadAtoms[atomIdx] = 1;
  }
  currentInvariants.swap(nextLayerInvariants);
}
```

ポイントは3つ:

1. **「初期不変量」の作り方** が ECFP のフレーバを決める (`MorganAtomInvGenerator::getAtomInvariants` → `getConnectivityInvariants`)。Daylight 風の場合は: heavy degree / atomic_num / 総 H 数 / formal charge / isotope / 環の中か。これを `boost::hash_range` でまとめる
2. **層ごとに `hash_combine` で潰す**: `seed ^= h + 0x9e3779b9 + (seed<<6) + (seed>>2)` という boost の定番式
3. **同一近傍の重複削除**: 「カバーした結合の集合」を bitset で持って `unordered_set<bitset>` に登録、既出ならその原子を以降の計算から除外 (deadAtoms)。これがないと環状分子で同じ部分構造が radius 倍カウントされてしまう


## 2. 考える — 4 つの設計判断

### (a) ハッシュ関数は boost::hash と互換にしない

RDKit のビットは `boost::hash<uint32_t>`, `boost::hash<pair<int32_t, uint32_t>>` といった boost 固有のハッシュ実装に依存している。bit-exact に合わせると boost のソースを Rust に丸ごと持ち込むことになって規模感が違う。

今回は **アルゴリズムの構造 (= 何を何で hash_combine するか) は完全に揃え、ハッシュ関数だけは Rust 側の自前 (boost 風の合成式) に置き換える** という方針にした。

```rust
fn hash_combine(seed: u32, v: u32) -> u32 {
    seed ^ (v
        .wrapping_add(0x9e37_79b9)
        .wrapping_add(seed << 6)
        .wrapping_add(seed >> 2))
}
```

代償として **同じ分子でも本家とは違うビットが立つ**。これは Tanimoto 類似度のランキング (= 「アスピリン に似てる分子 top10」) で本家と order が合えば実用上 OK で、次回 (#03 Tanimoto) でその検証を入れる予定。

### (b) 環の判定は自前で

ducksmiles のパーサ (`parser.rs`) は SMILES の環クロージャ (`c1ccccc1` の `1`) を見て結合は張ってくれるが、「各原子が環の中か」のフラグは持っていない。RDKit は SSSR (Smallest Set of Smallest Rings) を計算してから ring membership を出す。

今回はそこまで凝らず **DFS の back-edge を見つけたら、その閉路上の全原子を `in_ring=true`** にする素朴な実装にした:

```rust
fn compute_in_ring(mol: &Molecule) -> Vec<bool> {
    // DFS で各原子を辿り、back-edge (= 既訪問の非親頂点へ) を発見したら
    // u から v まで親リンクを遡って通った全頂点を ring とマーク
    ...
}
```

SSSR の正規最小環集合とは厳密には違う (たとえばナフタレンの fused ring で「中央の共有結合」の扱い)が、 ECFP の初期不変量に使う `in_ring` フラグの目的 (= 「環の中か外か」の二値) としては十分。

### (c) 出力は固定長 BLOB

RDKit の `getFingerprint` は `SparseIntVect<uint32_t>` (= ハッシュ ID → count の map) を返すのが本来の形だが、DuckDB スカラ関数で MAP を返すのは扱いにくい。

今回は **`getFingerprintAsBitVect` 相当 (ハッシュ ID を `% n_bits` してビットを立てる)** を素直にやって、`BLOB` を返す。
DuckDB の `BLOB` 型は `&` `|` や `bit_count()` を直接サポートしないが、**`CAST(blob AS BIT)` でビット列に変換すれば全部の論理演算が SQL レベルで使える** ようになる:

```sql
-- Tanimoto 類似度 (CCO vs CCN)
WITH x AS (
  SELECT CAST(morgan_fp_bits('CCO') AS BIT) AS a,
         CAST(morgan_fp_bits('CCN') AS BIT) AS b
)
SELECT bit_count(a & b)::DOUBLE / bit_count(a | b) AS tanimoto FROM x;
-- → 0.3333
```

次回 #03 ではこのパターンを軸に専用関数 `tanimoto_bit(BLOB, BLOB) → DOUBLE` を実装する。

### (d) スカラ関数のオーバーロード

`morgan_fp_bits(smi)` でデフォルト (ECFP4, 2048 bit) を呼び、`morgan_fp_bits(smi, radius, n_bits)` で全制御できるよう **同名で 2 つのスカラ関数を register** した。

```cpp
loader.RegisterFunction(ScalarFunction("morgan_fp_bits",
    {LogicalType::VARCHAR}, LogicalType::BLOB, MorganFpBitsFunc1));
loader.RegisterFunction(ScalarFunction("morgan_fp_bits",
    {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::INTEGER},
    LogicalType::BLOB, MorganFpBitsFunc3));
```

DuckDB の `ScalarFunction` 登録は同名異シグネチャ OK で、呼び出し側の引数型から自動で正しい方が選ばれる。


## 3. 作る — 250行のコード + FFI + DuckDB 登録

### Rust 側 ([crates/smiles/src/morgan.rs](https://github.com/nkwork9999/duckSMILES/blob/main/crates/smiles/src/morgan.rs))

主要部分:

```rust
pub fn morgan_invariants(mol: &Molecule, radius: u32) -> Vec<u32> {
    let in_ring = compute_in_ring(mol);
    let mut current: Vec<u32> = (0..mol.atoms.len())
        .map(|i| connectivity_invariant(mol, &in_ring, i))
        .collect();

    let mut result: Vec<u32> = current.clone();         // round 0
    let mut atom_nbhd: Vec<Vec<bool>> = ...;            // atom -> covered bond set
    let mut seen_nbhds = HashSet::new();
    let mut dead = vec![false; n];

    for layer in 0..radius {
        let mut next = current.clone();
        let mut this_round: Vec<(Vec<bool>, u32, usize)> = vec![];

        for atom_idx in 0..n {
            if dead[atom_idx] { continue; }
            let mut round_nbhd = atom_nbhd[atom_idx].clone();
            let mut neighbor_pairs = vec![];
            for &(bi, other, order) in &bonds_of[atom_idx] {
                round_nbhd[bi] = true;
                for b in 0..nbonds { if atom_nbhd[other][b] { round_nbhd[b] = true; } }
                neighbor_pairs.push((bond_invariant(order), current[other]));
            }
            neighbor_pairs.sort();
            let mut invar = layer;
            invar = hash_combine(invar, current[atom_idx]);
            for (bi, ni) in &neighbor_pairs {
                invar = hash_combine(invar, *bi);
                invar = hash_combine(invar, *ni);
            }
            next[atom_idx] = invar;
            this_round.push((round_nbhd, invar, atom_idx));
        }

        this_round.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)));
        for (nbhd, invar, atom_idx) in &this_round {
            if seen_nbhds.insert(nbhd.clone()) { result.push(*invar); }
            else { dead[*atom_idx] = true; }
        }
        ...
        current = next;
    }
    result
}

pub fn morgan_bits(mol: &Molecule, radius: u32, n_bits: u32) -> Vec<u8> {
    let mut bits = vec![0u8; ((n_bits + 7) / 8) as usize];
    for inv in morgan_invariants(mol, radius) {
        let bit = (inv % n_bits) as usize;
        bits[bit / 8] |= 1 << (bit % 8);
    }
    bits
}
```

### FFI ([crates/smiles/src/lib.rs](https://github.com/nkwork9999/duckSMILES/blob/main/crates/smiles/src/lib.rs))

```rust
#[unsafe(no_mangle)]
pub extern "C" fn ds_morgan_fp_bits(
    ptr: *const u8, len: usize,
    radius: u32, n_bits: u32,
    out: *mut u8, out_cap: usize,
) -> i32 {
    let n_bytes = ((n_bits as usize) + 7) / 8;
    if n_bytes > out_cap || n_bits == 0 { return -1; }
    let s = unsafe { ... };
    match parse(s) {
        Some(mol) => {
            let bits = morgan_bits(&mol, radius, n_bits);
            unsafe { std::ptr::copy_nonoverlapping(bits.as_ptr(), out, n_bytes); }
            n_bytes as i32
        }
        None => -1,
    }
}
```

### DuckDB 側 ([src/ducksmiles_extension.cpp](https://github.com/nkwork9999/duckSMILES/blob/main/src/ducksmiles_extension.cpp))

```cpp
static void MorganFpBitsFunc3(DataChunk &args, ExpressionState &state, Vector &result) {
    auto smi_data    = FlatVector::GetData<string_t>(args.data[0]);
    auto radius_data = FlatVector::GetData<int32_t>(args.data[1]);
    auto nbits_data  = FlatVector::GetData<int32_t>(args.data[2]);
    auto &validity   = FlatVector::Validity(result);
    for (idx_t i = 0; i < args.size(); i++) {
        uint8_t buf[16384];
        int32_t len = ds_morgan_fp_bits(
            (const uint8_t *)smi_data[i].GetData(), smi_data[i].GetSize(),
            (uint32_t)radius_data[i], (uint32_t)nbits_data[i],
            buf, sizeof(buf));
        if (len < 0) { validity.SetInvalid(i); continue; }
        FlatVector::GetData<string_t>(result)[i] =
            StringVector::AddStringOrBlob(result, (const char *)buf, len);
    }
}
```

`BLOB` を返すスカラ関数は VARCHAR と同じく `string_t` に書き出すが、`AddString` ではなく `AddStringOrBlob` を使うのがポイント (BLOB が embedded null `\x00` を含むので、文字列終端で切られると困る)。


## 4. 検証 — 性質ベースのテスト

bit-exact RDKit 一致は今回見送ったので、代わりに **アルゴリズムの性質ベース** で 11 個のテストを書いた:

```rust
#[test] fn methane_round0_only() { ... }                    // 1原子分子 + radius 0 → 1 invariant
#[test] fn benzene_has_ring_invariants() { ... }            // 全原子が in_ring
#[test] fn ethanol_chain_no_ring() { ... }                  // 鎖は in_ring なし
#[test] fn cyclohexane_in_ring() { ... }                    // 6原子全部 in_ring
#[test] fn same_smiles_same_bits() { ... }                  // 決定的
#[test] fn different_smiles_different_bits() { ... }        // CCO ≠ c1ccccc1
#[test] fn bit_count_grows_with_radius() { ... }            // r2 ≥ r0
#[test] fn bits_are_2048() { ... }                          // 256 byte
#[test] fn aspirin_radius2_sets_many_bits() { ... }         // 複雑な分子は ≥10 bit
#[test] fn empty_radius_still_returns_correct_size() { ... }// radius 0 でも valid
#[test] fn caffeine_radius2_distinct_from_benzene() { ... } // 別の分子は別の指紋
```

全部 pass。`cargo test -p ducksmiles_smiles` で 291 テスト中 11 がこの morgan モジュール。

実機 popcount での確認 (`cargo test ... popcount_report -- --nocapture` 実測):

| 分子 | heavy 原子数 | ECFP4 (r=2, 2048bit) | ECFP6 (r=3, 4096bit) |
|---|---|---|---|
| methane (C)         |  1 |  1 |  1 |
| water (O)           |  1 |  1 |  1 |
| ethanol (CCO)       |  3 |  6 |  6 |
| benzene             |  6 |  3 |  4 |
| phenol              |  7 | 11 | 13 |
| aspirin             | 13 | 26 | 33 |
| caffeine            | 14 | 28 | 37 |
| cholesterol         | 28 | 47 | 67 |

立つビット数は heavy 原子数に対して概ね単調に増え、radius を 2→3 に上げると環/分岐の多い分子で増分が大きい (cholesterol +20bit) — 半径が広がって新しい部分構造が見え始める挙動と合う。

ただし**ビット位置そのものは RDKit 本家と一致しない** (前述の通り hash 関数が違うため)。 「アスピリンに似た分子 top10」 のような Tanimoto 類似度ランキングが RDKit と概ね揃うか、は次回 #03 で AqSolDB / ChEMBL から検証する。


# 雑感

Morgan の本体は `hash_combine` × `dead_atoms 管理` の数十行で、Crippen のときと同じく「読むと意外と素朴」なアルゴリズムだった。重いのは boost の templates と FingerprintGenerator 基盤の方で、本体は層ごとの BFS + 重複削除。

次回 **#03 Tanimoto** で、この BLOB ペアから類似度を返す `tanimoto_bit(BLOB, BLOB) → DOUBLE` を実装する (`CAST AS BIT` 経由の素朴な `bit_count(a & b) / bit_count(a | b)` をベースに、SIMD popcount で高速化)。ChEMBL などのコンパウンドライブラリに対して「アスピリンに似た 100 件」を SQL 一本で返せるところまで持っていく予定。

その先は **#04 MACCS keys** (166bit 固定の単純なもの) → **#05 Lipinski 5** (#01 logP の再利用)、で featurization レイヤーがおおむね揃う。


---

### 参照

- 論文: D. Rogers, M. Hahn, *Extended-Connectivity Fingerprints*, J. Chem. Inf. Model. **50**, 742-754 (2010). DOI: [10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
- RDKit 本家: [MorganFingerprints.cpp](https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Fingerprints/MorganFingerprints.cpp) / [MorganGenerator.cpp](https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Fingerprints/MorganGenerator.cpp)
- ducksmiles repo: [github.com/nkwork9999/duckSMILES](https://github.com/nkwork9999/duckSMILES)
- DuckDB community extensions: [duckdb.org/community_extensions/](https://duckdb.org/community_extensions/)
