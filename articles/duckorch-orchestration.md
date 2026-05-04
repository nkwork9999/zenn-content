---
title: "DuckDB単体でDAGオーケストレーションを完結させる自作拡張機能を作った"
emoji: "🦆"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["duckdb", "rust", "sql", "dataengineering", "dataops"]
published: true
published_at: 2026-05-10 21:00
---


# アイデア

DuckDBは高速な前処理や、データパイプラインとなる十分なポテンシャルはあるもののデータ基盤としては弱いため、**duckdb上でSQLでワークフロー制御できないか？**機能として試作したかった。

### できること
- `-- @task name=...` と冒頭につけるだけでDAG定義
- 依存関係はSQL本文から自動抽出 
- タスク完了後テスト・差分処理・定期実行など可能
- テーブル/カラム単位のリネージが`__orch__.lineage_edges` / `__orch__.column_lineage` に自動で貯まる (OpenLineage送出 + Mermaid可視化付き)


# そもそも

DuckDBは非常に強力なデータ分析ツールだが、「複数のSQLを順番に実行して、リネージを残す」といったパイプライン的な処理は本体ではカバーされていない。

これをやろうとすると、以下のような選択肢があると思う。

- **dbt** を別プロセスで立てる
- **Airflow / Dagster / Prefect** を別途構築する
- **SQLMesh** を使う

しかしこれだとローカルで分析したいだけなのに、フォルダ構成 / 依存パッケージ / 環境変数 / 接続設定 を整える必要がある。
DuckDBはデータ分析ツールであるだけでなく、それ単体でDBエンジンとして動くツールでもある。これにワークフロー処理を加えたら済むような場面もあるのでは？と思い今回作成してみました。
(あとDuckLakeとの連携もしてみたかった)


# 作ったもの


## duckorch
https://duckdb.org/community_extensions/extensions/duckorch
https://github.com/nkwork9999/duck-orch

「DuckDB一個でDAG実行 + リネージ取得などが全部終わる」ような少量かつローカルデータならduckdb基盤でよくなるような状態にできればと思っている。
とりあえず記事では以下機能の実行を試す。


1. SQLファイル1つ = 1タスクとしてタスク実行する。
2. 依存関係を書かなくても自動抽出する。
3. タスク完了後テスト・差分処理・定期実行も実行する。
4. カラムレベルリネージをついでにOpenLineageのサブタイプ(IDENTITY / FILTER / GROUP_BY 等)に分類する。
5. リネージをMermaidで可視化する。
6. DuckLake上のデータでもOpenLineageと連携する。
7. 全CLIサブコマンドに`--json`を生やしてAIエージェントから扱えるようにする。
   このテーブルを変更すると何が壊れるかを構造化JSONで返せる。


# 使ってみる

データは[NeurIPS Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)のtrain.csv<br>
（7,974件のポリマー物性データ：`SMILES`/`Tg`(ガラス転移温度)/`FFV`(自由体積分率)/`Tc`(熱伝導率)/`Density`/`Rg`(回転半径)）。<br>抜けが多いのでそれを掃除して密度別に集計するという素直なパイプラインで動かす。

実機で実行したログをそのまま貼る。

```sh
duckdb test.db
```

```sql
INSTALL duckorch FROM community;
LOAD duckorch;
PRAGMA orch_init;
```

`PRAGMA orch_init`で`__orch__`スキーマ配下に状態テーブル(`tasks` / `runs` / `lineage_edges` / `column_lineage` / `task_edges` / `tests` / `schedules`)が作られる。

CSVを生のままテーブルに取り込む。

```sql
CREATE OR REPLACE TABLE raw_polymers AS
SELECT * FROM read_csv_auto('train.csv');
```

## 1. SQLファイル1つ = 1タスクとしてタスク実行する

組むパイプラインは以下2タスク。

- clean_polymers: 元の生の分子データ7,974件から、一部の行だけを残す前処理。<br>ここでは`Density` が抜けている行や、`Tg`/`FFV`/`Tc` のどれも取れていない行を便宜的に捨てている。
- density_stats: クリーン済みの polymer を密度3バンド(low/mid/high)に分けて、件数と `Tg`/`FFV` の平均を出す集計

それぞれ普通のSQLファイルとして書く。まずは `clean_polymers`:

```sql
-- @task name=clean_polymers
-- @outputs clean_polymers
-- @retries 2

CREATE OR REPLACE TABLE clean_polymers AS
SELECT id, SMILES, Tg, FFV, Tc, Density, Rg
FROM raw_polymers
WHERE Density IS NOT NULL
  AND (Tg IS NOT NULL OR FFV IS NOT NULL OR Tc IS NOT NULL);
```

冒頭のコメントヘッダの意味:

- `-- @task name=clean_polymers`: このSQLを `clean_polymers` という名前のタスクとして登録する
- `-- @outputs clean_polymers`: このタスクが書き込むテーブル名。依存解決と影響範囲調査で使われる
- `-- @retries 2`: 失敗したら最大2回まで指数バックオフでリトライする

次に `density_stats`:

```sql
-- @task name=density_stats
-- @outputs density_stats
-- @test "SELECT COUNT(*) FROM density_stats WHERE n < 0" expect 0

CREATE OR REPLACE TABLE density_stats AS
SELECT
  CASE
    WHEN Density < 1.0 THEN 'low'
    WHEN Density < 1.3 THEN 'mid'
    ELSE 'high'
  END AS density_band,
  COUNT(*) AS n,
  AVG(Tg)  AS avg_tg,
  AVG(FFV) AS avg_ffv
FROM clean_polymers
GROUP BY density_band;
```

新しく出てくる記法:

- `-- @test "<SQL>" expect 0`: タスク完了後にこのSQLを実行して、結果が`0`（= 1行目の値が0）であればOK。`n < 0` という不正な行が残っていたらFAIL扱いになる

`@inputs` は両タスクとも書いていないが、SQL本文に `raw_polymers` / `clean_polymers` が出てくるのでduckorchが自動で依存を解決する（次節）。

登録して実行する:

```sql
PRAGMA orch_register('./tasks/');
PRAGMA orch_run;

SELECT task_name,
       status,
       retry_count,
       (epoch_ms(finished_at) - epoch_ms(started_at)) AS duration_ms
FROM __orch__.runs
ORDER BY started_at;
```

実機の出力:

```
┌────────────────┬─────────┬─────────────┬─────────────┐
│   task_name    │ status  │ retry_count │ duration_ms │
├────────────────┼─────────┼─────────────┼─────────────┤
│ clean_polymers │ success │           0 │           4 │
│ density_stats  │ success │           0 │           3 │
└────────────────┴─────────┴─────────────┴─────────────┘
```

依存順に実行され、`@retries 2`で失敗時のリトライ、最終FAIL時は下流の自動スキップが効く。

集計結果も普通に引ける:

```sql
SELECT * FROM density_stats ORDER BY density_band;
```

```
┌──────────────┬───────┬────────┬─────────┐
│ density_band │   n   │ avg_tg │ avg_ffv │
├──────────────┼───────┼────────┼─────────┤
│ high         │    20 │  56.20 │  0.3904 │
│ low          │   375 │  58.27 │  0.3844 │
│ mid          │   190 │ 145.77 │  0.3661 │
└──────────────┴───────┴────────┴─────────┘
```

中密度(1.0〜1.3)のポリマーが高Tg、というのがざっくり見える。

## 2. 依存関係を書かなくても自動抽出する

タスクファイルに `@inputs` を書いていなくても、SQL本文から「読んでるテーブル」「書いてるテーブル」を自動抽出する。中で動いているのは `orch_extract_io` というスカラー関数で、これ単体でも呼べる:

```sql
SELECT orch_extract_io(
  'CREATE OR REPLACE TABLE clean_polymers AS
   SELECT id, SMILES, Tg, FFV, Tc, Density, Rg
   FROM raw_polymers WHERE Density IS NOT NULL'
) AS result;
```

実機の出力:

```
┌──────────────────────────────────────────────────────────┐
│                          result                          │
├──────────────────────────────────────────────────────────┤
│ {"inputs":["raw_polymers"],"outputs":["clean_polymers"]} │
└──────────────────────────────────────────────────────────┘
```

SQLを解析して `inputs` / `outputs` を返す。タスク登録時にも同じ関数が呼ばれて依存リストが組まれている。

## 3. タスク完了後テスト・差分処理・定期実行も実行する

### `@test` — タスク完了後アサーション

`density_stats.sql`の冒頭に書いた

```
-- @test "SELECT COUNT(*) FROM density_stats WHERE n < 0" expect 0
```

がアサーション。タスク登録時に`__orch__.tests`に貯まり、`PRAGMA orch_test`で一括実行する:

```sql
PRAGMA orch_test;
```

実機の出力:

```
Tests: 1 passed, 0 failed
```

アサーションは `expect 0` 以外にも以下が使える:

- `expect N`: 結果(1行目1列目)が `N` と等しければOK
- `expect_gt N`: 結果が `N` より大きければOK
- `expect_lt N`: 結果が `N` より小さければOK
- `expect_empty`: 結果が0行ならOK（「想定外の行が1件もない」のチェック向き）
- `expect_non_empty`: 結果が1行以上あればOK（「最低限のデータが入っている」のチェック向き）

### `@incremental_by` — 差分処理

毎回全件再計算したくない場合、`@incremental_by` と `{{ last_processed_at }}` プレースホルダで差分処理ができる。`{{ last_processed_at }}` は **そのタスクが最後に成功した時の処理対象タイムスタンプ** に自動で展開される。

ポリマーデータには `added_at` カラムが無いので、合成の小データで動作確認する。

```sql
-- 100行のraw_dataをタイムスタンプ付きで作る
CREATE OR REPLACE TABLE raw_data AS
SELECT i AS id, random() AS val,
       TIMESTAMP '2026-01-01 00:00:00' + INTERVAL (i) SECOND AS added_at
FROM range(100) t(i);

-- 受け側テーブルは事前に作っておく(空)
CREATE OR REPLACE TABLE clean_inc(id BIGINT, val DOUBLE, added_at TIMESTAMP);
```

差分処理タスク:

```sql
-- tasks/clean_inc.sql
-- @task name=clean_inc
-- @outputs clean_inc
-- @incremental_by added_at

INSERT INTO clean_inc
SELECT id, val, added_at
FROM raw_data
WHERE added_at > {{ last_processed_at }};
```

1回目の実行(100行投入される):

```sql
PRAGMA orch_register('./tasks/');
PRAGMA orch_run;

SELECT count(*) AS rows, max(added_at) AS max_added FROM clean_inc;
SELECT task_name, last_processed_at FROM __orch__.runs ORDER BY started_at DESC LIMIT 1;
```

```
┌──────┬─────────────────────┐
│ rows │      max_added      │
├──────┼─────────────────────┤
│  100 │ 2026-01-01 00:01:39 │
└──────┴─────────────────────┘
┌───────────┬─────────────────────┐
│ task_name │  last_processed_at  │
├───────────┼─────────────────────┤
│ clean_inc │ 2026-01-01 00:01:39 │
└───────────┴─────────────────────┘
```

ここで raw_data に新しい行を追加してから、もう一度 `orch_run`:

```sql
INSERT INTO raw_data
SELECT 100 + i, random(), TIMESTAMP '2026-01-01 00:02:00' + INTERVAL (i) SECOND
FROM range(50) t(i);

PRAGMA orch_run;

SELECT count(*) AS rows, max(added_at) AS max_added FROM clean_inc;
SELECT task_name, last_processed_at FROM __orch__.runs ORDER BY started_at DESC LIMIT 1;
```

```
┌──────┬─────────────────────┐
│ rows │      max_added      │
├──────┼─────────────────────┤
│  150 │ 2026-01-01 00:02:49 │
└──────┴─────────────────────┘
┌───────────┬─────────────────────┐
│ task_name │  last_processed_at  │
├───────────┼─────────────────────┤
│ clean_inc │ 2026-01-01 00:02:49 │
└───────────┴─────────────────────┘
```

2回目では **追加した50行だけ** が `clean_inc` に流れ込んで150行になり、最初の100行は再処理されていない。`last_processed_at` も新しい最終時刻まで進んでいる。

### `duck-orch schedule` — cron常駐(定期実行)

```bash
duck-orch --db mydata.db schedule add density_stats "0 6 * * *"
duck-orch --db mydata.db schedule list --json
```

実機の出力:

```json
[{"pipeline_or_task":"density_stats","cron_expr":"0 6 * * *","enabled":"true","next_trigger_at":"2026-05-04 17:03:15.556949"}]
```

`duck-orch schedule daemon` ターミナルで起動しておけば、30秒ごとにスケジュール実行する。

注意点:
- HTTPサーバではなくただの常駐プロセス。ポートも開かない、無限ループでDBをポーリングしているだけ
- 常駐させたいなら `systemd` / `launchd` / `tmux` を現状は使用する必要あり。

この辺りはおいおい改善できればと思っている。

## 4. カラムレベルリネージをついでにOpenLineageのサブタイプに分類する

テーブル間の流れは `__orch__.lineage_edges`、列単位は `__orch__.column_lineage` に貯まる。


```sql
SELECT src_dataset, dst_dataset, via_task FROM __orch__.lineage_edges;
```

```
┌────────────────┬────────────────┬────────────────┐
│  src_dataset   │  dst_dataset   │    via_task    │
├────────────────┼────────────────┼────────────────┤
│ raw_polymers   │ clean_polymers │ clean_polymers │
│ clean_polymers │ density_stats  │ density_stats  │
└────────────────┴────────────────┴────────────────┘
```

カラム単位:

```sql
SELECT src_column, dst_column, transform_kind, subtype
FROM __orch__.column_lineage
WHERE via_task = 'density_stats'
ORDER BY src_column, dst_column;
```

実機の出力:

```
┌──────────────┬──────────────┬────────────────┬────────────────┐
│  src_column  │  dst_column  │ transform_kind │    subtype     │
├──────────────┼──────────────┼────────────────┼────────────────┤
│ Density      │ density_band │ DIRECT         │ TRANSFORMATION │
│ FFV          │ avg_ffv      │ INDIRECT       │ AGGREGATION    │
│ Tg           │ avg_tg       │ INDIRECT       │ AGGREGATION    │
│ density_band │ avg_ffv      │ INDIRECT       │ GROUP_BY       │
│ density_band │ avg_tg       │ INDIRECT       │ GROUP_BY       │
│ density_band │ density_band │ INDIRECT       │ GROUP_BY       │
│ density_band │ n            │ INDIRECT       │ GROUP_BY       │
└──────────────┴──────────────┴────────────────┴────────────────┘
```

読み解くと:

- `Density → density_band` (`TRANSFORMATION`): `CASE WHEN Density < 1.0 ...` で値を変換した
- `Tg → avg_tg`、`FFV → avg_ffv` (`AGGREGATION`): `AVG()` で集計された
- `density_band → 各列` (`GROUP_BY`): グループキーとして効いている

つまり「`avg_tg`は元を辿ると`Tg`カラムを`AVG`で集計したもの」がSQLで取れる。

## 5. リネージをMermaidで可視化する

CLI から `duck-orch graph <mode>` で出す。`<mode>` は `lineage`(テーブル間) / `dag`(タスク間依存) / `combined`(両方) を切り替えられる。

```bash
duck-orch --db test.db graph lineage
```

実機の出力:

```
graph LR
    raw_polymers[(raw_polymers)] --> clean_polymers[(clean_polymers)]
    clean_polymers[(clean_polymers)] --> density_stats[(density_stats)]
```

```bash
duck-orch --db test.db graph combined
```

```
graph LR
    raw_polymers[(raw_polymers)] --> clean_polymers[(clean_polymers)]
    clean_polymers[(clean_polymers)] --> density_stats[(density_stats)]
    clean_polymers_task --> density_stats_task
```


(SQL側でも `PRAGMA orch_visualize('lineage')` で同じ出力が取れる。ただし `'lineage'` モードは実行済みのリネージから描くので、`PRAGMA orch_run` まで通した後で叩く必要がある。)

## 6. DuckLake上のデータでもOpenLineageと連携する(試作)

### そもそもDuckLakeって？

DuckLake は **DuckDBが扱える軽量なデータレイクのフォーマット**。Iceberg / Delta Lakeと同じ系統で、

- **メタデータ**(テーブル定義 / バージョン管理)は `mylake.ducklake` というDuckDB形式のファイル
- **実体データ**(中身のparquet)は`DATA_PATH`で指定した場所(ローカルディレクトリ / S3 / GCS等)

に分かれる。

```sql
INSTALL ducklake; LOAD ducklake;
ATTACH 'ducklake:mylake.ducklake' AS lake (DATA_PATH 's3://my-bucket/lake/');
USE lake;

CREATE TABLE foo AS SELECT 1;
-- メタデータは mylake.ducklake に
-- 実体parquetは s3://my-bucket/lake/foo/<uuid>.parquet に書かれる
```

なので**DuckDBから書き込んだデータ実体が S3 にある状態**を作れる。同じS3バケットをSpark / Trino / Flink等が直接読み書きすることもできる、というのがDuckLakeの売り。

### この節で何が嬉しいのか

共有レイク運用、つまり「**Sparkが s3://my-bucket/lake/ に書いて、DuckDBが s3://my-bucket/lake/ から読む**」という状態を考える。両エンジンは**同じテーブル**を扱っているので、リネージ画面でも繋がってほしい。

OpenLineageでは、このテーブルの同一性を `namespace` フィールドで表現する。Sparkの公式 OpenLineage インテグレーションは S3パスを namespace にするので、duckorch側も同じS3パスを namespace に入れれば、Marquez/DataHub上で**両エンジンのリネージが同じテーブルノードを中継して1本に繋がる**。

duckorch はattach情報の`DATA_PATH`を読み取って、namespace を自動でそのS3パスに揃える。つまり**設定を書かなくても勝手に繋がる**ようにしてある。

### 実機で確認

DuckLakeにattachして、`debug=true`で実際に出るOpenLineageイベントを観察する。今回はローカルパスでデモする(S3の認証なしでも動くため):

```sql
-- DuckLake拡張をロードしてattach
INSTALL ducklake; LOAD ducklake;
ATTACH 'ducklake:mylake.ducklake' AS lake (DATA_PATH '/tmp/lake_data');
USE lake;

-- duckorchはいつも通り
PRAGMA orch_init;
SET orch_openlineage_url='http://localhost:5000/api/v1/lineage';
SET orch_openlineage_debug=true;   -- HTTP POSTする内容をstderrに吐かせる

CREATE OR REPLACE TABLE raw_polymers AS
SELECT * FROM read_csv_auto('/tmp/poly_demo/train.csv') LIMIT 100;
PRAGMA orch_register('./tasks/');
PRAGMA orch_run;
```

実機のstderr出力（`clean_polymers` タスクの START イベント、JSON整形済み）:

```json
{
  "eventType": "START",
  "eventTime": "2026-05-04T08:55:17.378Z",
  "producer":  "https://github.com/nkwork9999/duck-orch",
  "schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json",
  "run": {
    "runId": "d886b646-ba04-4da7-bd88-76c3edf06134",
    "facets": {
      "parent": {
        "_producer":  "https://github.com/nkwork9999/duck-orch",
        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ParentRunFacet.json",
        "run": { "runId": "9a489e8f-35a2-4f99-a575-eb09c012f0c6" },
        "job": { "namespace": "duckdb", "name": "pipeline" }
      }
    }
  },
  "job":     { "namespace": "duckdb", "name": "clean_polymers", "facets": {} },
  "inputs":  [{ "namespace": "duckdb", "name": "raw_polymers" }],
  "outputs": [{ "namespace": "duckdb", "name": "clean_polymers" }]
}
```

`inputs` / `outputs` の `namespace` が `"duckdb"` になっているのは、`DATA_PATH` がローカルパスのときのフォールバック値。これを以下のようにS3へ向けると:

```sql
ATTACH 'ducklake:mylake.ducklake' AS lake (DATA_PATH 's3://my-bucket/lake/');
```

`namespace` が 自動で `"s3://my-bucket/lake/"` に切り替わる。Sparkが同じS3バケットに対してOpenLineageを設定していれば、向こうも同じnamespaceを使うので、Marquez/DataHubの画面で「Sparkが書いた raw_polymers → duckorchが作った clean_polymers」が1本のグラフに繋がる。

これに関しては試作段階で良し悪しは検証していきたい。

## 7. 全CLIサブコマンドに`--json`を生やしてAIエージェントから扱えるようにする

代表例として、`impact`(このテーブル変えると何が壊れる？)を `--json` で叩く:

```bash
duck-orch --db mydata.db impact clean_polymers --json
```

実機の出力:

```json
[{"table_name":"density_stats","via_task":"density_stats"}]
```

`clean_polymers` を変更すると `density_stats` が下流で影響を受ける、という予測が返ってきている。

### この予測が当たるか実機で検証

予測通り壊れるか試してみる。`clean_polymers.sql` から `Density` 列を意図的に外して再実行:

```sql
-- 改変版 tasks/clean_polymers.sql
CREATE OR REPLACE TABLE clean_polymers AS
SELECT id, SMILES, Tg, FFV, Tc, Rg     -- ← Density を削除
FROM raw_polymers
WHERE Tg IS NOT NULL OR FFV IS NOT NULL OR Tc IS NOT NULL;
```

```sql
PRAGMA orch_run;

SELECT task_name, status, substring(error_message, 1, 100) AS err
FROM __orch__.runs ORDER BY started_at DESC LIMIT 2;
```

実機の出力:

```
┌────────────────┬─────────┬───────────────────────────────────────────────────────────┐
│   task_name    │ status  │                            err                            │
├────────────────┼─────────┼───────────────────────────────────────────────────────────┤
│ density_stats  │ failed  │ Binder Error: Referenced column "Density" not found in... │
│ clean_polymers │ success │ NULL                                                      │
└────────────────┴─────────┴───────────────────────────────────────────────────────────┘
```

`clean_polymers` は問題なく通るが、その下流の `density_stats` が `Density` 列を見失って **`Binder Error` で落ちる**。impact の予測通り。

つまり `impact --json` は「lineageから機械的に予測した依存関係」を返しているだけだが、実際にスキーマ変更を入れると本当にその下流が壊れる、という対応関係が確認できた。

### 他のサブコマンドも同様

`status` / `lineage` / `validate` / `schedule list` も全て `--json` を付けると構造化JSONを返す。Claude Codeのようなエージェントが「このSQL変えたら何壊れる？」を聞かれたとき、内部でこのコマンドを叩いて返ってきたJSONをそのままパースして次の判断ができる。


# 雑感

データパイプラインツールは群雄割拠で、最近はdbtが抜きん出ている、というのが所感。
ただ簡単なものなら「DB自体に機能として生やせばいいのでは？」と思い試作してみた。

DuckDBの特性上、大規模データやストリーミングデータには向かないが
中小規模なローカル / エッジ領域のデータであれば、duckorchで「DuckDB起動 → `LOAD duckorch` → `PRAGMA orch_run`」の3手で済む、というのはなんとなく良いかもしれないと試作して思った。

試作段階でアラが多いと思うので個人のデータ基盤で使用しつつ改良していきたい。
