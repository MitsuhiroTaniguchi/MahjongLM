# 天鳳牌譜メモ

更新日: 2026-03-12

目的:

- 牌譜 parser / tokenizer 実装で見落としやすい点を固定する
- 元記事の読み直しコストを減らす
- この repository の実装判断も合わせて残す

## 参照元

- 牌譜形式（json）
  - [電脳麻将の牌譜形式](https://blog.kobalab.net/entry/20151228/1451228689)
- 牌譜形式中の中国語一覧
  - [電脳麻将のプログラム中の中国語一覧](https://blog.kobalab.net/entry/20170722/1500688645)
- 天鳳牌譜解析シリーズ
  - [天鳳の牌譜形式を解析する(1)](https://blog.kobalab.net/entry/20170225/1488036549)
  - [天鳳の牌譜形式を解析する(2)](https://blog.kobalab.net/entry/20170228/1488294993)
  - [天鳳の牌譜形式を解析する(3)](https://blog.kobalab.net/entry/20170312/1489315432)
  - [天鳳の牌譜形式を解析する(4)](https://blog.kobalab.net/entry/20170720/1500479235)

## 実装で重要な前提

### 1. ログには「実行行動」しか基本出ない

- 鳴けたが鳴かなかった
- 立直できたがしなかった
- ロンできたが見送った

この種の未実行合法手は、牌譜 replay で復元するしかない。

この repository ではそれを `opt_*` / `pass_*` として明示化している。

### 2. 牌譜は順序依存

局内イベントは時系列 replay が前提。

特に壊れやすいのは:

- `fulou`
- `gang`
- `gangzimo`
- `kaigang`
- `hule`
- `pingju`

槓後ドラ、嶺上、槍槓、北抜き補充は event 順で扱わないと破綻する。

### 3. `m` デコードは手牌更新の中心

`N` タグの `m` はビット符号化なので、`chi / pon / daiminkan / kakan / ankan` を正しく分岐する必要がある。

ここを雑に扱うと:

- concealed count
- meld 状態
- furiten
- 次の合法手

が全部ずれる。

### 4. 実牌譜優先

天鳳実牌譜が反例になるなら、一般論より実牌譜を優先する。

この repository では実際に次のようなケースで unit test より実牌譜を優先して仕様を固めた。

- `penuki` 後の `ron`
- `qianggang` 文脈
- `temporary_furiten` の解除タイミング
- `haidi/houtei` 判定

## この repository での具体的な設計

### decision token

- self: `opt_self_*`, `take_self_*`, `pass_self_*`
- react: `opt_react_*`, `take_react_*`, `pass_react_*`

これが tokenizer の中心。

### multiview

- complete view 1 本
- imperfect view 各 seat 1 本

imperfect view では:

- hidden 初期手牌は `hidden_haipai_{seat}` 1 トークン
- hidden draw は `draw_{seat}_hidden`
- viewer 以外の self decision は出さない
- reaction decision は viewer 自身のものだけ残す

### `qijia`

最終順位復元と `view_imperfect_{player}` の `player` 計算に使う。

同点最終順位を正しく決めるには `game.qijia` が必要。

## 実装チェックリスト

- `qipai` で全員手牌・点数・ドラ表示・局情報を初期化
- `zimo` / `dapai` ごとに手牌を更新
- `dapai` 直後に他家 `ron/chi/pon/minkan` を計算
- `fulou` / `gang` は `m` デコード結果で消費牌を決定
- 槓後の嶺上ツモとドラ公開順を event 順で反映
- `penuki` 後の replacement draw を通常の draw state に接続
- `hule` / `pingju` で点棒と順位を確定
- 実牌譜 failure は fixture 化して回帰テストへ入れる

## 注意

- kobalab 記事は実装の出発点として有用だが、運用上の例外や後年の牌譜差異までは保証しない
- Wikipedia や点数表は補助にはなるが、実データと食い違うなら盲信しない
- tokenizer は latent decision を補うので、単なる parser より state の整合性要求が高い
