# 4. 理論III: マルコフ連鎖サンプリング

[目次](README.md) · 前へ: [3. 理論II: 変分波動関数](03-theory-wavefunction.md) · 次へ: [5. 理論IV: 確率的再構成法](05-theory-sr.md)

[第2章](02-theory-vmc-hamiltonian.md)の配置 $x$ は、メトロポリス法により
$\rho(x)\propto|\psi(x)|^2$ から生成されます。すべての更新は*局所的*です。すなわち、1個の電子がホッピングする、
逆スピンの2個の電子がサイトを交換する、または(FSZの経路では)スピンを反転させる、のいずれかです。採択された更新のたびに、
パフィアンとその逆行列は、再計算するのではなく低ランク更新の公式で更新されます。

本章では、このアルゴリズムを**乱数の規約とあわせて**説明します。固定された制御経路における `gen_rand32()` と `genrand_real2()` の乱数生成の列は
Cの挙動の一部であり、Rustでも厳密に保たれます(乱数の状態は厳密に比較し、浮動小数点値は許容誤差つきで比較します。
[11.4](11-compatibility.md#114-数値比較ポリシー)を参照してください)。

## 4.1 サンプラーのループ

1回のSRステップ(または1回のPhysCalサンプル)に対して、`VMCMakeSample` は次の手順を実行します。

1. **開始配置。** 最初の呼び出し時(`BurnFlag == 0`)には、ランダムな配置を生成します([4.2](#42-初期配置))。以降の呼び出しでは、直前のステップの最後の配置を再利用します(`copyFromBurnSample` が `eleIdx`、`eleCfg`、`eleNum` と射影カウンタを復元し、RBMカウンタは `MakeRBMCnt` で再構築されます)。これが「バーンインの持ち越し」です。
2. すべてのパフィアンと逆行列、$\ln\mathrm{IP}$ を計算します。$\ln\mathrm{IP}$ が
   有限でない場合は、新しい初期配置を生成して再計算します。
3. **外側のステップ。** 外側のステップ数は
   $$
   n_{\rm out}=\begin{cases}\texttt{NVMCWarmUp}+\texttt{NVMCSample}&\text{first call},\\ \texttt{NVMCSample}+1&\text{later calls}.\end{cases}
   $$
   です。各外側ステップでは $n_{\rm in}=\texttt{NVMCInterval}\times N_s$ 回の提案を行います
   ([4.3](#43-提案))。外側ステップ $o\ge n_{\rm out}-\texttt{NVMCSample}$ の終了時に、現在の
   配置、その射影カウンタ(およびRBMカウンタ)と $\ln\mathrm{IP}$ が
   サンプル番号 $o-(n_{\rm out}-\texttt{NVMCSample})$ として保存されます。
4. **リフレッシュ。** 直前の完全な評価以降に $N_s$ 回を超える提案が採択されるたびに
   (`nAccept > Nsite`)、パフィアン/逆行列を最初から再計算し、$\ln\mathrm{IP}$ を再評価したうえで、カウンタをリセットします。これにより、
   ランク2更新における丸め誤差の蓄積が抑えられます。
5. 次の呼び出しのために最終配置を保存し、`BurnFlag = 1` に設定します。

> **実装**
> - C: `VMCMakeSample` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:45`
> - C: `VMCMakeSample_real` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_real.c:45`
> - C: `VMCMakeSample_fsz` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_fsz.c:42`
> - C: `saveEleConfig` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:445`
> - Rust: `vmc_make_sample_with_reducer_timed` — `crates/mvmc-core/src/sampling/driver.rs:623`
> - Rust: `vmc_make_sample_real_with_reducer_timed` — `crates/mvmc-core/src/sampling/driver.rs:122`
> - Rust: `vmc_make_sample_fsz_with_reducer_timed` — `crates/mvmc-core/src/sampling/driver.rs:1175`
> - Rust: `vmc_make_sample` — `crates/mvmc-core/src/sampling/driver.rs:604`
> - 整合性: 外側ステップ数、保存サンプルの添字、および `nAccept > Nsite` のリフレッシュは `vmcmake.c:141, 309, 334-349` に従っています(Rustでは `driver.rs:271-278, 532-552, 558-562`)。リフレッシュの判定は厳密な `>` を用います。最終ステップ後の `copyToBurnSample` は `ElectronConfiguration` の「burn」バッファで再現されており、持ち越しは `counter[9] != 0` で検出されます(`driver.rs:167`)。

## 4.2 初期配置

`makeInitialSample` はランダムな配置を構築します。

1. 局在スピンを持つすべてのサイト(`LocSpn[ri] == 1`)について、組 `(mi, si)` が未使用になるまで
   `mi = gen_rand32() % Ne`、`si = (genrand_real2() < 0.5) ? 0 : 1` を繰り返し抽出し、その電子をそのサイトに配置します。
2. 各スピン $s=0,1$ と残りの各電子 $m$ について、スピン $s$ に対してサイトが空であり、かつ
   局所スピンサイトでなくなるまで `ri = gen_rand32() % Nsite` を繰り返し抽出します。
3. `eleNum` を設定し、射影カウンタを構築(`MakeProjCnt`)して、パフィアンを
   計算します。いずれかのランクが特異行列を報告した場合は繰り返します(最大100回試行し、
   それでもだめなら異常終了します)。

> **実装**
> - C: `makeInitialSample` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:359`
> - Rust: `make_initial_sample` — `crates/mvmc-core/src/sampling/initial.rs:76`
> - Rust: `make_initial_sample_normal_with_info` — `crates/mvmc-core/src/sampling/normal_initial.rs:251`
> - Rust: `make_initial_sample_fsz` — `crates/mvmc-core/src/sampling/initial.rs:273`
> - Rust: `generate_initial_fsz_configuration` — `crates/mvmc-core/src/sampling/initial.rs:198`
> - Rust: `init_loc_spn` — `crates/mvmc-core/src/sampling/projection.rs:86`
> - 整合性: 乱数の抽出順序(局在スピン、次にスピン0の電子、次にスピン1の電子)、100回の再試行の上限、およびランク間でのパフィアン状態の集団最大値(`MPI_Allreduce(..., MPI_MAX)`)が再現されています。型付き状態/協調のテストは `normal_initial.rs` にあります。

## 4.3 提案

内側の各ステップでは、まず `NExUpdatePath` から**更新の種類**を選択します
(`getUpdateType`)。

| `NExUpdatePath` | 乱数の抽出 | 結果 |
|-----------------|-------|--------|
| 0 | なし | ホッピング |
| 1 | `genrand_real2` を1回; $<0.5$ | 交換、そうでなければホッピング |
| 2、Szが保存される場合(`iFlgOrbitalGeneral == 0`) | なし | 交換 |
| 2、一般軌道で `2Sz = -1`(固定されない) | `genrand_real2` を1回 | 交換($<0.5$)または局所スピン反転 |
| 2、一般軌道で `2Sz` が固定 | なし | 交換 |
| 3(近藤型) | 1回、最初の値が $\ge0.5$ なら2回目も | ホッピング($<0.5$)、そうでなければ2回目の抽出により交換または局所スピン反転 |
| その他 | なし | 更新なし |

Cマニュアルの目安は次のとおりです。`0` はホッピングのみ、`1` は電子系向けのホッピング+交換、
`2` はスピン系向け(`NLocSpin = 2Ne`、Cのリーダーで強制されます)。

**ホッピング候補**(`makeCandidate_hopping`):

1. サイト `ri` が局在スピンを持つ間、`mi = gen_rand32() % Ne; s = (genrand_real2() < 0.5) ? 0 : 1; ri = eleIdx[mi + s*Ne]` を繰り返します。
2. `eleCfg[rj + s*Nsite] != -1` または `LocSpn[rj] == 1` の間、`rj = gen_rand32() % Nsite` を繰り返し、`Nsite*Nsite` 回を超えて試行した場合は断念します(その場合 `rejectFlag = 1`)。

**交換候補**(`makeCandidate_exchange`): 単占有のサイトが存在しない場合は、(乱数を抽出せずに)直ちに棄却します。そうでなければ、上と同様に `(mi, s, ri)` を、`ri` の逆スピン側の
サイトが空になるまで繰り返し、次にスピン $t=1-s$ について `rj = eleIdx[mj + t*Ne]` に対して同じ条件が成り立つまで
`mj = gen_rand32() % Ne` を繰り返します。この更新は2個の電子を入れ替えます。すなわち、電子
`mi`($s$)が $r_i\to r_j$ にホッピングし、電子 `mj`($t$)が $r_j\to r_i$ にホッピングします。

**局所スピン反転**(FSZのみ): `makeCandidate_LocalSpinFlip_localspin` と
`makeCandidate_LocalSpinFlip_conduction` は、それぞれ局在スピンサイト上の電子、または
伝導電子のスピンラベルを反転させます。

`rejectFlag` が設定された候補は、メトロポリスの乱数抽出の**前**にループを `continue` させるため、`genrand_real2` は消費されません。それ以外のすべての提案では、重みが
有限でない場合やゼロの場合でも、採択判定でちょうど1回 `genrand_real2` が消費されます。

> **実装**
> - C: `getUpdateType` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:606`
> - C: `makeCandidate_hopping` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:515`
> - C: `makeCandidate_exchange` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:548`
> - C: `updateEleConfig` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:585`
> - C: `makeCandidate_hopping_fsz` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_fsz.c:594`
> - C: `makeCandidate_exchange_fsz` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_fsz.c:670`
> - C: `makeCandidate_LocalSpinFlip_localspin` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_fsz.c:728`
> - C: `makeCandidate_LocalSpinFlip_conduction` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_fsz.c:756`
> - Rust: `get_update_type` — `crates/mvmc-core/src/sampling/candidate.rs:306`
> - Rust: `make_candidate_hopping` — `crates/mvmc-core/src/sampling/candidate.rs:136`
> - Rust: `make_candidate_exchange` — `crates/mvmc-core/src/sampling/candidate.rs:216`
> - Rust: `make_candidate_hopping_fsz` — `crates/mvmc-core/src/sampling/candidate.rs:373`
> - Rust: `make_candidate_exchange_fsz` — `crates/mvmc-core/src/sampling/candidate.rs:606`
> - Rust: `make_candidate_local_spin_flip_localspin` — `crates/mvmc-core/src/sampling/candidate.rs:555`
> - Rust: `make_candidate_local_spin_flip_conduction` — `crates/mvmc-core/src/sampling/candidate.rs:480`
> - Rust: `update_ele_config` — `crates/mvmc-core/src/sampling/projection.rs:345`
> - Rust: `revert_ele_config` — `crates/mvmc-core/src/sampling/projection.rs:368`
> - 整合性: 乱数の抽出回数と順序は厳密に一致します。`gen_rand32() % n`(`Sfmt19937Rng::gen_rand32`、`crates/sfmt19937/src/lib.rs:138`)と `genrand_real2`(`crates/sfmt19937/src/lib.rs:182`)は、棄却された更新や再試行ループでの抽出も含めて、同じ順序で消費されます。テスト `rejected_candidate_consumes_no_rng_and_mutates_nothing`(`one_move.rs:640`)が、「`rejectFlag` では乱数を抽出しない」という規則を固定しています。

## 4.4 採択判定

電子 $m$ のホッピング($r_i\to r_j$)で新しいカウンタを $c'$ とすると、メトロポリスの重みは
振幅比の2乗

$$
w=\Big|\frac{\psi(x')}{\psi(x)}\Big|^2
=\exp\Big\{2\operatorname{Re}\big[\Delta\ln P+\Delta\ln\mathcal N_{\rm RBM}+\ln\mathrm{IP}(x')-\ln\mathrm{IP}(x)\big]\Big\},
$$

であり、$\Delta\ln P=\sum_k\operatorname{Re}\alpha_k\,(c'_k-c_k)$(`LogProjRatio`)です。更新は
$r=\texttt{genrand\_real2()}\in[0,1)$ に対して $w>r$ のときに限り採択されます。有限でない $w$ は $-1$ に置き換えられます(常に棄却されます)。
採択された場合は、パフィアン/逆行列、射影カウンタ、RBMカウンタ、
$\ln\mathrm{IP}$ が更新されます。棄却された場合は占有配列が元に戻され(`revertEleConfig`)、それ以外は何も変化しません。

> **実装**
> - C: `VMCMakeSample` (acceptance block `w = exp(2.0*(creal(x+logIpNew-logIpOld)))`) — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:194`
> - Rust: `metropolis_weight` — `crates/mvmc-core/src/sampling/metropolis.rs:41`
> - Rust: `metropolis_decision` — `crates/mvmc-core/src/sampling/metropolis.rs:59`
> - Rust: `attempt_hopping_move` — `crates/mvmc-core/src/sampling/one_move.rs:199`
> - Rust: `attempt_exchange_move` — `crates/mvmc-core/src/sampling/one_move.rs:282`
> - 整合性: Cは `x = LogProjRatio(...)`、`x += LogRBMRatio(...)`(複素数)を形成し、その後 `exp(2.0*(creal(x + logIpNew - logIpOld)))` を計算します。Rustは `2.0 * ((proj + rbm.re + ip_new.re) - ip_old.re)` を評価し、同じ実部の和を同じ順序で取りますが、libmの `exp` ではなく `julia_exp::exp` を用います。有限でない重みには `-1.0` が代入され、乱数は常に消費されます(`metropolis_decision` は比較の前に抽出します)。libmとJuliaの `exp` の差は、最下位ビットの差が採択を反転させうる数少ない箇所の一つです。これは想定された事象であり、[11.4](11-compatibility.md#114-数値比較ポリシー)で扱われます。

## 4.5 パフィアン比と逆行列の更新

$W=\texttt{InvM}=X^{-1}$([3.2](03-theory-wavefunction.md#32-パフィアンペア積部分))、
$a$ を移動した電子の添字、$u_j=F(r'_a,r_j)$ を*新しい*占有スピン軌道で評価した
スレーター表の新しい行とします($r_j$ はすべての電子の
現在位置、$r'_a$ は電子 $a$ の新しい位置です)。

**1電子更新(ホッピング)。** 歪対称行列の行/列 $a$ を置き換えると、パフィアンは次のように
変化します。

$$
\frac{\mathrm{Pf}\,X'}{\mathrm{Pf}\,X}=-\sum_{j}W_{aj}\,u_j ,
$$

(`CalculateNewPfM2`、セクター $q$ ごとに1つの値。$j\le N_e$ の和はスピン軌道 $r_j$ を、
残りは $r_j+N_s$ を用います)。更新が採択された場合、
`updateMAll_child` は $v=Wu$(`vec1`)、
$\mathrm{Pf}\,X\leftarrow-v_a\mathrm{Pf}\,X$、$s_i=-W_{ai}/v_a$(`vec2`)を用いてランク2更新を行います。

$$
W_{ij}\leftarrow W_{ij}+v_is_j-v_js_i,\qquad
W_{ia}\leftarrow W_{ia}-s_i,\qquad W_{aj}\leftarrow W_{aj}+s_j .
$$

**2電子更新(交換)。** 電子 $a,b$ に対し、行を $u_i=F(r'_a,r_i)$、
$v_i=F(r'_b,r_i)$ とします(どちらも新しい位置で評価するので、$u_b=F(r'_a,r'_b)$、$v_a=F(r'_b,r'_a)$ です)。

$$
\frac{\mathrm{Pf}\,X'}{\mathrm{Pf}\,X}
= W_{ab}\,v_a + W_{ab}\,(v^{\!T}Wu)+p_aq_b-p_bq_a,
\qquad
p_c=\sum_iW_{ci}u_i,\ \ q_c=\sum_iW_{ci}v_i\ (c=a,b),
$$

(`calculateNewPfMTwo_child_fcmp`; $v^{T}Wu=\sum_iv_i\sum_jW_{ij}u_j$)。採択された更新
(`updateMAllTwo_child_fcmp`)は、対応するランク4のウッドベリー型更新であり、
$p,q$ から構成される6個のスカラー係数 $a,b,c,d,e,f$、$2\times2$ の
行列式 $\det=ad-bc-ef$、およびベクトル $s=\det^{-1}W_{a\cdot}$、
$t=\det^{-1}W_{b\cdot}$ を用います(`update_two_complex` を参照してください)。

同じ更新の仕組みは、[2.5](02-theory-vmc-hamiltonian.md#25-グリーン関数の比)のグリーン関数の比の計算でも、配置の*コピー*に対して用いられます。Lanczos法の経路の `calHCA`/`calHCACA` は、保存されている
`InvM`/`PfM` を一時的に更新したのち復元します(`copyMAll`)。

> **実装**
> - C: `CalculateNewPfM2` — `extern/mVMC-1.3.0/src/mVMC/pfupdate.c:78`
> - C: `UpdateMAll` — `extern/mVMC-1.3.0/src/mVMC/pfupdate.c:119`
> - C: `updateMAll_child` — `extern/mVMC-1.3.0/src/mVMC/pfupdate.c:143`
> - C: `CalculateNewPfMTwo2_fcmp` — `extern/mVMC-1.3.0/src/mVMC/pfupdate_two_fcmp.c:73`
> - C: `calculateNewPfMTwo_child_fcmp` — `extern/mVMC-1.3.0/src/mVMC/pfupdate_two_fcmp.c:107`
> - C: `UpdateMAllTwo_fcmp` — `extern/mVMC-1.3.0/src/mVMC/pfupdate_two_fcmp.c:189`
> - C: `updateMAllTwo_child_fcmp` — `extern/mVMC-1.3.0/src/mVMC/pfupdate_two_fcmp.c:217`
> - C: `UpdateMAll_fsz` — `extern/mVMC-1.3.0/src/mVMC/pfupdate_fsz.c:114`
> - Rust: `calculate_new_pf_m2_complex_flat` — `crates/mvmc-core/src/sampling/updates.rs:26`
> - Rust: `calculate_new_pf_m2_real_flat` — `crates/mvmc-core/src/sampling/updates.rs:64`
> - Rust: `update_m_all_complex_flat` — `crates/mvmc-core/src/sampling/updates.rs:354`
> - Rust: `update_one_complex` — `crates/mvmc-core/src/sampling/updates.rs:821`
> - Rust: `calculate_new_pf_m_two2_complex_flat` — `crates/mvmc-core/src/sampling/updates.rs:166`
> - Rust: `two_ratio_complex` — `crates/mvmc-core/src/sampling/updates.rs:1085`
> - Rust: `update_m_all_two_complex_flat` — `crates/mvmc-core/src/sampling/updates.rs:519`
> - Rust: `update_two_complex` — `crates/mvmc-core/src/sampling/updates.rs:933`
> - Rust: `update_m_all_two_real_flat` — `crates/mvmc-core/src/sampling/updates.rs:570`
> - 整合性: `updateMAll_child` 内のCの `invM` の累積順序は、$j$ にわたる `vec1[msi] += -invM_j[msi] * sltE_aj` の後に $i$ を外側とするランク2更新です。Rustのカーネルはこのループの入れ子とスカラーの順序を保っており、結果は丸め誤差の範囲で一致します。**既知の癖(観測):** 2電子更新において、Cは4つのすべての変種で `rsbOld = raOld + t*Nsite` を定義しています(`rbOld` ではなく `raOld` を用います)(`pfupdate_two_fcmp.c:227`、`pfupdate_two_real.c:227`、およびFSZのファイル)。Rustの*実数*版の更新はこれを意図的に再現していますが(`update_m_all_two_real_flat`、`updates.rs:565`、`rsb_old = ra_old + ...` は `updates.rs:588`)、Rustの*複素数の通常*版の更新は `rb_old` を用います(`updates.rs:537`)。[11.5](11-compatibility.md#115-未解決の観測事項)を参照してください。

## 4.6 サンプラー内の並列化

`NSplitSize = 1` の場合、各MPIランクはRNGシード
`RndSeed + rank` で独自のチェーンを実行し(C: `init_gen_rand(RndSeed + group1)`、
`group1 = rank0/NSplitSize`、`vmcmain.c:257`)、セクターの和はすべてローカルです。
`NSplitSize > 1` では、1つのグループのランクが1つのチェーンを共有します。セクター範囲
`[qpStart, qpEnd)` はそれらの間で分割され(`SplitLoop`)、$\mathrm{IP}$ は `CalculateIP_fcmp` 内でグループのコミュニケータ上の
`MPI_Allreduce` によって合計されます。その他の乱数抽出は
グループの全ランクで同一に行われます。[8.4](08-running.md#84-mpiとグループ実行)を参照してください。

> **実装**
> - C: `SplitLoop` — `extern/mVMC-1.3.0/src/mVMC/splitloop.c:31`
> - C: `CalculateIP_fcmp` (group `MPI_Allreduce`) — `extern/mVMC-1.3.0/src/mVMC/qp.c:110`
> - Rust: `partition_range` — `crates/mvmc-core/src/parallel.rs:88`
> - Rust: `assign_group` — `crates/mvmc-core/src/parallel.rs:67`
> - Rust: `resolve_rnd_seed` — `crates/mvmc-core/src/run.rs:1840`
> - 整合性: `partition_range` は、「余りを最後のランクに割り当てる」規則と少量の仕事量の場合の分岐を含めて `SplitLoop` を再現します。`resolve_rnd_seed` は基本シードにグループ番号を `i64` のラップアラウンド演算で加え、その結果が `u32` に収まることを要求します(`seeded_rng`、`run.rs:1799`)。負の `RndSeed` では、出力ルートで読み取った1つの時計の値を用い、ブロードキャストします(Juliaのライフサイクル)。
