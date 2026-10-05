# 2. 理論I: 変分モンテカルロ法とハミルトニアン

[目次](README.md) · 前へ: [1. 概要とインストール](01-overview-install.md) · 次へ: [3. 理論II: 変分波動関数](03-theory-wavefunction.md)

この章では、記法を定め、他のすべての章で用いるモンテカルロ推定量を導出し、Expert モードで定義できるハミルトニアンの項を列挙し、各項が波動関数振幅の比(*ローカルエネルギー*と*グリーン関数の比*)としてどのように評価されるかを説明します。

## 2.1 記法

| 記号 | 意味 | コード上の名前 |
|--------|---------|-------------|
| $N_s$ | サイト数 | `Nsite` (`NSite`) |
| $N_e$ | sz 保存(「通常」)経路におけるスピンあたりの電子数。$N = 2N_e$ はパフィアン行列のサイズ | C 入力では `Ne`/`Nelectron`(Rust 入力では `NElec`/`Nelec`、フィールド `nelec`)、`Nsize = 2*Ne` |
| $N_{\rm QP}$ | 量子射影セクターの数 (`NQPFull`) | `NQPFull` |
| $x$ | 実空間配置 | `eleIdx`, `eleCfg`, `eleNum` |
| $\psi(x)=\langle x\vert\psi\rangle$ | 試行状態の振幅 | `ip`, `logIp` |
| $\alpha_k$ | $k$ 番目の変分パラメータ(複素数) | `Para[k]` |

`Ne` は `modpara.def` から、直接(`Nelec`/`Nelectron`)読み込まれるか、`Ncond` から導出されます。`Ncond` が与えられた場合、$N_e=(N_{\rm locspin}+N_{\rm cond})/2$ であり、`Ncond` は偶数でなければなりません(`readdef.c`, *CalcNCond*, 589-595 行。Rust は `crates/mvmc-expert-parsers/src/lib.rs:308-313`)。局在スピンは電子として数えられます。ハイゼンベルク模型では `NLocSpin = Nsite`, `Ncond = 0` となります。

配置 $x$ は基底状態

$$
|x\rangle=\prod_{n=1}^{N_e}c^\dagger_{r_{n\uparrow}\uparrow}\prod_{n=1}^{N_e}c^\dagger_{r_{n\downarrow}\downarrow}|0\rangle ,
$$

です。ここで $r_{n\sigma}$ はスピン $\sigma$ の $n$ 番目の電子のサイトです
([Cマニュアル, *アルゴリズム*](../../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst))。
コードはこれを 3 つの配列として保持します(C の名前。Rust ではスネークケースの同じ名前を使います)。

- `eleIdx[n + s*Ne]` – スピン $s$(0 = アップ、1 = ダウン)の $n$ 番目の電子のサイト,
- `eleCfg[r + s*Nsite]` – サイト $r$ にあるスピン $s$ の電子のインデックス $n$。なければ $-1$,
- `eleNum[r + s*Nsite]` – 占有数(0 または 1)。

FSZ/一般軌道の経路では、電子は明示的なスピンラベル(`eleSpn`)を持ち、$N_\uparrow\neq N_\downarrow$ が許されます。

## 2.2 変分モンテカルロ法の推定量

試行状態 $|\psi\rangle$ に対する演算子 $A$ の期待値は

$$
\langle A\rangle=\frac{\langle\psi|A|\psi\rangle}{\langle\psi|\psi\rangle}
=\sum_x\rho(x)\,\frac{\langle\psi|A|x\rangle}{\langle\psi|x\rangle},
\qquad
\rho(x)=\frac{|\langle x|\psi\rangle|^2}{\langle\psi|\psi\rangle}.
$$

です。[第 4 章](04-theory-sampling.md)のマルコフ連鎖は $x\sim\rho(x)$ をサンプリングするので、すべてのサンプルが同じ重みを持ちます。C コードはこれを明示しており(`w = 1.0`, `vmccal.c:153`。コメントアウトされた再重み付けの式は使われません)、Rust への移植でも重み変数を 1 に保っています。ローカル推定量

$$
F(x,A)=\frac{\langle\psi|A|x\rangle}{\langle\psi|x\rangle},
$$

を用いて、コードは保存された $N_{\rm smp}$ = `NVMCSample` 個の配置にわたって

$$
\langle H\rangle\simeq\frac{1}{W}\sum_x w\,F(x,H),\qquad
\langle H^2\rangle_{\rm est}\equiv\frac{1}{W}\sum_x w\,\overline{F(x,H)}\,F(x,H),
\qquad W=\sum_x w .
$$

を累積します。

`Etot2` として保存される 2 次モーメント(`zvo_out.dat` で $\langle H^2\rangle$ と呼ばれる列)は $\langle F^\dagger F\rangle=\langle\lvert E_{\rm loc}\rvert^2\rangle$ であり、$\langle F(x,H^2)\rangle$ では**ない**ことに注意してください。C マニュアルはその理由を説明しています。積の形のほうが数値的に安定で、有限個のサンプルに対して非負の分散を与えるためです
([Cマニュアル, *パワー Lanczos 法*](../../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst))。
`zvo_out.dat` に書き出される相対分散は

$$
\text{variance}=\operatorname{Re}\frac{\langle H^2\rangle_{\rm est}-\langle H\rangle^2}{\langle H\rangle^2}.
$$

です。独立なマルコフ連鎖(MPI ランク、`NSplitSize` グループ)にわたる平均では、$W$、$\sum wF$、$\sum w|F|^2$ および SR の累積量を和してから $W$ で割ります
([第 8 章](08-running.md))。

> **実装**
> - C: `VMCMainCal` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:82`
> - C: `WeightAverageWE` — `extern/mVMC-1.3.0/src/mVMC/average.c:41`
> - C: `outputData` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:640`
> - Rust: `accumulate_observables_local` — `crates/mvmc-core/src/run.rs:2716`
> - Rust: `reduce_accumulators` — `crates/mvmc-core/src/run.rs:1638`
> - Rust: `weight_average_we` — `crates/mvmc-core/src/average.rs:18`
> - Rust: `output_data` — `crates/mvmc-core/src/io.rs:89`
> - 整合性: `Etot2 += w * conj(e) * e` (`vmccal.c:191`) は `run.rs:2771` の `etot2 += w * e.conj() * e` に対応し、積の順序は同じです。`output_data` では、相対分散は $|\langle H\rangle|>10^{-14}$ のときにのみ計算され、それ以外は `0.0` として書き出されます(Julia のガード)。C は無条件に割ります。最適化出力の複素数除算には `julia_complex::divide` を使い、PhysCal 出力(`output_phys_data`, `io.rs:161`)には C99 方式の `c_complex::divide` を使います。
> - 整合性: パフィアンのセットアップに失敗したサンプルや、エネルギーが有限でないサンプルは、どちらの実装でもスキップされます(C は警告を出力して `continue` し、Rust は `continue` します)。

## 2.3 ハミルトニアンの項

Expert モードは、ハミルトニアンを 7 つの族の和として定義します
([Cマニュアル, *Expert モードの入力ファイル*](../../../extern/mVMC-1.3.0/doc/en/source/expert.rst), 24-67 行):

$$
\begin{aligned}
\mathcal H_T&=-\sum_{ij}\sum_{\sigma_1\sigma_2}t_{ij\sigma_1\sigma_2}\,c^\dagger_{i\sigma_1}c_{j\sigma_2}, &
\mathcal H_U&=\sum_i U_i\,n_{i\uparrow}n_{i\downarrow},\\
\mathcal H_V&=\sum_{ij}V_{ij}\,n_in_j, &
\mathcal H_H&=-\sum_{ij}J^{\rm Hund}_{ij}\,(n_{i\uparrow}n_{j\uparrow}+n_{i\downarrow}n_{j\downarrow}),\\
\mathcal H_E&=\sum_{ij}J^{\rm Ex}_{ij}\,(c^\dagger_{i\uparrow}c_{j\uparrow}c^\dagger_{j\downarrow}c_{i\downarrow}+c^\dagger_{i\downarrow}c_{j\downarrow}c^\dagger_{j\uparrow}c_{i\uparrow}), &
\mathcal H_P&=\sum_{ij}J^{\rm Pair}_{ij}\,c^\dagger_{i\uparrow}c_{j\uparrow}c^\dagger_{i\downarrow}c_{j\downarrow},\\
\mathcal H_I&=\sum_{ijkl}\sum_{\sigma_1\ldots\sigma_4}I_{ijkl\sigma_1\sigma_2\sigma_3\sigma_4}\,c^\dagger_{i\sigma_1}c_{j\sigma_2}c^\dagger_{k\sigma_3}c_{l\sigma_4},
\end{aligned}
$$

ここで $n_{i\sigma}=c^\dagger_{i\sigma}c_{i\sigma}$、$n_i=n_{i\uparrow}+n_{i\downarrow}$ です。
定義ファイルの各行は和の**ひとつの**項に寄与します。プログラムは $(i,j)$ と $(j,i)$ を対称化しません(保存された行にわたってループする `CalculateHamiltonian` を参照)。唯一の例外は `PairHop` で、その行は読み込み時に同じ係数で $(i,j)$ と $(j,i)$ の両方に展開されます
(`ReadPairHopValue`, `NPairHopping = 2*NPairHop`)。

| 族 | ネームリストのキーワード / ファイル | 演算子の評価方法 | Rust のパース済みフィールド |
|--------|------------------------|-----------------------|-------------------|
| $\mathcal H_T$ | `Trans` (`trans.def`) | $-t\,G^{(1)}$ | `transfer_terms` |
| $\mathcal H_U$ | `CoulombIntra` | 対角 $U_i n_{i\uparrow}n_{i\downarrow}$ | `coulomb_intra_terms` |
| $\mathcal H_V$ | `CoulombInter` | 対角 $V_{ij}(n_{i\uparrow}+n_{i\downarrow})(n_{j\uparrow}+n_{j\downarrow})$ | `coulomb_inter_terms` |
| $\mathcal H_H$ | `Hund` | 対角 $-J(n_{i\uparrow}n_{j\uparrow}+n_{i\downarrow}n_{j\downarrow})$ | `hund_terms` |
| $\mathcal H_E$ | `Exchange` | $J\,[G^{(2)}_{\uparrow\downarrow}+G^{(2)}_{\downarrow\uparrow}]$ | `exchange_terms` |
| $\mathcal H_P$ | `PairHop` | $J\,G^{(2)}(i,j,i,j;\uparrow,\downarrow)$ | `pair_hop_terms` |
| $\mathcal H_I$ | `InterAll` | $I\,G^{(2)}$ | `inter_all_terms`(別途管理。評価以外はここでは文書化しません) |

対角成分の族($U$, $V$, Hund)は占有数だけを必要とします。それ以外は配置を変化させ、振幅の比として評価されます。[2.5](#25-グリーン関数の比)を参照してください。スピン模型は $S=1/2$ に対してボゴリューボフ表現
$S_z=\tfrac12(n_\uparrow-n_\downarrow)$, $S^+=c^\dagger_\uparrow c_\downarrow$,
$S^-=c^\dagger_\downarrow c_\uparrow$ を用い(サポートされるのは $S=1/2$ のみ)、局在スピンを持つサイトは `locspn.def` で指定されます。サンプラーはそれらの電子をホッピングで動かすことはありません([第 4 章](04-theory-sampling.md))。

## 2.4 ローカルエネルギー

$H$ を[2.2](#22-変分モンテカルロ法の推定量)の推定量に代入すると、

$$
E_{\rm loc}(x)=F(x,H)=\sum_{x'}H_{x'x}\,\overline{\frac{\psi(x')}{\psi(x)}} ,
$$

となります。したがって、対角項は $x$ での値を与え、非対角項 $c^\dagger_ic_j\cdots$ はその係数に、移動後の配置と $x$ との間の振幅比の複素共役を掛けたものを与えます。コードは次の順序で評価します。

$$
E_{\rm loc}=\sum_i U_i n_{i\uparrow}n_{i\downarrow}
+\sum V_{ij}n_in_j-\sum J^{\rm H}_{ij}(\ldots)
-\sum t_{ij}G^{(1)}_{ij\sigma}
+\sum J^{\rm P}_{ij}G^{(2)}_{\rm pair}
+\sum J^{\rm E}_{ij}(G^{(2)}_{\uparrow\downarrow}+G^{(2)}_{\downarrow\uparrow})
+\sum I\,G^{(2)} .
$$

> **実装**
> - C: `CalculateHamiltonian` — `extern/mVMC-1.3.0/src/mVMC/calham.c:59`
> - C: `CalculateHamiltonian0` (diagonal part, used by Lanczos) — `extern/mVMC-1.3.0/src/mVMC/calham.c:198`
> - C: `CalculateHamiltonian1` (transfer part, Lanczos) — `extern/mVMC-1.3.0/src/mVMC/calham.c:244`
> - C: `CalculateHamiltonian2` (two-body part, Lanczos) — `extern/mVMC-1.3.0/src/mVMC/calham.c:299`
> - C: `CalculateHamiltonian_fsz` — `extern/mVMC-1.3.0/src/mVMC/calham_fsz.c:49`
> - Rust: `calculate_local_energy_timed` — `crates/mvmc-core/src/observables.rs:2881`
> - Rust: `calculate_local_energy` — `crates/mvmc-core/src/observables.rs:1913`
> - Rust: `calculate_hamiltonian_diagonal` — `crates/mvmc-core/src/observables.rs:551`
> - Rust: `calculate_local_energy_fsz` — `crates/mvmc-core/src/observables.rs:1246`
> - 整合性: 累積の順序は、CoulombIntra, CoulombInter, Hund(マイナス符号), Transfer(マイナス符号), PairHop, Exchange(`tmp = G(0,1) + G(1,0)`、その後 `J*tmp`)、そしてファイル順の InterAll です。C コードはこれらのループを `schedule(dynamic)` の OpenMP `reduction(+:e)` の下で実行するため、項にわたる C の総和順序はスレッド数によって固定されません。Rust はファイル順に逐次的に総和します。*実*の遷移係数の場合、Rust は Transfer セクション全体を別の和(`transfer_energy`)に累積し、あとで対角部分に加えます(`observables.rs:2807-2889`)。2 つの和を結合すると SR 勾配が丸め誤差レベルで変わるため、この順序は維持しなければなりません。
> - 整合性: 実波動関数では、C は `CalculateHamiltonian_real(creal(ip), ...)` (`calham_real.c`) を呼び出し、その累積量は `double` です。したがって Rust の実数経路は `InterAll` 係数の虚部を捨てます(`observables.rs:2984` のコメント。C の `calham_real.c:52` は `double myEnergy` を宣言し、`calham_real.c:136` は `creal(ParaTransfer[idx])` を使うため、実数経路では虚部を持つ `Trans` 係数も捨てられます)。
> - 整合性: サイトインデックスが $0\ldots N_s-1$ の範囲外にある項は Rust ではスキップされます(`InterAll` については、それより前に検証で拒否されます。[7.5](07-input-files.md#75-サポートされる入力と拒否される入力)を参照)。

## 2.5 グリーン関数の比

1 体および 2 体の*ローカルグリーン関数*は

$$
G^{(1)}_{ij\sigma}(x)=\frac{\langle\psi|c^\dagger_{i\sigma}c_{j\sigma}|x\rangle}{\langle\psi|x\rangle},\qquad
G^{(2)}_{ijkl\sigma\tau}(x)=\frac{\langle\psi|c^\dagger_{i\sigma}c_{j\sigma}c^\dagger_{k\tau}c_{l\tau}|x\rangle}{\langle\psi|x\rangle}.
$$

です。$x'_1=c^\dagger_{i\sigma}c_{j\sigma}|x\rangle$ を、サイト $j$ の電子を空きサイト $i$ へ移して得られる配置とすると、
$G^{(1)}=\overline{\psi(x'_1)/\psi(x)}$ であり、コードは

$$
\frac{\psi(x')}{\psi(x)}=
\underbrace{e^{\Delta\ln P}}_{\text{ProjRatio}}\;
\underbrace{R_{\rm RBM}}_{\text{RBMRatio}}\;
\frac{\mathrm{IP}(x')}{\mathrm{IP}(x)},\qquad
\mathrm{IP}(x)=\sum_q w_q\,\mathrm{Pf}\,X_q(x),
$$

を計算します。ここで $\mathrm{IP}$ は、[3.2](03-theory-wavefunction.md#32-パフィアンペア積部分)および[3.4](03-theory-wavefunction.md#34-量子数射影)で定義された射影パフィアン内積です。`GreenFunc1` は移動する電子を設定し、射影カウンタを更新し、保存されている逆行列を変更せずに 1 電子パフィアン更新([4.5](04-theory-sampling.md#45-パフィアン比と逆行列の更新))で $\mathrm{IP}(x')$ を評価し、配置を元に戻して `conj(z/ip)` を返します。

振幅を評価する前に実装されている選択則(`GreenFunc1`): $G^{(1)}_{ii\sigma}=n_{i\sigma}$ であり、サイト $i$ がすでに占有されているか、サイト $j$ が空であれば($i\ne j$)$G^{(1)}_{ij\sigma}=0$ です。

$G^{(2)}$ では、インデックスが一致する場合は $G^{(1)}$ または占有数に帰着します(`GreenFunc2` に列挙された数演算子の恒等式。たとえば $k=l$ の $c^\dagger_ic_jc^\dagger_kc_l$ は、$\sigma=\tau$ で $n_k$ が占有されているとき $G^{(1)}_{ij}n_k$ です)。一般の場合は、まず $c^\dagger_{k\tau}c_{l\tau}$ を作用させ($\tau$ 電子を $l$ から $k$ へ移動)、次に $c^\dagger_{i\sigma}c_{j\sigma}$ を作用させます($\sigma$ 電子を $j$ から $i$ へ移動)。2 回移動した配置のパフィアンは、[4.5](04-theory-sampling.md#45-パフィアン比と逆行列の更新)の 2 電子更新で得られます。

> **実装**
> - C: `GreenFunc1` — `extern/mVMC-1.3.0/src/mVMC/locgrn.c:41`
> - C: `GreenFunc2` — `extern/mVMC-1.3.0/src/mVMC/locgrn.c:86`
> - C: `GreenFunc1_fsz` — `extern/mVMC-1.3.0/src/mVMC/locgrn_fsz.c:29`
> - C: `CalculateIP_fcmp` — `extern/mVMC-1.3.0/src/mVMC/qp.c:110`
> - Rust: `green_func1` — `crates/mvmc-core/src/observables.rs:1463`
> - Rust: `green_func1_impl` — `crates/mvmc-core/src/observables.rs:1742`
> - Rust: `green_func2` — `crates/mvmc-core/src/observables.rs:626`
> - Rust: `green_func2_impl` — `crates/mvmc-core/src/observables.rs:731`
> - Rust: `green_func2_fsz` — `crates/mvmc-core/src/observables/fsz_green.rs:20`
> - 整合性: 最終的な振幅比は複素共役を取ります(`conj(z/ip)`)。Rust では `divide(proj_ratio * new_ip, ip).conj()` が、C カーネルのインスタンス化(`C_KERNEL = true`)では `c_complex::divide`(C99 の意味論)を、それ以外では `julia_complex::divide` を選択します(`observables.rs:1874-1877`)。`ProjRatio` は射影指数の*実部*の $\exp$ のみを使います。C が Gutzwiller/Jastrow/DH パラメータを実数と宣言しているためです(`projection.c:41-56`, "we assume gutzwiller and jastrow is real")。
> - 整合性: C の `GreenFunc1` は RBM 以外の分岐で `UpdateProjCnt` を 2 回呼び出します(`locgrn.c:68`。同一引数での冗長な重複)。これは結果を変えません。
> - 整合性: 遷移係数のキャッシュ(`refresh_transfer_cache`, `observables.rs:1533`)と、オプションのマルチスレッド遷移ループ(`MVMC_RS_INNER_THREADS`)が Rust の実数経路に存在します。どちらも各項について逐次的な項順序の結果を保ちます(総和順序は変わりません)。

## 2.6 対角成分のみで求まる量

`Sz` の累積量(`sztot`, `sztot2`, `zvo_out.dat` の最後の 2 列)は、`calculate_sz` (`crates/mvmc-core/src/observables.rs:192`) により `eleNum` から計算されます。sz 保存経路では、これらは恒等的にゼロです。
