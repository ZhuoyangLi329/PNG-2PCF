######对ai的要求
1.你写的代码应当有详细的注释，中文注释，例如，每个函数具体是干啥的，有啥参数分别是什么，输入和输出是什么东西。并在代码开头有一个大纲，代码中的各个部分，各个函数的执行逻辑关系是什么？
2.千万不要大批量删我文件，要删除很多个文件之前，请再三向我确认，因为之前我看很多ai出过把用户一整个盘的文件全部删掉的情况
3.执行代码的时候，因为我有很多个conda 环境，可能会碰到no module named... 找不到包的情况， 你先不要急着去安装那个包，先向我确认其他环境中有没有这个包，让我来找一下对应环境。
4.除了2,3点以外的操作，我应该都给了你权限，执行时无需向我确认。





###任务3 参数估计与建模

##3.1 功率谱的参数估计
在测量了halo的功率谱后，我要用现有的模型去拟合，限制宇宙学参数
/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk-pcf-model/model.ipynb是一个代码例子。它读取了功率谱文件测量的P0。从desilike中建立了理论模型。并用ultranest做了mcmc，获取了最佳拟合参数。并画了图，对比测量p0的平均值与最佳拟合曲线。

你需要学习这个代码，原始代码已经很好了，稍微优化下，把数据路径改成刚刚测量功率谱的位置/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk
然后新的代码放到同文件夹中的model.ipynb中。注意数据格式匹配
tips:运行这些代码之前，需要conda activate desilike激活desilike python内核。


##3.2 2PCF的建模
已经对P0进行了正确的建模，我们希望用最佳参数拟合得到的P0曲线求出对应的2PCF monopole \xi_0
你肯定很容易找到这个功率谱和2PCF著名的积分关系
$$\xi_0(s) = \int_0^\infty \frac{k^2}{2\pi^2} P_0(k) j_0(ks) \, dk$$

有很多软件包，例如mcfit，能够处理这样的无穷上下限的数值积分（汉克尔变换）。 你可以去了解一下。
在这类积分中，传统的数值积分方法（如辛普森法则或高斯积分）会面临巨大的困难。因为球贝塞尔函数 $j_\ell(ks)$ 在大尺度（高 $ks$ 值）下会表现出高频的剧烈振荡。如果你用普通的数值积分，不仅计算极慢，而且极易累积截断误差，导致结果发散或不准确。

mcfit 能够高效处理它的核心原因，是因为它底层使用了在宇宙学中赫赫有名的 FFTLog 算法（最初由 Andrew Hamilton 在 2000 年引入宇宙学领域）：

请你先不要用mcfit软件包，自己想办法用刚刚说的FFTlog算法，写一个从desilike模型的P0曲线计算\xi_0的模型代码，我的要求是包含k积分的自定义上下限。 动手之前先给我一个可行的方案和小步骤出来。

    ###3.2.1 2PCF的建模稳定性验证（已完成）
    看起来你自己写的fftlog算法不错。算出来的2PCF没有出现很奇怪的东西。但是如何检验我们自己写的算法是对的呢？现在你可以调研一下mcfit了，用mcfit计算这个积分，看一看我们的积分和mcfit是否一致（改变kmax和kmin）。用百分比误差计量。
    https://github.com/eelregit/mcfit
    你可以专门另外写一个代码只用来做这个检验，这样更简洁

    


##3.3 重现PNG 2PCF现有建模的问题。（已完成）
看起来mcfit和现有的积分比较接近，我们就用现有的fftlog。
你可以注意到，目前拟合的数据是fastpm fnl100。 因此拟合出来的png参数fnl会接近100.
请你首先测试，在bestfit参数，固定kmax=20不动，kmin从0.007开始往下降到0.00001,2PCF怎么变
再测试，bestfit参数里的fnl改为0 ，kmin从0.007开始往下降，2PCF还有明显的变化吗？


###4 真正的问题
经过漫长的样本生成和模型检验过程，我们终于开始要解决我的科学问题了
在3.3节中，我们发现了这样一件事情：当fNL=0的时候，从pk到2PCF的模型中，当kmax固定20，kmin降低到一定程度，2PCF曲线是收敛的。
而fNL=100的时候，并不收敛（随着kmin降低2PCF在较大尺度一直抬升）
这背后的本质是：png的fnl会导致halo功率谱在大尺度上有个1/k^2的scale-dependent bias(请参考https://arxiv.org/abs/2411.17623)的第二节Theory部分。 1/k^2项会使得pk到2PCF的无穷积分不收敛。这是不物理的。 而我并不打算从物理的角度解决这个问题（去修正1/k^2项在大尺度的行为）。而是从数值的角度去解决。

##4.1 测量到的2PCF和盒子长度有关系吗？
1gpc fastpm fnl=100的测量2pcf结果在/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pcf和/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk
3gpc fnl100的测量结果在/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pcf和/pscratch/sd/l/lzy/3gpc-UNIT-fastpm/pk中
我们需要先比较一下不同盒子长度测量到的2PCF的平均值有区别吗？

    

##4.2 k-cut问题会与盒子长度有关系吗？
目前对于理论模型中1/k^2项会使得pk到2PCF的无穷积分不收敛这个问题。一个“妥协”是在积分做一个kmin cut（这个cut通常为2*pi/L，也就是盒子长度对应的最大的模式）。

这个妥协并不是很好，你可以测试一下在1gpc盒子情况建模的时候，先拟合功率谱得到best-fit pk参数。再用best-fit pk数值积分得到2PCF的理论曲线，会与测量的平均值符合吗？

然后调研一下，3gpc盒子建模的时候，和测量平均值的差异会变小吗？
    #4.2.1调研一下best-fit pk 曲线在不同盒子有区别吗？ 以及直接测量数据的平均值pk。不同大小的盒子有区别吗？只需要画图看一下即可，最好在图上标记处best-fit参数值。


##4.3 fnl=0的时候还会有上述问题吗？
调查一下，对于现有的fastpm 3gpc fnl=0样本,当我们将best-fit pk积分得到2PCF的时候(这时候kmin可以取得很小，因为没有fnl使得这个积分发散)，会和测量数据直接的2PCF的平均值在大尺度符合吗？？


###5 问题的解决
这部分的2PCF建模参考代码：/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk-pcf-model/model.ipynb

从第4节和第3节的情况来看
当fnl=0的时候，有了一个良好的pk模型，利用fftlog积分可以建模2PCF。
由于这个积分在fnl0的时候是收敛的，只要盒子长度够长（3Gpc/h），kmax取比较大（20例如？），kmin=2pi/L时，积分得到的模型2PCF曲线可以与测量数据的2PCF平均值对上。

当fnl=100的时候，上述办法失效了。我们发现kmin=2pi/L的时候，积分得到的模型2PCF曲线系统性地低于测量数据的2PCF。（如果kmin取得更小，引入更大的模式，可以补偿这一差距，但目前没有理由这么做）。

我的想法是先从数值上解决这个问题：我希望在积分时把kmin设得更小（例如 kmin_global=0.0001），但在 k < k_f=2pi/L 的区域对功率谱（或PNG项）乘一个“IR窗口函数”来压制低k的 1/k^2（bias）导致的功率谱翘起，使得 k->0 时功率谱不再发散/积分收敛。

直觉上，这样做有两个目的：
1) 数值上：让 pk->2PCF 的积分在低k端收敛，避免 fnl=100 时 kmin 越低 2PCF 越“抬升且不收敛”的问题。
2) 经验上：由于 baseline 方案把积分下限硬截断在 k_f=2pi/L，可能漏掉了 k<k_f 这一段对大尺度2PCF的贡献，从而导致模型2PCF在大尺度系统性偏小；通过引入“受控的”低k贡献（窗口抑制后的贡献）来补偿这部分缺失。

我们的目标是：在积分代码中加入“IR窗口函数”后，使用 kmin_global=0.0001、kmax=20 的 FFTLog 积分，bestfit 的 pk 生成的 2PCF 能与观测的平均值在大尺度上对齐（仅需测试3gpc，1gpc不用管）。

补充：如果窗口法效果不稳（对窗口形式/参数过于敏感，或者出现明显FFTLog振铃），可以尝试一个更“干净”的数值替代方案（hybrid）：
在低k端不做连续积分，而是直接用周期盒子的离散 k 模式做求和；当 k 足够大（离散性不再重要）时，再切回连续的 FFTLog 积分。
这样做的核心动机是：PNG 的 IR 行为会让最前几个 k-shell 的贡献异常重要，而这些 shell 在有限盒子里只有极少数离散模式，连续积分近似会更差。


#### 执行清单（给代码专精 AI）
目标：仅针对 3Gpc 盒子、fastPM halo、RSD、fnl=100。用 best-fit 的功率谱模型 $P_0(k)$ 经过“带 IR 处理的积分/FFTLog”得到的 $\xi_0(r)$，在大尺度上与测量 $\xi_0$ 的均值对齐，并量化对齐程度。

1. 明确输入输出与常数
- 盒子边长：`L = 3000.0` (Mpc/h)。
- 基本模式（盒子最小非零模长）：`k_f = 2*pi/L` (h/Mpc)。
- 全局积分下限（数值用）：`kmin_global = 1e-4` (h/Mpc)。
- 固定高 k 截断：`kmax = 20.0` (h/Mpc)（与第 3-4 节一致）。
- 统一对比对象：都用 `r^2 * xi0(r)`，并采用和 `model.ipynb` 一样的插值对齐到测量的 `r` bin。

2. 准备 3Gpc 的观测 2PCF（均值与误差）
- 读取 3Gpc 的 `pcf_rsd_N*.dat`（N=1..50）并计算：
- `xi_mean(r) = mean_N xi_N(r)`，`xi_std(r) = std_N xi_N(r)`。
- 记录 `r` 的范围与 binning（后续所有模型都插值到这套 `r` 上比较）。

3. 得到 3Gpc 的 best-fit $P_0(k)$（仿照现有 notebook 3.1）
- 读取 3Gpc 的 `pk_rsd_N*.dat`（N=1..50），构造 mocks 协方差。
- 用 desilike 拟合得到 `bestfit_params`（输出/记录 best-fit 数值）。
- 在一个高分辨率 `k_grid` 上评估 best-fit 的 `P0(k)` 并缓存成数组（后续扫描窗口参数时不要反复初始化 desilike）。

4. 实现并对比 3 种 IR 处理方案（同一 best-fit $P_0(k)$）
- Baseline（sharp cut）：积分/FFTLog 只取 `[k_f, kmax]`。
- Test A（只扩展积分下限，不加窗）：积分/FFTLog 取 `[kmin_global, kmax]`，且直接用原始 best-fit `P0(k)`（预期可能发散或严重抬升，用来对照“需要窗口”的事实）。
- Test B（IR窗口）：积分/FFTLog 取 `[kmin_global, kmax]`，并在 `k < k_f` 对 `P0(k)` 乘一个平滑窗 `W(k)`，要求：
- `W(k)=1` for `k>=k_f`（不动可观测区间的理论功率谱）
- `W(k)->0` as `k->0`（让低k不再发散/积分收敛）
- `W(k)` 在 `k_f` 附近尽量平滑（避免FFTLog振铃）
- 推荐起步形式：对数k空间的“余弦过渡窗”或归一化的 `exp_power` 窗
  `W(k)=[1-exp(-(k/k_f)^alpha)]/[1-exp(-1)]`（`k<k_f`），`W(k)=1`（`k>=k_f`），
  其中可扫少量 `alpha`，例如 1,2,3,4,6。
- （可选但更稳）只对 PNG 引起的“额外项”加窗：先算 `P0_fnl100(k)`，再算 `P0_fnl0(k)`（其他参数保持 best-fit 不变，仅置 `fnl_loc=0`），取差 `ΔP_png=P0_fnl100-P0_fnl0`，只在 `k<k_f` 让 `ΔP_png -> W(k)*ΔP_png`，这样不会误伤 Gaussian 部分。
- 扫描少量窗口参数（例如 5-10 组），找到能最好对齐观测大尺度的那组。
- Test C（hybrid：离散低k求和 + 连续高k FFTLog）：不引入窗口函数，而是在低k端用离散模式求和来替代积分：
- 选一个切换点 `k_split = N_split * k_f`（例如 `N_split` 取 3, 4, 6, 8 做稳定性测试）。
- 低k部分（离散求和）：遍历整数三元组 `(n_x,n_y,n_z)`（不全为0），构造离散波矢 `k = k_f * (n_x, n_y, n_z)`，取满足 `|k| < k_split` 的模式，计算
- `xi_low(r) = (1/V) * sum_{0<|k|<k_split} P0(|k|) * j0(|k| r)`，其中 `V=L^3`，`j0(x)=sin(x)/x`。
- 高k部分（连续FFTLog）：用现有 FFTLog，把积分下限设为 `kmin = k_split`，上限 `kmax=20`，得到 `xi_high(r)`。
- 合成：`xi_total(r) = xi_low(r) + xi_high(r)`，再与观测均值对比。
- 说明：这个方案的意义是尊重有限盒子在最低几个 k-shell 的离散性，尤其适用于 PNG 这种 IR 权重过强的情况。

5. 量化“对齐程度”（给出数值指标，不只看图）
- 在“大尺度区间”定义一个 mask（例如 `r >= 200 Mpc/h`，可与 `model.ipynb` 保持一致）。
- 指标至少给 1 个：
- `mean(|(Data-Model)/sigma|)` on large scales
- 或 `chi2/ndof` on large scales
- 输出 baseline / TestA / TestB(best) 三者的指标对比表。

6. 必要图像（最少 3 张）
- 图1：3Gpc `P0(k)` mean vs best-fit（标注 best-fit 参数值）。
- 图2：3Gpc `r^2 xi0(r)`：观测均值+误差 vs baseline/TestA/TestB(best)。
- 图3：`(Data-Model)/sigma` vs `r`（只画 baseline 和 TestB(best) 也行，突出“系统性偏低是否被消掉”）。

7. sanity check（避免“为了对齐而对齐”）
- 用同样的 IR 处理流程，在 3Gpc fnl=0 的 best-fit 上复跑一遍，确认不会引入明显的、非物理的大尺度偏差（至少图上要正常、指标不要变差太多）。
- 做一个稳定性检查：在 Test B 的最优窗口参数下，把 `kmin_global` 再降低 10 倍（例如 `1e-5`）看 2PCF 大尺度是否基本不变（确认“窗口真的让IR收敛”）。
- 对 Test C 做稳定性检查：扫 `k_split`（例如 3,4,6,8 倍 `k_f`），看大尺度 `xi_total(r)` 是否对 `k_split` 不敏感；若对 `k_split` 极敏感，说明仍需要额外的 IR 正则或改造理论的 IR 行为。

当前口径（简记）：
- TestB 的经验最佳参数化窗口：`W(k)=[1-exp(-(k/k_f)^x)]/[1-exp(-1)]`（`k<k_f`），`W=1`（`k>=k_f`）。
- 形状参数取盒长依赖：`x(L)=4*(L/1000)`；即 `1Gpc -> x=4`，`3Gpc -> x=12`。
- 该口径配套：`kmin_global=1e-4`，`kmax=20`，并保留 FFTLog 内部 taper。
- 参考：目前的方法请在/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk-pcf-model/model.ipynb代码参考。






##6 validation with Nbody(已通过)
现在在nbody样本中测试这个参数化窗口的效果
我有一些 quijote nobody simulation 长度都是1Gpc
目前有三个宇宙学，fnl100，fnl50，fnl0，每个宇宙学有500个realization
对于所有的realization我都测量了他们的功率谱和2PCF /pscratch/sd/l/lzy/pks_2pcfs
你要做如下几点
    #6.1插值
    对于每个realization（标号一致）的fnl100，fnl50，fnl0的功率谱/2pcf数据，用cubic插值方法插值fnl75,fnl30,fnl20的功率谱/2pcf数据
    这样我们就有了其他fnl的功率谱/2pcf数据（也是每个fnl500个）

    #6.2 验证窗口
    对于每个fnl，利用model.ipynb的方法，先拟合功率谱的平均值（covariance用当前fnl已有的500个realization数据做）。然后best-fit通过最新的参数化窗口的办法计算2PCF，与数据的平均值做对比，

    注意：拟合的时候让p fix在1.2


##7 validation with different halo mass(已通过)
在quijote png nbody simulation的validation看起来效果不错
我们回到fastpm。 你们应该还记得，在生成fastpm 样本的时候会制定一个halo mass的范围。我现在想检验，最大masscut不动，改变最小的masscut，现在对2PCF的建模还会有效吗？
这里我们主要关注 1gpc fnl100的fastpm，它的fof文件位置在/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/fofs_fnl100。你需要生成不同masscut范围的halo样本，并测量pk，2PCF（skill里也有），然后测试2PCF的建模
从halo fof到halo位置的skill在/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/skills/fastpm-halo-rsd-export

现在的mass_min大概是1.4e13,你从1e12开始，一直到1e15,取几个mass_min测试一下。
代码记得提交sbatch，不要直接在登陆节点进行。



##8窗口方法的解析解释(已完成)
我们在前面找到的ir窗口是一个纯数值解决方法，效果还不错。但我们不希望止步于纯数值方法（它没任何物理意义）。
我们知道，在对一个盒子(L,L,L)里的halo数值计算功率谱的时候，是一个离散的，网格上的计算。（例如，目前用的ngrid是512）
既然是离散的计算，我们测到的P(k)就不是连续的，也是离散的，只在某个特定的k上有值。（也许这对我们数值建模2PCF有帮助）

    #8.1一维空间，离散P(k)的解析形式
    假如在一维空间，这一系列特定的k非常简单，就是(k_base,2*k_base,..... ,n*k_base) 其中k_base=2*pi/L
    离散的P(k)可以被\delta function来定义 P(k)= P(k_base)*\delta(k-k_base)+P(2*k_base)*\delta(k-2*k_base)+...........
    当然，实际的离散功率谱测量会取一个kbin，一个bin可能覆盖不止一个有值的k数据点（那就对bin里的k数据点值平均），这个不是我们的重点

    #8.2三维空间，离散P(k)的解析形式
    三维空间，长度为(L,L,L)的盒子，这一系列有值的k要复杂一些了，并不能简单地写为n*k_base (n是整数)
    有值的三维k矢量可以用(a*k_base,b*k_base,c*k_base),其中(a,b,c)都是整数来表示,范围是[0,ngrid]。

    不过既然我们计算的还是一维功率谱P(k)，那这里的k^2就是(k_x^2+k_y^2+k_z^2)了
    显然第一个有值的k是k_base,但是第二个有值的k不再是2*k_base,而是sqrt(2)*k_base了。（对应a,b,c其中2个取1，另一个取0的情况），后续有值的k需要你自己推断。

    我们希望对这样的离散的P(k)仍然用\delta函数解析建模。 P(k)=P(k_base)*\delta(k-k_base)+..............

    其中这里的\delta函数系数P(k_base)，P(sqrt(2)*k_base)等，认为它是前边##3.1 功率谱的参数估计部分,fit测量功率谱时，best-fit功率谱模型的P(k_base)，P(sqrt(2)*k_base)的值....

    你需要首先写出这时候P(k)的解析形式（可以在mission8_log文件夹创建一个ipynb笔记本记录，公式可直接用latex，如果涉及函数运算用sympy）

    #通过解析建模的离散P(k)建模2PCF
    解析建模“离散的P(k)”之后。 从数学上可以做傅立叶变换（hankel变换），或许能解析地算出2PCF。
    这里就没有前面提到的IR发散问题了，因为P(k)在k从(0,k_base)是没有值的！
    
    请你调研，这个变换能被解析地算出来吗？效果与IR纯数值窗相比如何？ 
    
    如果不行的话，尝试用一个解析函数去近似替代\delta函数，2PCF还能被解析地算出来吗？
    然后还需要尝试，对于P(k)，在前几个（例如前5个）最小的有值的k使用离散建模方法。更大的k的P(k)采用连续建模方法（也就是best-fit功率谱模型的P(k)）函数直接拿来用，这样建模出来的2PCF是否准确。

##9 全离散方法的改进
在第8部分中（请看总结），对于离散建模，我们找到的最好方法是全离散，其pipeline是这样的

1. 先用测量的功率谱单极矩均值 `P0(k)` 做 `desilike` 拟合，得到 best-fit 的连续理论功率谱 `P0_best(k)`。
2. 对有限盒子边长 `L`，定义基本模长 `k_f = 2*pi/L`，并把允许的离散波数写成
   `k_q = k_f * sqrt(q)`，
   其中 `q = n_x^2 + n_y^2 + n_z^2`，`n_x,n_y,n_z` 为整数。
3. 对每个 `q` 计算该离散壳层的简并度 `g_q`，也就是满足
   `n_x^2 + n_y^2 + n_z^2 = q`
   的整数三元组个数。
4. 在每个离散壳层上，不再做连续积分，而是直接取理论模型在该壳层模长上的值
   `P0_best(k_q)`。
5. 最终直接做全离散求和建模 2PCF monopole：
   `xi_0(r) = (1 / V) * sum_{q, k_q <= kmax} g_q * P0_best(k_q) * j_0(k_q r)`，
   其中 `V = L^3`，`j_0(x) = sin(x) / x`。
6. 这个方法的物理意义是：在有限体积模拟盒中，低 `k` 模本来就是离散且数量有限的，因此用离散模求和比连续 `k` 积分更贴近真实盒中可存在的傅里叶模式结构。
7. 当前实践中，这个“原始全离散”方法仍然是离散建模里最稳的基准版本；相较于若干平滑化、mode-cell 或低-k 修正方案，它没有引入额外经验窗口，也较少产生额外的人为结构。

它也能够补偿baseline方法建模2PCF在大尺度的系统性偏低，但补偿有些过度了，导致还是不如加ir 窗口。
请你自由探索，想想现在的理论建模在物理，数学上还有什么缺陷吗？ kmax我认为需要15，不能太低。
如果有多核代码需求请提交sbatch，单核代码直接在登陆节点跑。

备注：此部分已完成，叫做databin方法，具体的实现方法请看/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission9_log/databin_method.md和同文件夹里的python脚本


##10全离散/databin方法的区别
我们可以注意到，虽然全离散，databin的方法的数学形式非常接近，但是结果差异挺大。databin效果要好很多。
然而我们需要回答如下问题
1.在实际的2PCF建模中，我们有P_{model}，可以拿它来作为输入用全离散方法算出2PCF，但是我们没有P_{measure},这个时候databin方法还能用吗？
2.我认为是不能用的，如果不能用，该怎么办呢？全离散方法还有什么与测量无关的改进空间吗？（当然，你仍然可以用测量的P(k)拟合得到best-fit P_{model},这一步在我们的研究中是允许的。）

备注：此部分建立了一个新方法 `BinAvgFit + FullDiscrete`。它的核心思想是在功率谱拟合阶段先对理论 `P_model(k)` 做与数据定义一致的 bin-average，再把得到的 best-fit 连续模型输入全离散求和；这样可以在不直接把 `P_measure` 塞回 `pk->2pcf` 的情况下，数值上逼近 databin 的效果。具体实现和结果请看 `/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/agent/mission10_log/`。















##11 全离散求和的加速（已经完成）

全离散求和（FullDiscrete）目前的计算瓶颈：3Gpc 盒子 kmax=15 时共有约 4270 万个非零壳层，暴力逐壳层求和耗时约 100s。

**曾经尝试的 Hybrid 方案（低 k 离散 + 高 k FFTLog）不可行**，原因如下：
- 设想把 k < k_split 用离散求和，k > k_split 用 FFTLog 连续积分来替代高 k 的大量壳层。
- 但 FFTLog 输出的 s 网格范围 ~ 1/k_min，高 k 段的积分下限是 k_split，导致 FFTLog 能覆盖的最大 r ~ 1/k_split。
- k_split 取得越大（节省的壳层越多），FFTLog 能覆盖的大尺度 r 就越小，完全覆盖不到我们关心的 r ~ 100-380 Mpc/h。
- 换句话说：大尺度 xi0(r) 的贡献几乎全来自低 k 模式，高 k 的 FFTLog 对大尺度本来就没有有效贡献，Hybrid 节省不了真正的瓶颈。
寻找其他可行的加速方案，我的要求：直接在python脚本里跑（单核）也要足够快。




##12 binavgfit+FullDiscrete的2PCF建模的参数拟合和功率谱直接参数拟合的区别。
好了，在11节的时候我们找到了一个速度很快，并且很不错的建模方法。示范代码在/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm/pk-pcf-model/model_3_20_fast.ipynb

现在我要做一个检验。我们知道，在desilike中，用功率谱数据，不需要跑mcmc，用profiler可以找一个best-fit值出来（当前是fix p,sn,拟合fnl,b1,sigmas）。
现在呢我们有了2PCF的model，具体应该是输入参数，先计算P_model，然后用加速的全离散计算2PCF
我想试试3Gpc fastpm fnl100的样本中，2PCFmodel （拟合范围先设定在100-350Mpc内）和功率谱直接拟合（用前20个kbin这样子？别忘了拟合的时候要注意kbin的问题，k很小的时候不能用kcenter拟合）得到的bestfit有无差异。






##额外修改（已经解决）
1.一个问题，目前我们拟合功率谱的时候，用fastpm样本的平均值作为拟合的平均值，covariance也是用fastpm样本。但是fastpm只有50/100个。
拟合平均值和covariance在model.ipynb代码这段的(data_ps和covariance)被运用
observable = TracerPowerSpectrumMultipolesObservable(
    data=data_ps,
    covariance=mock_ps_list,
    klim={0: [float(k_values.min()), float(k_values.max()), float(k_values[1] - k_values[0])]},
    theory=theory,
)

我现在生成了一堆3Gpc的ezmock样本，可以用来作为covariance的估计（fnl0，100的情况都没问题）。它们的功率谱已经被计算好，路径在/pscratch/sd/l/lzy/cov_mock/B3000G768Z0N4417983_b0.18d5r270c1.65_seed{myseed}/PK_EZmock_B3000G768Z0N4417983_b0.18d5r270c1.65_seed{myseed}_RSD.dat. 其中myseed=10*i+5
你可以看一下我有多少个EZmock，碰到哪个文件夹的pk缺失就跳过。 现在请把EZmock的功率谱数据作为covariance
