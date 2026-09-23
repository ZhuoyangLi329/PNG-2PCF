#!/usr/bin/env python3
"""Package diagnostics without replacing any frozen formal product."""
import task432_joint_b1_cause_tests as c
from task432_joint_b1_cause_tests import np,Path,json,save,sha
import shutil,csv,datetime

def main():
    out=c.OUT;read=lambda n:json.loads((out/n).read_text())
    summary=read('comparison_summary.json');win=read('window_pilot.json');report=read('report_audit.json')
    assert report['formal_inputs_unchanged'] and report['all_candidate_ML_success']
    with (out/'comparison_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary['rows'][0]));w.writeheader();w.writerows(summary['rows'])
    text="""# Hybrid + GIC：joint b1 定向归因检查（EZmock1000）

本目录是已批准探索计划的第一轮定向实验交付，不是新的正式后验。正式 hybrid 动力学、GIC、数据向量和 EZmock1000 covariance 均未改写。没有接受任何替换模型，也没有为了使 joint 居中而固定 sn0 或调整协方差。

## 固定口径

- 25 个 halo lightcone phase 的均值；74 点：P0 13 点、P2 9 点、xi0/xi2 各 26 点。
- kmax=0.08，P2 kmin=0.015；smin=50，ds=10，BAO mask 80<=s<120，其他选择继承 9.18。
- 拟合采用 EZmock1000 完整 P-xi 交叉协方差 C_single；以下均为最大似然 ML，不是 MCMC 中位数。
- chi-square 未乘 Hartlap；该固定标量不改变 ML。数值误差审计中的 C_mean=C_single/25 仅用于检验数值精度。
- 登陆节点计算，所有计算限制在 CPU 6-13 内，独立 ph000 复算使用原脚本的 CPU 6-11；未提交 Slurm。

## 主要结果

| 检查 | b1(P) | b1(xi) | b1(joint) | joint raw chi-square |
|---|---:|---:|---:|---:|
"""
    for r in summary['rows']:
        text+=f"| {r['label']} | {r['p02']:.6f} | {r['xi02']:.6f} | {r['joint']:.6f} | {r['joint_chi2_Csingle']:.6f} |\n"
    text+="""
### Shot-noise 响应

独立核对 25 phase 的 D/R 权重、alpha、平方权重与 I2 归一化，最大相对误差约 3.5e-13。估计器已正确扣除其定义的 Poisson 自配对项；sn0 是扣除后的剩余随机项，幅度为 1e4*sn0。

替换为未加 k-box 截断的窗口常数模板、以及直接观测 P0 常数模板，joint b1 只改变约 -1.1e-4。额外构造独立加权 Poisson 点集，再把每个 data 粒子拆成两个同位置、半权重粒子：密度场和归一化不变，自配对扣除量按解析公式改变。实际 P0 接触响应误差约 2e-15，P2 泄漏约 1.3e-15。

因此常数随机项的实现不是主要嫌疑。这不能排除 halo exclusion 等导致的尺度依赖，也没有向有限 s 的 xi 加入任意常数。

### 共同理论与跨模型注入

使用同一 P_L、alpha、Kaiser x Lorentzian^2 FoG，构造 P 与宽 k Hankel 变换的 xi；全窗口 GIC 同步计算。主要控制分支匹配 P 的 k-box 支持，连续 xi 分支仅作积分支持对照。

六个真值点覆盖 b1=2.35/2.42/2.49、sigma=2.15/5、fNL=0/+30/-30。自闭合从偏离真值的起点开始，所有恢复 chi-square<1e-8；更高分辨率积分/插值审计最大误差<5e-9（C_mean 度量）。

共同理论生成数据、当前 hybrid 拟合：六组 joint b1 均在 P 与 xi 之间，没有复现实际观测中的低于两边。反向注入可以出现低于两边，说明差异方向很重要，不能把任何一方当作真实模型。实际数据改用共同 Kaiser 控制后，joint b1 更低，这个具体控制模型没有解决问题。

这些是无噪声归因实验，不是完整统计显著性检验。共同模型 joint 仍为四参数；xi-only 使用共享 sigma，因而由 hybrid 的两参数变为三参数。正式 hybrid 没有增加自由参数。P 侧 sn0 始终是独立接触模板。

### 窗口：历史重算差异与 phase 变化分开

使用完整 data 和 x25 random、mesh256、原 local-LOS 估计器，重算 ph000、ph012、ph024。P0/P2 测量与历史文件完全一致，但历史 ph000 窗口没有逐元素复现。当前 mesh2.py 的修改时间晚于历史窗口，且这里没有归档历史库源码，所以不能确定是哪一个具体版本改动造成差异。

"""
    shift=win['within_current_implementation_joint_b1_shifts']['three_phase_pilot_mean']
    drift=win['variants']['ph000']['fits']['joint']['theta'][1]-summary['rows'][0]['joint']
    text+=f"当前实现独立 ph000 复算相对差为 {win['current_repeat_relative_error']:.3e}。历史窗口换成当前 ph000 时 joint b1 改变 {drift:+.6f}；当前三 phase 平均相对当前 ph000 改变 {shift:+.6f}。二者分别记录，不能都叫 phase 变化。\n"
    text+="""

这只是三个 phase 的试算，不是全部 25 phase 平均。当前发现的历史实现差异和 phase 差异不足以解释约 0.04 的主偏移，因此本轮没有扩大到 25 个窗口，也没有将试算替换进正式结果。

### 旧 boxsafe RIC 单项探针

旧算子来自 task43_make_ric_factorized_operator_boxsafe.py，只把理论 ell=0 映射到观测 P0。P2 和 xi 侧没有径向 IC 修正。它将径向 auto 项取负作为净修正近似，不计算完整的两个交叉项；sn0 也没有按 clustering 送入 RIC。

joint b1 从 2.38316 上移至 2.40471，P-only 从 2.41950 变为 2.41502，xi-only 维持 2.42316；偏离缩小，但 joint 仍低于两边。raw chi-square 从 4.51912 变为 4.61182，差值仅 +0.09270。准确结论是：拟合质量基本没有区别，同时参数中心明显移动。不能以此否定 RIC。它尚不能作为正式修正，是因为算子范围与一致性未经验证。

当前 random_z 直接由各 phase 的 data_z 有放回抽样，FKP n(z) 也从该 phase 的数据估计，因此径向选择函数确实会吸收径向起伏。

完整场投影：C_rad=(I-Pi_r) C (I-Pi_r)^T=C-Pi_r C-C Pi_r^T+Pi_r C Pi_r^T。

权重与投影定义一致时，完整径向约束已经包含全局均值约束，不能与标量 GIC 无条件重复相加。负 auto 近似在文献的某些几何中有效，但本项目各多极矩仍需验证。

参考：[de Mattia & Ruhlmann-Kleider (2019)](https://arxiv.org/html/1904.08851v3)，式 (6)、(20)，第 2.4 节。

## 本轮边界与后续入口

已完成：shot 权重审计与实际估计器注入、两个共同理论控制、双向六点注入、旧 RIC 单项响应、三个 phase 窗口与独立 ph000 复算。

未实施：完整各向异性 RIC、替换 random 的完整 P/xi 配对测量、25 phase 全窗口平均、halo-matter 外部锚定、新模型长链。没有新增可接受的正式替换模型。

下一轮优先做“同一 data、固定 FKP、仅改变 random 径向选择”的配对闭合，再补齐 P0/P2/xi0/xi2 一致的 RIC。改变估计器后，旧 C 只能作为诊断度量，不能直接宣称新的正式约束。

## 复现文件

- 6 页 PDF 为主报告；comparison_summary.csv/json 为 ML 表。
- shot_response.json、shot_contact_injection.*：权重及受控接触注入。
- common_theory_cross_injection.json、common_theory_numerical_audit.json、common_kaiser_grid.npz：共同理论与双向注入。
- ric_existing_probe.json：旧算子的路径、hash、范围和响应。
- window_pilot.json、window_pilot_matrices.npz：窗口复算和 phase 比较。
- input_manifest.json、report_audit.json、source_manifest.json：冻结输入和代码审计。
- source/：本轮及相关模型、估计器源文件快照；共享库只读，未修改。
- SHA256SUMS：交付文件校验。

拟合使用现有 desilike Python，窗口和接触注入使用已安装的 cosmodesi main 环境。入口见 source/run_cause_tests.sh。大目录和模型依赖由输入清单指向 NERSC 原路径，不重复复制 catalog。科学图只输出 PDF。
"""
    (out/'README.md').write_text(text)
    sources=out/'source';sources.mkdir(exist_ok=True)
    names=['task432_joint_b1_cause_tests.py','task432_joint_b1_common_theory.py','task432_window_cause_analysis.py','task432_shot_estimator_contact.py','task432_cause_report.py','task432_finalize_cause_bundle.py','task432_joint_b1_diagnostics.py','task432_hybrid_gic.py','task432_hybrid_gic_0918_contract.py','task432_hybrid_jaxpower_0918_contract.py','task432_lightcone_png_velocileptors.py']
    files=[c.ROOT/'codes/task432'/n for n in names]
    files += [c.ROOT/'codes/task43'/n for n in ['task43_make_ric_factorized_operator_boxsafe.py','task43_rsd_boxsafe_p02_increment.py','task43_measure_rsd_lightcone_p02_jaxpower.py','task43_measure_rsd_lightcone_pk0_jaxpower.py','task43_pk_common.py','task43_build_rsd_lightcone_random.py']]
    files += [Path('/pscratch/sd/l/lzy/desi-clustering/clustering_statistics/spectrum2_tools.py'),Path('/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20260321-1.0.0/code/jaxpower/main/lib/python3.12/site-packages/jaxpower/mesh2.py')]
    manifest={}
    for p in files:
        shutil.copy2(p,sources/p.name);manifest[str(p)]={'sha256':sha(p),'mtime_utc':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()}
    save(out/'source_manifest.json',manifest)
    c.MEETING.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if p.is_file() and p.suffix in ('.pdf','.json','.npz','.csv','.md') and not p.name.startswith('window_recompute_drift_preliminary'):
            shutil.copy2(p,c.MEETING/p.name)
    shutil.copytree(sources,c.MEETING/'source',dirs_exist_ok=True)
    hashes=[f'{sha(p)}  {p.relative_to(c.MEETING)}' for p in sorted(c.MEETING.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    (c.MEETING/'SHA256SUMS').write_text('\n'.join(hashes)+'\n')
    print(json.dumps({'meeting':str(c.MEETING),'files':len(hashes),'audit':report}),flush=True)

if __name__=='__main__':main()
