#!/usr/bin/env python3
"""交付 EZmock1000 full-RIC contour、四分量诊断 PDF、数值表与报告。

执行大纲：要求收敛和后验数值检查通过 -> 复用9.18绘图函数 ->
按原 Percival 约定显示区间 -> 区分 MAP 与中位数是否居中 -> 保存归因。
仅创建 PDF；不创建 PNG，不修改任何旧结果。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from task432_full_ric_compile import Engine, common
from task432_full_ric_geometry import ROOT, OUT, BASE, PK_DIR, sha
from task432_full_ric_response import Response
from task432_full_ric_validation import shell_poles, HybridGIC, WindowConvolvedP02Model, PAYLOAD_NPZ
from task432_hybrid_gic_0918_contract import GICEmulator
from task43_plot_ezmock506_covariance_mcmc_vs_jaxpower import triangle_page

DEST = ROOT/'9.22meeting/task432_hybrid_fullRIC_ezmock1000_9.18contract'
STEM = 'task432_hybrid_fullRIC_ezmock1000_P02xi02_joint_kmax0p08_smin50_baomask80_120'


def decompose(data, prediction, cov):
    """原始完整 joint chi2 的 P + xi|P 条件残差分解。"""
    r = data-prediction; cp, cx, xp = cov[:22, :22], cov[22:, 22:], cov[22:, :22]
    shift = xp@np.linalg.solve(cp, r[:22])
    conditional = r[22:]-shift
    schur = cx-xp@np.linalg.solve(cp, xp.T)
    return {'P_marginal': float(r[:22]@np.linalg.solve(cp, r[:22])),
            'xi_marginal': float(r[22:]@np.linalg.solve(cx, r[22:])),
            'xi_conditional_given_P': float(conditional@np.linalg.solve(schur, conditional)),
            'joint': float(r@np.linalg.solve(cov, r)),
            'conditional_xi_residual': conditional.tolist(), 'cross_covariance_shift': shift.tolist()}


def global_prediction(engine, t, pedges, centers):
    """只施加全局投影（含同口径Poisson/sn0），用于区分GIC处方改变和额外径向IC。"""
    g = OUT/'geometry'
    pr = Response(g/'ph000_qmc4322217_n4096_dchi0_refine1_endpoint_inmidpoint_ell2.npz')
    xr = Response(g/'ph000_qmc4322217_n4096_dchi0_refine1_midpoint_inmidpoint_ell8.npz')
    with np.load(g/'production_rr_mu.npz') as a:
        rr = a['rr_mean']
    with np.load(PK_DIR/'task43_rsd_lightcone_p02_ph000_mesh256_kmax0p300_dk0p002.npz') as a:
        tk, te, edges = a['theory_k'], a['theory_ell'], a['theory_edges']
    with np.load(PAYLOAD_NPZ) as a:
        zeff = float(a['zeff'])
    pm = WindowConvolvedP02Model(np.zeros((1, len(tk))), tk, te, zeff=zeff, sigma_step=30.)
    f, b, sig, sn = t; q = f*2*1.686*(b-1); fg = engine.f
    coeff = np.array([b*b, 2*b*q, q*q, 2*b*fg, 2*q*fg, fg*fg])
    rows = np.r_[np.arange(13), np.arange(17, 26)]
    dp = pr.pk_theory_matrix(pedges, edges, te, nquad=16, inner_nquad=16).reshape(26, -1)[rows]@(pm._theory_basis(sig)@coeff)
    model = HybridGIC()
    poles = shell_poles(model, xr.edges, f, b)
    dx = np.einsum('ild,ld->i', xr.xi_matrix(centers, rr_smu=rr), poles)
    shot = json.loads((OUT/'smoke/full_response_pilot.json').read_text())
    amp = [shot['data_shot_amplitude']+shot['random_P_shot_amplitude'],
           shot['data_shot_amplitude']+shot['random_xi_shot_equivalent_P_amplitude']]
    with np.load(g/'stochastic_sn0_response.npz') as a:
        ratio = float(a['I2_ratio'])
    noise = []
    for i, (los, ell, response) in enumerate((('endpoint', 2, pr), ('midpoint', 8, xr))):
        terms = []
        for kind in ('shot', 'shot_white'):
            with np.load(g/f'ph000_{kind}_n16384_dchi0_refine1_{los}_ell{ell}.npz') as a:
                mom = (a['auto']-a['cross'])/response.meta['pair_volume']
            terms.append(response.xi_project(mom, centers, rr_smu=rr) if los == 'midpoint' else
                         response.p_project(mom, pedges, nquad=16).reshape(26)[rows])
        noise.append(amp[i]*terms[0]+sn*1e4*ratio*terms[1])
    raw_x = engine.x(t)[0]-engine.x.correction(t)[0]-engine.x.shot-sn*engine.x.stochastic
    return np.r_[engine.ps(sig)@coeff+sn*engine.shot+dp+noise[0], raw_x+dx+noise[1]]


def run(compiled, run_name='formal'):
    """保存两份 PDF 与审计 JSON，明确 scientific scope 和原先验边界行为。"""
    assert json.loads((OUT/run_name/'postflight.json').read_text())['status'] == 'pass'
    DEST.mkdir(parents=True, exist_ok=True)
    result = {}; chains = {}; old = {}
    for variant in ('p02', 'xi02', 'joint'):
        dest = OUT/run_name/variant
        r = json.loads((dest/'summary.json').read_text()); assert r['status'] == 'pass'
        with np.load(dest/'samples.npz') as a:
            raw = a['chain'].reshape(-1, len(r['parameter_names']))
        median = np.median(raw, axis=0)
        corrected = median+(raw-median)*r['finite_mock_corrections']['percival_sigma_factor']
        for i, name in enumerate(r['parameter_names']):
            assert np.allclose(np.percentile(corrected[:, i], [16, 50, 84]),
                               [r['posterior'][name][k] for k in ('q16', 'q50', 'q84')], rtol=1e-9, atol=1e-9)
        chains['ezmock1000_'+variant] = {'chain': corrected, 'summary': r}
        r['sn0_raw_mass_above_0p9'] = float(np.mean(raw[:, r['parameter_names'].index('sn0')] > .9))
        result[variant] = r
        old[variant] = json.loads((BASE/'fits'/variant/'summary.json').read_text())
    ranges = json.loads(common.REFERENCE.read_text())['shared_axis_ranges']
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 10., 'axes.linewidth': 1.,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
    main = DEST/(STEM+'_9.18style_v1.pdf')
    diagnostic = DEST/(STEM+'_diagnostics_v1.pdf')
    if main.exists() or diagnostic.exists():
        raise FileExistsError('Immutable final figure exists')
    with PdfPages(main) as pdf:
        triangle_page(pdf, cov='ezmock1000', cov_display='EZmock-1000 (hybrid + full IC)', chains=chains, ranges=ranges)
    engine = Engine('joint', compiled)
    with np.load(BASE/'frozen_inputs.npz') as a:
        covariance, pedges, centers = a['covariance'], a['p_edges'], a['xi_centers']
    t = np.asarray(result['joint']['map_theta']); new_prediction = engine.predict(t)[0]
    oldx = GICEmulator(BASE/'xi_emulator.npz', json.loads((BASE/'input_audit.json').read_text())['emulator_method'])
    f, b, sig, sn = t; q = f*2*1.686*(b-1); fg = engine.f
    coeff = np.asarray([b*b, 2*b*q, q*q, 2*b*fg, 2*q*fg, fg*fg])
    old_at_new = np.r_[engine.ps(sig)@coeff+sn*engine.shot, oldx(t)[0]]
    with np.load(compiled) as a:
        fixed, stochastic = a['fixed_shot_vector'], a['sn0_ric_response']
    delta = new_prediction-old_at_new
    global_at_new = global_prediction(engine, t, pedges, centers)
    global_change = global_at_new-old_at_new
    radial_extra = new_prediction-global_at_new
    assert np.allclose(global_change+radial_extra, delta, rtol=1e-10, atol=1e-10)
    with np.load(OUT/'paired/paired_74vectors.npz') as a:
        paired, paired_model, names = a['differences'], a['model_reference'], a['names']
    xs = [pedges.mean(axis=1), pedges[4:].mean(axis=1), centers, centers]
    slices = [slice(0, 13), slice(13, 22), slice(22, 48), slice(48, 74)]
    labels = [r'$P_0$', r'$P_2$', r'$\xi_0$', r'$\xi_2$']
    std = np.sqrt(np.diag(covariance))
    with PdfPages(diagnostic) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ax, x, sl, label in zip(axes.flat, xs, slices, labels):
            ax.axhline(0, color='0.7', lw=.7)
            ax.plot(x, delta[sl]/std[sl], '-o', ms=3, label='Full IC - old scalar GIC')
            ax.plot(x, fixed[sl]/std[sl], '--', label='Fixed Poisson IC')
            ax.plot(x, sn*stochastic[sl]/std[sl], ':', label='Existing sn0 IC')
            ax.set_title(label); ax.set_ylabel(r'$\Delta m / \sigma_{\rm single,bin}$')
            ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$' if sl.start < 22 else r'$s\ [h^{-1}\mathrm{Mpc}]$')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Deterministic model changes at the full-IC joint MAP\nSame parameters, frozen EZmock1000 covariance')
        pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ax, x, sl, label in zip(axes.flat, xs, slices, labels):
            ax.axhline(0, color='0.7', lw=.7)
            ax.plot(x, paired_model[sl]/std[sl], 'k-', lw=2., label='Radial - global model')
            for i, name in enumerate(names):
                ax.plot(x, paired[i, sl]/std[sl], '-o', ms=2, lw=.8, alpha=.7, label=str(name))
            ax.set_title(label); ax.set_ylabel(r'$\Delta / \sigma_{\rm single,bin}$')
            ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$' if sl.start < 22 else r'$s\ [h^{-1}\mathrm{Mpc}]$')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Measured shuffled - reference changes: one halo and two EZ mocks\nQualitative check only; C_single is not the paired covariance')
        pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ax, x, sl, label in zip(axes.flat, xs, slices, labels):
            ax.axhline(0, color='0.7', lw=.7)
            ax.plot(x, global_change[sl]/std[sl], '-o', ms=3, label='Geometric global IC - old scalar GIC')
            ax.plot(x, radial_extra[sl]/std[sl], '-s', ms=3, label='Additional radial IC')
            ax.plot(x, delta[sl]/std[sl], 'k--', label='Total change')
            ax.set_title(label); ax.set_ylabel(r'$\Delta m / \sigma_{\rm single,bin}$')
            ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$' if sl.start < 22 else r'$s\ [h^{-1}\mathrm{Mpc}]$')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Separate global prescription changes from additional radial IC\nCommon full-IC joint MAP; fixed Poisson and existing sn0 included')
        pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for j, variant in enumerate(('p02', 'xi02', 'joint')):
            for offset, rr, color, label in ((-.10, old[variant], '0.55', 'hybrid + scalar GIC'),
                                            (.10, result[variant], 'tab:blue', 'hybrid + full IC')):
                p = rr['posterior']['b1']; m = p['q50']
                ax.errorbar(m, j+offset, xerr=[[m-p['q16']], [p['q84']-m]], fmt='o', color=color,
                            label=label if j == 0 else None, capsize=4)
                ax.plot(rr['map_theta'][1], j+offset, 'x', color=color, ms=7)
        ax.set_yticks(range(3), ['P-only', 'xi-only', 'joint']); ax.invert_yaxis()
        ax.set_xlabel(r'$b_1$'); ax.legend(); ax.grid(axis='x', alpha=.2)
        ax.set_title('Bias constraints: median with 68% interval; crosses mark MAP\nEZmock1000 C_single, Hartlap likelihood and Percival intervals')
        pdf.savefig(fig); plt.close(fig)
    between = {}
    for key in ('MAP', 'median'):
        bb = {v: r['map_theta'][1] if key == 'MAP' else r['posterior']['b1']['q50'] for v, r in result.items()}
        between[key] = {'b1': bb, 'joint_between': min(bb['p02'], bb['xi02']) <= bb['joint'] <= max(bb['p02'], bb['xi02']),
                        'joint_minus_lower_single': bb['joint']-min(bb['p02'], bb['xi02'])}
    conditionals = {'new_MAP': decompose(engine.data, new_prediction, covariance),
                    'old_model_at_new_MAP': decompose(engine.data, old_at_new, covariance)}
    attribution = {name: {'vector': v.tolist(), 'squared_norm_Csingle': float(v@np.linalg.solve(covariance, v))}
                   for name, v in (('global_prescription_change', global_change), ('additional_radial_IC', radial_extra), ('total', delta))}
    report = {'status': 'pass', 'results': result, 'old_GIC_results': old, 'joint_b1_comparison': between,
              'conditional_chi2': conditionals, 'IC_attribution_at_common_MAP': attribution,
              'compiled_sha256': sha(compiled), 'axis_ranges': ranges,
              'pdfs': {str(p): sha(p) for p in (main, diagnostic)},
              'parameter_note': 'No new RIC amplitude; joint/P have fNL,b1,sigma_s_P,sn0. xi now also marginalizes existing sn0 in [-1,1].',
              'run_directory': str(OUT/run_name),
              'scope': 'All 74 data/covariance entries and full cross block frozen. Full radial IC includes GIC once. Independently scrambled integration kernels averaged after numerical validation. Covariance remains common-random EZ1000, whereas halo uses shuffled-z randoms.'}
    common.save(DEST/(STEM+'_results.json'), report)
    common.save(OUT/run_name/'results.json', report)
    lines = ['# Hybrid + full RIC：EZmock1000 / 9.18 口径', '',
             '完整径向 IC 已用于 P0、P2、xi0、xi2，包含两个 cross、auto 和正确区分 data/random 的 shot 项；GIC 由总投影包含一次，喵～', '',
             '| 拟合 | b1 MAP | b1 中位数及68%区间 | fNL 中位数及68%区间 | raw chi2 |',
             '|---|---:|---|---|---:|']
    for v, r in result.items():
        bp, fp = r['posterior']['b1'], r['posterior']['fNL']
        lines.append(f"| {v} | {r['map_theta'][1]:.6f} | {bp['q50']:.6f} [{bp['q16']:.6f}, {bp['q84']:.6f}] | {fp['q50']:.3f} [{fp['q16']:.3f}, {fp['q84']:.3f}] | {r['map_raw_chi2']:.6f} |")
    xi_sn = result['xi02']['posterior']['sn0']
    boundary_note = '（位于原先验边界）' if abs(result['xi02']['map_theta'][-1]) > .999 else ''
    lines += ['', f"MAP 是否居中：{between['MAP']['joint_between']}；中位数是否居中：{between['median']['joint_between']}，喵～", '',
              'RIC 没有增加新的自由振幅；原 sn0 在投影后影响 xi，因此 xi-only 从2参数改为3参数，joint仍为4参数，喵～',
              f"xi-only sn0 MAP={result['xi02']['map_theta'][-1]:.4f}{boundary_note}，中位数={xi_sn['q50']:.4f}，68%=[{xi_sn['q16']:.4f},{xi_sn['q84']:.4f}]；应结合宽后验解读，不能把边界最优点视为明确检测，喵～", '',
              'raw chi2 均使用 C_single，不除以25；Hartlap用于似然，Percival用于显示区间，喵～小幅chi2变化不应描述为显著改善拟合质量，喵～', '',
              '主图调用原9.18 triangle_page，保留颜色、轴范围、68%/95%轮廓与图例规则；图例仍显示fNL最大似然及区间，中位数单列在表中，喵～', '',
              '少量配对测量只核对方向和量级，未重建配对covariance，也未证明固定公共random的EZ covariance与shuffled halo完全等价，喵～P的连续Bessel RIC与离散mesh估计器之间仍保留eBOSS式近似；输入LOS近似已单独量化，喵～', '',
              '重现入口：codes/task432/task432_full_ric_fit.py、task432_full_ric_postflight.py、task432_full_ric_deliver.py，完整核和输入hash在full_ric_p02xi02_v1目录，喵～']
    (DEST/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'event': 'delivery_done', 'main': str(main), 'diagnostic': str(diagnostic), 'b1': between}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--compiled', type=Path, required=True)
    p.add_argument('--run-name', default='formal')
    a = p.parse_args(); run(a.compiled, a.run_name)
