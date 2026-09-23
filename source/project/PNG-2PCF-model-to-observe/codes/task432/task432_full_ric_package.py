#!/usr/bin/env python3
"""打包本轮核心产物，并在PDF视觉验收后更新任务书与小文件归档清单。

执行大纲：要求最终数值gate通过 -> 只打包明确列出的报告/源码/编译输入 ->
视觉验收后追加当前任务状态 -> 归档本轮不足1MB的小型传输包。
不删除任何catalog、核、链或旧结果，不递归修改其他任务目录。
"""
import argparse
import datetime
import json
import shutil
import tarfile
from pathlib import Path
from task432_full_ric_geometry import ROOT, OUT, BASE, sha, save_json

MEETING = ROOT/'9.22meeting/task432_hybrid_fullRIC_ezmock1000_9.18contract'
RUN = OUT/'formal_mean4'


def package():
    """制作轻量交付包；大目录、完整链和原始核留在远端manifest指定位置。"""
    report = json.loads((RUN/'results.json').read_text())
    assert report['status'] == 'pass'
    assert json.loads((RUN/'postflight.json').read_text())['status'] == 'pass'
    files = [(p, Path(p.name)) for p in sorted(MEETING.iterdir()) if p.is_file()]
    for p in sorted((ROOT/'codes/task432').glob('task432_full_ric_*.py')):
        files.append((p, Path('source')/p.name))
    for name in ('task432_hybrid_jaxpower_0918_contract.py', 'task432_hybrid_gic_0918_contract.py',
                 'task432_hybrid_gic.py', 'task432_lightcone_png_velocileptors.py'):
        p = ROOT/'codes/task432'/name
        if p.exists():
            files.append((p, Path('source/inherited')/p.name))
    for variant in ('p02', 'xi02', 'joint'):
        for name in ('summary.json', 'input_manifest.json', 'convergence_latest.json'):
            files.append((RUN/variant/name, Path('audits/fits')/variant/name))
    files.append((RUN/'postflight.json', Path('audits/postflight.json')))
    for p in sorted((OUT/'smoke').glob('*.json')):
        files.append((p, Path('audits/smoke')/p.name))
    for name in ('README.md', 'SHOT_AND_NUMERICS.md', 'pdf_visual_audit.json', 'repository_audit.json'):
        if (OUT/name).exists():
            files.append((OUT/name, Path(name)))
    compiled = OUT/'pilot/mean4xi_mean2P_prodrr_stochastic'
    for name in ('engine.npz', 'provenance.json', 'model_validation.json'):
        files.append((compiled/name, Path('model')/name))
    files.append((OUT/'paired/paired_74vectors.npz', Path('audits/paired_74vectors.npz')))
    manifest = {'status': 'pass', 'remote_root': str(OUT), 'recommended_run': str(RUN),
                'baseline_frozen_inputs': {'path': str(BASE/'frozen_inputs.npz'), 'sha256': sha(BASE/'frozen_inputs.npz')},
                'large_reproducibility_inputs_retained_remotely': ['geometry/', 'paired/', 'formal_mean4/*/chain.h5', 'formal_mean4/*/samples.npz'],
                'entries': [{'source': str(p), 'bundle_path': str(rel), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p, rel in files]}
    manifest_path = OUT/'delivery_manifest.json'; save_json(manifest_path, manifest)
    target = OUT/'delivery_bundle.tar.gz'
    with tarfile.open(target.with_suffix('.tmp.tar.gz'), 'w:gz') as tar:
        for p, rel in files:
            tar.add(p, arcname=str(rel))
        tar.add(manifest_path, arcname='delivery_manifest.json')
    target.with_suffix('.tmp.tar.gz').replace(target)
    print(json.dumps({'bundle': str(target), 'bytes': target.stat().st_size, 'files': len(files)}), flush=True)


def finalize():
    """仅在本地逐页视觉验收回传后追加任务书；小传输包按清单可恢复归档。"""
    visual = json.loads((OUT/'pdf_visual_audit.json').read_text()); assert visual['status'] == 'pass'
    for item in visual['files']:
        assert sha(MEETING/item['file']) == item['sha256'], 'PDF changed after visual review'
    result = json.loads((RUN/'results.json').read_text()); assert result['status'] == 'pass'
    post = json.loads((RUN/'postflight.json').read_text()); assert post['status'] == 'pass'
    archive = ROOT/'old_doc_codes/task432_full_ric_transfer_20260922'; archive.mkdir(parents=True, exist_ok=True)
    previous_manifest = archive/'move_manifest.json'
    moves = json.loads(previous_manifest.read_text())['moves'] if previous_manifest.exists() else []
    names = ('new_sources.tar.gz', 'shot_updates.tar.gz', 'audit_updates.tar.gz', 'status_notes.tar.gz',
             'stochastic_updates.tar.gz', 'new_scripts.tgz', 'delivery_scripts.tgz', 'refinement_scripts.tgz')
    for name in names:
        source = OUT/name
        if source.exists():
            assert source.stat().st_size < 1024*1024
            target = archive/name
            if target.exists():
                raise FileExistsError(target)
            moves.append({'source': str(source), 'destination': str(target), 'bytes': source.stat().st_size,
                          'sha256': sha(source), 'reason': 'Small task-owned transfer archive; extracted source and all science inputs/results retained.'})
            shutil.move(source, target)
    save_json(archive/'move_manifest.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'moves': moves})
    task = ROOT/'agent/task.md'; sentinel = '### 2026-09-22 Task4.3.2 eBOSS full IC（EZmock1000 / 9.18）'
    text = task.read_text()
    if sentinel not in text:
        backup = archive/'task_before_full_ric_append.md'
        shutil.copy2(task, backup)
        lines = ['', sentinel, '',
                 '- 当前入口：`9.22meeting/task432_hybrid_fullRIC_ezmock1000_9.18contract/RESULTS.md`，喵～',
                 '- 完整数值/链目录：`outputs/task43_outputs/rsd_validation/task432_model_repair/full_ric_p02xi02_v1/formal_mean4/`，喵～',
                 '- 四分量全部加入作者cross+auto，GIC只记一次；保留74点、EZmock1000 C_single、full cross和原先验，喵～',
                 '- 既有sn0投影进入xi，xi-only启用同一sn0（3参数）；P和joint仍4参数，没有新RIC幅度，喵～',
                 '- xi采用四个独立scramble平均，P采用两个；独立双核均值、后验尾部、直接hybrid及积分门全部通过，喵～',
                 '- 每组64 walkers、burn5000；Hartlap用于似然，Percival用于区间，喵～', '']
        for v, r in result['results'].items():
            p = r['posterior']['b1']
            lines.append(f"- {v}: {r['nsteps']} steps, b1 MAP={r['map_theta'][1]:.6f}, median={p['q50']:.6f}, 68%=[{p['q16']:.6f},{p['q84']:.6f}], raw chi2={r['map_raw_chi2']:.6f}，喵～")
        comparison = result['joint_b1_comparison']
        lines += ['', f"- joint b1位于两侧之间：MAP={comparison['MAP']['joint_between']}，median={comparison['median']['joint_between']}；中心值位置和区间重叠须分别解读，喵～",
                  '- 一个halo与两个EZ完成四分量paired方向/量级检查；不能据此证明公共random covariance匹配shuffled halo，喵～',
                  '- 主图沿用9.18 triangle样式，另有四页PDF诊断（修正、paired、global/radial归因、b1前后），仅保留PDF，喵～',
                  '- 首轮mean2链MCMC已收敛，但xi尾部核差0.0109459超过0.01，因此保留作数值追溯，不推荐作为最终结果；没有放宽门限，喵～',
                  '- 推荐代码：task432_full_ric_compile.py、fit.py、postflight.py、deliver.py（后3个同task432_full_ric_前缀）；下一位先读formal_mean4/results.json和postflight.json，无需扫描所有outputs，喵～',
                  '- 本轮所有CPU计算绑定login11的6-13；断联后至少15分钟再重连，喵～',
                  '- 用户最新的持续授权覆盖任务书旧第3条：以后缺包自行查找现有环境或隔离安装，无须再次请求批准，喵～']
        with task.open('a') as stream:
            stream.write('\n'.join(lines)+'\n')
    files = [p for p in OUT.rglob('*') if p.is_file()]
    audit = {'status': 'pass', 'science_files_deleted': 0, 'science_files_moved': 0,
             'transfer_archives_moved': moves, 'task_output_file_count': len(files),
             'task_output_bytes': sum(p.stat().st_size for p in files),
             'png_files': [str(p) for p in files if p.suffix.lower() == '.png' and 'reference' not in p.relative_to(OUT).parts],
             'recommended_entry': str(MEETING/'RESULTS.md'), 'recommended_run': str(RUN),
             'retained_nonfinal': ['formal/ (mean2 kernel, failed posterior-tail geometry gate)', 'pilot/ (candidate optimizations and immutable compiled models)',
                                   'geometry/ (convergence evidence and formal dependency)', 'paired/ (actual catalogue checks)'],
             'task_document_sha256': sha(task), 'pdf_visual_audit_sha256': sha(OUT/'pdf_visual_audit.json')}
    assert not audit['png_files']
    save_json(OUT/'repository_audit.json', audit)
    package()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('package', 'finalize'))
    a = p.parse_args(); package() if a.action == 'package' else finalize()
