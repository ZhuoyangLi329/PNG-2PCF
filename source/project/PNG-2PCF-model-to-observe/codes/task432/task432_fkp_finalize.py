#!/usr/bin/env python3
"""FKP诊断交付收尾，不运行或修改科学计算。

流程：校验小型交付归档 -> 写入独立meeting入口 -> 归档少量旧预览
-> 备份并追加任务书 -> 输出收尾清单；大目录、几何核与正式链全部保留。
"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile

ROOT=Path('/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe')
OUT=ROOT/'outputs/task43_outputs/rsd_validation/task432_model_repair/fkp_conditional_diagnostics'
MEET=ROOT/'9.22meeting/task432_fkp_conditional_ezmock1000'
OLD=ROOT/'old_doc_codes/task432_fkp_conditional'

def sha(p):
    """返回小文件内容SHA256，用于传输与归档前后的一致性检查。"""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    """幂等发布已验证文件；已有同名不同内容时停止，避免覆盖他人文件。"""
    MEET.mkdir(parents=True,exist_ok=True);OLD.mkdir(parents=True,exist_ok=True)
    with tarfile.open(OUT/'final_report_delivery.tar.gz') as t:
        for m in t.getmembers():
            rel=Path(m.name)
            assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
            p=MEET/rel;raw=t.extractfile(m).read()
            assert not p.is_symlink()
            if p.exists():assert p.read_bytes()==raw, f'Existing different file: {p}'
            else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    manifest=json.loads((MEET/'final_manifest.json').read_text())
    for r in manifest['files']:assert sha(MEET/r['path'])==r['sha256'],r['path']
    audit=json.loads((MEET/'postflight.json').read_text());assert audit['status']=='pass'
    archived=[]
    for p in [OUT/'paired_fkp_preview.json',OUT/'paired_fkp_preview.npz',ROOT/'codes/task432/task432_fkp_inspect_inputs.py']:
        if not p.exists():continue
        dest=OLD/'early_preview'/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists();before=sha(p);size=p.stat().st_size
        shutil.move(str(p),str(dest));assert sha(dest)==before
        archived.append({'original':str(p),'archive':str(dest),'sha256':before,'bytes':size,'action':'move'})
    dense=OLD/'run_task432_fkp_ric_dense_parameters.sh'
    if dense.exists():archived.append({'archive':str(dense),'sha256':sha(dense),'bytes':dense.stat().st_size,'action':'earlier parameter backup retained'})
    task=ROOT/'agent/task.md';backup=OLD/'task_before_fkp_conditional_2026-09-22.md'
    heading='### 2026-09-22 Task4.3.2 FKP 与条件模式原因探索'
    text=task.read_text()
    if heading not in text:
        assert not backup.exists();shutil.copy2(task,backup)
        entry='''

### 2026-09-22 Task4.3.2 FKP 与条件模式原因探索

- 本轮诊断入口：`9.22meeting/task432_fkp_conditional_ezmock1000/RESULTS.md`；同目录四页PDF、机器结果、审计与source齐全，喵～
- 保持9.18的74点、EZmock1000 C_single和full-RIC正式结果；本轮没有新增自由参数或正式MCMC，喵～
- 以固定模型导数与EZ C定义5个低维模式，不按halo残差选模；25halo方差比均在25-EZ重采样95%区间，含相关项的2维/5维covariance-shape尾率为0.1712/0.7606，未检出明显失配，喵～
- 两个局部线性b1对比均值为-0.03507/-0.03945，近似均值噪声尾率0.0159；这是固定切线模型诊断，不能当作当前有界后验的精确tension显著性，喵～
- ph000/012/024同data/random位置，仅替换own FKP为其他24phase平均nbar的FKP；D/R同步换权并重算I2、shot、P window和full-IC，P用4N random、xi用2个共同LS blocks，喵～
- 三phase平均净响应：P/xi/joint的b1为+0.000773/+0.002428/-0.000548；joint-P与joint-xi为-0.001321/-0.002975，没有把joint拉回两者之间，喵～
- 固定增量诊断MAP：P=2.414221，xi=2.425208，joint=2.377713；raw joint chi2从4.00615到4.03504，拟合质量基本没有区别，喵～
- 不能把3phase/2blocks试验升级为25phase新权重后验；公共random EZ与shuffled halo完整covariance等价性仍未证明，未触发预留EZ random×FKP扩展分支，喵～
- 每phase两套独立IC差分全部通过预定0.001二次型及0.001 b1门限，最大值分别0.0003505/0.0003537；49项审计通过，全部PDF逐页视觉验收，无PNG，喵～
- 有界检索未找到现成halo-matter谱，公开z=0.5/0.8目录有field_rv_A/halo_rv_A；仅确认入口存在，尚未验证粒子完整性或测量bias锚点，喵～
- 下一步优先定位平均物理模型/观测算子的相干残差，包括宽带、尺度依赖随机项及lightcone红移权重；这些目前仍是待检假设，喵～
- 计算与大目录/核保留在`outputs/task43_outputs/rsd_validation/task432_model_repair/fkp_conditional_diagnostics/`；先读paired_fkp_results.json、postflight.json和收尾清单，无需递归扫描，喵～
- Python3.12测量环境切换到3.11汇总环境时需清除PYTHONPATH/PYTHONHOME，runner已修复；仅重跑汇总，所有测量与核复用，喵～
'''
        task.write_text(text+entry)
    if backup.exists():archived.append({'original':str(task),'archive':str(backup),'sha256':sha(backup),'bytes':backup.stat().st_size,'action':'taskbook backup before append'})
    previous=OLD/'archive_manifest.json'
    old=json.loads(previous.read_text())['files'] if previous.exists() else []
    records={r['archive']:r for r in old+archived}
    previous.write_text(json.dumps({'reason':'Retain superseded previews and parameter/taskbook history; no scientific inputs removed','files':list(records.values())},indent=2)+'\n')
    ps=subprocess.check_output(['ps','-u',os.environ.get('USER','lzy'),'-o','pid=,ppid=,args='],text=True)
    jobs=[s.strip() for s in ps.splitlines() if 'task432_fkp_' in s and 'finalize' not in s and 'remote.py' not in s]
    report={'status':'complete' if not jobs else 'check_remaining_processes','meeting_directory':str(MEET),
            'verified_delivery_files':len(manifest['files']),'postflight_checks':len(audit['checks']),
            'archived_files':list(records.values()),'taskbook_sha256':sha(task),'taskbook_backup':str(backup),
            'remaining_task_processes':jobs,'science_inputs_removed':False,
            'formal_results_replaced':False,'source_sha256':sha(__file__)}
    (OUT/'completion_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    files=[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size} for p in OUT.rglob('*') if p.is_file()]
    (OUT/'repository_audit.json').write_text(json.dumps({'scope':str(OUT),'files':files,'file_count':len(files),
        'total_bytes':sum(x['bytes'] for x in files),'science_inputs_removed':False,
        'recommended_entry':str(MEET/'RESULTS.md'),'archives_manifest':str(previous)},indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='archived_files'}),flush=True)

if __name__=='__main__':main()
