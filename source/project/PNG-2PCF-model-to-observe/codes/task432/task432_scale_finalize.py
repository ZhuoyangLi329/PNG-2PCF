#!/usr/bin/env python3
"""尺度选择收尾：验收现有结果、归档一次性网格、备份任务书、生成小交付包。

仅检查本轮明确目录，不扫描整个outputs；不删除文件，不复制大型链进meeting。
需在六页最终PDF视觉验收后运行，参数记录人工视觉核对的最终PDF hash。
"""
import argparse,datetime,json,shutil,tarfile
from pathlib import Path
from task432_scale_selection import ROOT,OUT,BASE,RIC,COMPILED,CASES,common,sha
from task432_scale_deliver import DEST,MAIN

def read(p):
    """读取小审计JSON供最终一致性检查。"""
    return json.loads(Path(p).read_text())

def run(visual_hash):
    """输入已视觉验收的PDF sha256，输出hygiene/STATUS和可下载交付包。"""
    assert sha(DEST/MAIN)==visual_hash
    report=read(DEST/'results.json');audit=read(OUT/'postflight.json')
    gates={'posterior_numeric':audit['status']=='pass' and all(audit['gates'].values()),
           'BAO_inputs':all(read(OUT/'expanded_input_audit.json')['checks'].values()),
           'PDF_matches_result':report['pdf_sha256']==visual_hash,
           'six_cases':set(report['map_scan']['cases'])==set(CASES)}
    for p,h in audit['input_hashes'].items():gates['unchanged_input:'+p]=sha(p)==h
    for p in (OUT/'chains').glob('*/*/input_manifest.json'):
        m=read(p);s=read(p.parent/'summary.json')
        gates['chain:'+str(p.parent.relative_to(OUT))]=s['status']=='pass' and all(s['gates'].values()) and s['nsteps']==30000
        gates['source:'+str(p.parent.relative_to(OUT))]=m['source_sha256']==sha(ROOT/'codes/task432/task432_scale_selection.py')
        gates['model:'+str(p.parent.relative_to(OUT))]=m['compiled_sha256']==sha(COMPILED) and m['baseline_sha256']==sha(BASE/'frozen_inputs.npz')
    gates['no_raster']=not any(p.suffix.lower() in ('.png','.jpg','.jpeg') for d in (OUT,DEST) for p in d.rglob('*') if p.is_file())
    assert all(gates.values()),gates
    arch=ROOT/'old_doc_codes/task432_scale_selection';arch.mkdir(parents=True,exist_ok=True)
    movefile=arch/'checkpoint_move_manifest.json';moves=read(movefile) if movefile.exists() else []
    grid=OUT/'logs/bao_raw_grid'
    if grid.exists():
        for p in sorted(grid.glob('row_*.npz')):
            q=arch/'bao_raw_grid'/p.name;q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists()
            moves.append({'source':str(p),'destination':str(q),'size':p.stat().st_size,'sha256':sha(p),
                          'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Completed one-off raw-grid checkpoint; final expanded_model retained; archive for rebuild'})
            shutil.move(p,q)
        if not list(grid.iterdir()):grid.rmdir()
    common.save(movefile,moves)
    for name in ('expanded_input_audit.json','expanded_model_manifest.json'):
        shutil.copy2(OUT/name,DEST/name)
    shutil.copy2(__file__,DEST/'source'/Path(__file__).name)
    task=ROOT/'agent/task.md';marker='### 2026-09-22 Task4.3.2 少量尺度选择（EZmock1000 / full IC）'
    if marker not in task.read_text():
        backup=arch/'task_before_scale_selection.md';assert not backup.exists();shutil.copy2(task,backup)
        text='\n\n'+marker+'\n\n'
        text+='- 当前入口：`9.22meeting/task432_scale_selection_ezmock1000/RESULTS.md`，同目录六页PDF、results.json、constraints.csv、executed_cut_manifest.json和postflight.json，喵～\n'
        text+='- 用户要求少数配置，实际仅baseline、kmin0p013、kmax0p06、smin80、smax250、bao_unmasked；没有执行早期39配置方案，喵～\n'
        text+='- 保留hybrid + full GIC/RIC、1000 EZ C_single/full cross、25halo均值及原先验，无新自由参数，喵～\n'
        text+='- kmax=0.06移除P0/P2各中心0.062/0.070/0.078三点；Pk/xi/joint的b1中位数=2.386964/2.418489/2.396711，MAP=2.396832/2.422866/2.400993，两种中心均居中，喵～\n'
        text+='- joint sigma68(b1)从0.068493到0.094234（+37.6%），sigma68(fNL)从26.0202到26.9815（+3.7%）；这是敏感性结果，未替换9.18基准，也不代表物理问题已经解决，喵～\n'
        text+='- 取消BAO mask的中位数Pk/xi/joint=2.409815/2.393127/2.371069，仍未居中；kmin、smax几乎不移动joint；smin80的MAP居中伴随xi信息大量丢失，局部sigma(b1)约0.54，喵～\n'
        text+='- 只为kmax(P/joint)及BAO(xi/joint)跑4条新链，每条64x30000、burn5000，全部严格门通过；不变单项data/C/模型等价性通过后复用原链，喵～\n'
        text+='- BAO补回8点由原1000EZ/25halo测量构建，旧74维和预测回切验证通过；68个实际后验数值探针全部通过，最大xi geometry q=0.005744<0.01，喵～\n'
        text+='- 高k六点在旧MAP的条件raw chi2=0.168790，在删点新MAP=2.197057；旧拟合没有明显异常点证据，不以不同向量chi2下降宣称拟合更好，喵～\n'
        text+='- 2000次同phase局部bootstrap的joint b1位移中位数+0.02322，[16%,84%]=[+0.01274,+0.03277]；局部近似、同样本和先验边界限制保留，喵～\n'
        text+='- 推荐代码：codes/task432/task432_scale_selection.py、scale_bao.py、scale_numerics.py、scale_diagnostics.py、scale_deliver.py（后4个同task432_前缀）；输入与链在scale_selection_ezmock1000，先读STATUS.json，不递归扫描大目录，喵～\n'
        text+='- 全部CPU在login07绑定6-13，已完成；41个BAO网格检查点及首版排版PDF在old_doc_codes/task432_scale_selection归档，完整链和扩展模型保留供复核，喵～\n'
        text+='- 后续若继续，优先查高k P0/P2通过sigma_s_P/sn0及cross covariance改变退化方向的机制，暂不扩大cut扫描，喵～\n'
        temporary=task.with_suffix('.scale.tmp');temporary.write_text(task.read_text()+text);temporary.replace(task)
    status={'status':'complete','science_status':'diagnostic_scale_sensitivity_baseline_not_replaced','cases':list(CASES),
        'new_chains':4,'recommended_entry':str(DEST/'RESULTS.md'),'numerical_audit':str(OUT/'postflight.json'),
        'visual_QA':{'pages':6,'status':'pass','pdf_sha256':visual_hash,'method':'All pages rendered in memory and visually inspected; no raster files retained'},
        'next_read':[str(DEST/x) for x in ['RESULTS.md','results.json','executed_cut_manifest.json','postflight.json']],
        'large_files_retained':'Four HDF backends for continuation/audit and four compressed postburn chains for exact plots; do not read by default',
        'archive_manifest':str(movefile),'all_work_on_login':'login07 taskset 6-13, BLAS/OMP=1; no Slurm'}
    common.save(OUT/'STATUS.json',status);common.save(DEST/'STATUS.json',status)
    inv=[]
    for directory,role in [(OUT,'numeric inputs/chains/audits'),(DEST,'delivery')]:
        for p in sorted(directory.rglob('*')):
            if p.is_file() and p.name not in ('hygiene_audit.json','artifact_manifest.json'):
                inv.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'role':role})
    hygiene={'status':'pass','gates':gates,'files':inv,'total_bytes':sum(x['bytes'] for x in inv),
        'archived_checkpoint_files':len(moves),'deleted_files':[],
        'taskbook':str(task),'taskbook_sha256':sha(task),'source_sha256':sha(__file__),
        'note':'Bounded inventory of this experiment only. HDF and NPZ are intentional checkpoint/export pairs. Old baseline physics files match preflight hashes.'}
    common.save(OUT/'hygiene_audit.json',hygiene);common.save(DEST/'hygiene_audit.json',hygiene)
    common.save(DEST/'artifact_manifest.json',{'status':'pass','files':{str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}})
    pack=OUT/'logs/scale_selection_delivery.tar.gz'
    with tarfile.open(pack,'w:gz') as tar:tar.add(DEST,arcname=DEST.name)
    print(json.dumps({'status':'complete','gates':len(gates),'archive_count':len(moves),'output_files':len(inv),
                      'package':str(pack),'package_bytes':pack.stat().st_size,'package_sha256':sha(pack)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--visual-hash',required=True);a=p.parse_args();run(a.visual_hash)
