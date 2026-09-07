#!/usr/bin/env python3
"""核对交接包的文件/哈希、导读相对链接、图索引及已复核科学疑点产物。
这是资料完整性检查，不是科学模型验证。输出JSON；失败返回非零状态。
"""
from pathlib import Path
from urllib.parse import unquote
import hashlib,json,re,sys
R=Path(__file__).resolve().parents[1]
def main():
    failures=[];manifest=json.loads((R/'provenance/FILES.json').read_text());listed=set()
    for row in manifest:
        p=R/row['path'];listed.add(row['path'])
        if not p.is_file():failures.append('missing:'+row['path']);continue
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:failures.append('hash:'+row['path'])
        if p.stat().st_size>=100_000_000:failures.append('GitHub large blob:'+row['path'])
    ignored={'provenance/FILES.json','provenance/bundle_validation.json','provenance/github_verification.json'}
    actual={str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git','__pycache__']) and p.name!='.DS_Store'}
    if actual-listed-ignored:failures.extend('unlisted:'+s for s in sorted(actual-listed-ignored))
    authored=[R/'README.md',R/'给GPT5.6pro.md',R/'papers/README.md',*list((R/'docs').glob('*.md'))]
    broken=[]
    for p in authored:
        if p.name=='source_task_rawbox_excerpts.md':continue
        for m in re.finditer(r'\]\(([^\n)]+)\)',p.read_text()):
            dest=m.group(1).strip().split(' "')[0].strip('<>')
            if dest.startswith(('http:','https:','#','mailto:')):continue
            target=(p.parent/unquote(dest.split('#')[0])).resolve()
            if not target.exists():broken.append({'file':str(p.relative_to(R)),'link':dest})
    if broken:failures.append('broken_links')
    required=['README.md','给GPT5.6pro.md',*[f'docs/{x}' for x in ['01_background.md','02_models_and_inference.md','03_covariance.md','04_results.md','05_review_questions.md','06_reproducibility.md','FIGURES.md','CODE_INDEX.md','RESULT_INDEX.md']], 'review_data/joint_covariance_reproduction.json','review_data/npz_inventory.json','source/background/pk-pcf-model/model_3_20_fast.ipynb','source/project/codes/task43/task43_rsd_rawbox_joint_4way.py','source/project/old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/codes/task4/task42_quijote_lcp50_profiler.py','source/external/desilike/theories/galaxy_clustering/primordial_non_gaussianity.py']
    failures.extend('required:'+p for p in required if not (R/p).is_file())
    fig=[p for p in (R/'source').rglob('*') if p.suffix.lower() in ['.pdf','.png','.jpg','.svg'] and '/observelearn/ref/' not in str(p)]
    index=(R/'docs/FIGURES.md').read_text()
    if any(p.name not in index for p in fig):failures.append('figure index coverage')
    paper_paths=[p for p in R.rglob('*.pdf') if '/papers/' in str(p) or '/observelearn/ref/' in str(p)]
    for p in paper_paths:
        if not p.read_bytes().startswith(b'%PDF'):failures.append('invalid paper:'+str(p.relative_to(R)))
    reproduced=json.loads((R/'review_data/joint_covariance_reproduction.json').read_text())
    if reproduced['raw_floor']['n_floored']!=57 or not all(v['violates_unchanged_block_lower_bound'] for v in reproduced['necessary_chi2_checks'].values()):failures.append('missing expected diagnostic evidence')
    report={'status':'pass' if not failures else 'failed','meaning':'review bundle integrity only, not scientific validation','files_hashed':len(manifest),'bytes':sum(x['bytes'] for x in manifest),'python_source_files':sum(p.suffix=='.py' for p in (R/'source').rglob('*')),'figure_files':len(fig),'paper_pdfs':len(paper_paths),'npz_files':len(list((R/'source').rglob('*.npz'))),'broken_links':broken,'failures':failures}
    print(json.dumps(report,ensure_ascii=False,indent=2));return 1 if failures else 0
if __name__=='__main__':sys.exit(main())
