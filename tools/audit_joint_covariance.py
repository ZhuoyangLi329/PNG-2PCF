#!/usr/bin/env python3
"""离线复核 rawbox joint 数值疑点，不运行 MCMC、不导入原流程入口。

流程：读缓存和测量 → 只加载原源码三个 covariance 函数的 AST →
重建四极联合的边际块 → cross=0 控制 → 重现原始特征值 floor →
比较 xi 块变化、量纲缩放敏感性，并检查现成 summary 的 chi² 可加性。
输入：仓库 source 中的原始缓存、P02测量、JSON。输出：stdout JSON。
依赖：Python 3.10+、NumPy。保留原方程；此工具不是修复或重新拟合。
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import ast,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
R=Path(__file__).resolve().parents[1]
S=R/'source/project'; B=S/'outputs/task43_outputs/rsd_validation/rawbox'

def load_function(path,name,namespace):
    """只编译具名函数定义，避免原模块的输入输出副作用；输入命名空间提供类型/常量。"""
    tree=ast.parse(path.read_text()); node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[]))
    exec(compile(module,str(path),'exec'),namespace)
    return namespace[name]

def floored(cov):
    """逐字对应原 joint assemble 的原始量纲矩阵 floor 规则，返回矩阵及诊断。"""
    cov=(cov+cov.T)/2;e,v=np.linalg.eigh(cov);floor=max(1e-14*float(e[-1]),1e-300)
    if e[0]<floor:cov=(v*np.maximum(e,floor)[None,:])@v.T;cov=(cov+cov.T)/2
    return cov,{'min_before':float(e[0]),'max_before':float(e[-1]),'floor':floor,'n_floored':int((e<floor).sum())}

def main():
    """重建89维naive四向联合并报告；不声称校准了模型或协方差物理正确性。"""
    with np.load(S/'outputs/task43_outputs/rsd_validation/theory/task43_rsd_fulldiscrete_z0p725000_box2000_kmax3_ell02.npz',allow_pickle=False) as d:
        model=SimpleNamespace(**{k:d[k] for k in d.files})
    model.mu,model.wmu=np.polynomial.legendre.leggauss(64);model.mu2=model.mu**2;model.ell_values=(0,2)
    model.legendre={0:np.ones(64),2:0.5*(3*model.mu2-1)}
    ns={'np':np,'S_EDGES':model.s_edges}
    pscript=S/'codes/task43/task43_fit_rsd_rawbox_pk0_vs_xi0_smin50.py'
    tree=ast.parse(pscript.read_text())
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id in ['LIGHTCONE_FIT_EDGES','RAWBOX_FIT_EDGES'] for z in n.targets):
            exec(compile(ast.Module(body=[n],type_ignores=[]),str(pscript),'exec'),ns)
    edges=ns['RAWBOX_FIT_EDGES']
    with np.load(B/'pk/task43_rsd_rawbox_p02_AbacusSummit_base_c000_ph000_mmin1p4e13_mesh400.npz',allow_pickle=False) as d:
        ids=[int(np.flatnonzero(np.all(np.isclose(d['k_edges'],e,rtol=0,atol=1e-12),axis=1))[0]) for e in edges];nmodes=d['nmodes'][ids]
    summary=json.loads((B/'joint_p02xi02/audits/task43_rsd_rawbox_joint_4way_summary.json').read_text());nbar=summary['covariance']['nbar_mean']
    code=S/'codes/task43/task43_rsd_rawbox_joint_4way.py'
    ang=load_function(code,'angular_totals',ns)(model,b1=2.55,sigma_s=8.0,nbar=nbar)
    cpp=load_function(code,'pk_pole_cov',ns)(model,edges,nmodes,ang)
    cxx=load_function(S/'codes/task43/task43_fit_rsd_rawbox_x25.py','periodic_gaussian_covariance',ns)(model,b1=2.55,sigma_s=8.0,nbar=nbar)
    s=(model.s_edges[1:]+model.s_edges[:-1])/2;ids=np.r_[np.flatnonzero(s>=50),32+np.flatnonzero(s>=80)];cxx=cxx[np.ix_(ids,ids)]
    c=np.zeros((len(cpp)+len(cxx),)*2);c[:len(cpp),:len(cpp)]=cpp;c[len(cpp):,len(cpp):]=cxx
    after,meta=floored(c);xx_after=after[len(cpp):,len(cpp):]
    scaling=np.r_[np.full(len(cpp),1e-6),np.ones(len(cxx))];rescaled,_=floored(c*np.outer(scaling,scaling));restored=rescaled/np.outer(scaling,scaling)
    out={'scope':'offline reproduction of uncorrelated (cross=0) rawbox four-way assembly; no new fit',
         'raw_floor':meta,'xi_variance_ratio_median':float(np.median(np.diag(xx_after)/np.diag(cxx))),
         'xi_sigma_ratio_median':float(np.median(np.sqrt(np.diag(xx_after)/np.diag(cxx)))),
         'xi_block_relative_frobenius_change':float(np.linalg.norm(xx_after-cxx)/np.linalg.norm(cxx)),
         'unit_change_xi_relative_frobenius':float(np.linalg.norm(restored[len(cpp):,len(cpp):]-xx_after)/np.linalg.norm(xx_after)),
         'necessary_chi2_checks':{}}
    for name,file,p,x,j in [('monopole','joint_pkxi/audits/task43_rsd_rawbox_joint_pkxi_summary.json','p0_marginal','xi0_marginal_s50','joint_naive_s50'),('fourway','joint_p02xi02/audits/task43_rsd_rawbox_joint_4way_summary.json','p02_marginal','xi02_marginal','joint_naive')]:
        d=json.loads((B/file).read_text())['results'];bound=d[p]['nominal']['chi2']+d[x]['nominal']['chi2'];actual=d[j]['nominal']['chi2']
        out['necessary_chi2_checks'][name]={'independent_minima_sum':bound,'reported_naive_joint_minimum':actual,'violates_unchanged_block_lower_bound':actual<bound-1e-8}
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
