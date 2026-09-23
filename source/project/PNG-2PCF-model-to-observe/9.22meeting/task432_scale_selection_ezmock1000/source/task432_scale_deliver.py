#!/usr/bin/env python3
"""少量尺度选择的最终交付：原9.18风格三页contour与三页诊断，仅PDF。

先验证链和数值验收，再生成报告、精确cut清单与来源hash；既有物理输入
和旧正式结果只读。误差是原Hartlap/Percival约定，不把局部曲率当后验。
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from task432_scale_selection import ROOT,OUT,RIC,BASE,COMPILED,Engine,CASES,VARIANTS,common,sha
from task43_plot_ezmock506_covariance_mcmc_vs_jaxpower import triangle_page,COLORS,VARIANTS as STYLE_VARIANTS

DEST=ROOT/'9.22meeting/task432_scale_selection_ezmock1000'
MAIN='task432_scale_selection_ezmock1000_contours_and_diagnostics.pdf'
LONG=('baseline','kmax0p06','bao_unmasked')
LABEL={'baseline':'Baseline','kmin0p013':r'$k_{\min}=0.013$','kmax0p06':r'$k_{\max}=0.06$',
       'smin80':r'$s_{\min}=80$','smax250':r'$s_{\max}=250$','bao_unmasked':'BAO unmasked'}
COLOR={v:COLORS[c] for v,c,_ in STYLE_VARIANTS}
DISPLAY={'p02':'Pk','xi02':'2PCF','joint':'Joint'}

def read(p):
    """读取机器审计文件，保留原字段。"""
    return json.loads(Path(p).read_text())

def chain_dir(case,variant):
    """只有模型/data/C完全一致的单项复用旧正式链，等价性已在postflight检查。"""
    if case=='baseline' or (case,variant) in [('kmax0p06','xi02'),('bao_unmasked','p02')]:
        return RIC/'formal_mean4'/variant
    return OUT/'chains'/case/variant

def load(case,variant):
    """对原始链应用一次Percival宽度修正，并核对报告全部参数的百分位数。"""
    d=chain_dir(case,variant);s=read(d/'summary.json');assert s['status']=='pass' and all(s['gates'].values())
    with np.load(d/'samples.npz') as a:raw=a['chain'].reshape(-1,len(s['parameter_names']))
    med=np.median(raw,axis=0);corrected=med+(raw-med)*s['finite_mock_corrections']['percival_sigma_factor']
    for i,name in enumerate(s['parameter_names']):
        assert np.allclose(np.percentile(corrected[:,i],[16,50,84]),[s['posterior'][name][k] for k in ('q16','q50','q84')],rtol=1e-9,atol=1e-9)
    s['chain_directory']=str(d);s['chain_reused']=not str(d).startswith(str(OUT))
    return {'chain':corrected,'summary':s}

class AnnotatedPDF:
    def __init__(self,pdf,case):self.pdf=pdf;self.case=case
    def savefig(self,fig,**kwargs):
        """覆盖原绘图函数的固定cut标题；颜色、轴范围、legend和contour原样沿用。"""
        k='0.06' if self.case=='kmax0p06' else '0.08';mask='no BAO mask' if self.case=='bao_unmasked' else 'BAO mask [80,120)'
        fig.suptitle(f'Hybrid + full GIC/RIC | EZmock1000 | {LABEL[self.case]}\n'
                     f'0.4 < z_obs < 0.8; kmax={k}; 50 <= s < 350; {mask}',fontsize=9.5,y=1.025)
        fig.text(.5,.015,'25-halo mean; C_single; Hartlap likelihood; Percival intervals; original 9.18 axes/style',ha='center',fontsize=7)
        self.pdf.savefig(fig,**kwargs)

def diagnostics(pdf,scan,result,paired):
    """先看全部MAP，再看真实后验区间；条件残差和配对局部响应单独标注。"""
    fig,ax=plt.subplots(1,2,figsize=(10,6),gridspec_kw={'width_ratios':[2.3,1]})
    y=np.arange(len(CASES))
    for j,v in enumerate(VARIANTS):
        vals=[scan['cases'][c]['b1_MAP'][v] for c in CASES]
        ax[0].plot(vals,y+(j-1)*.14,'o',color=COLOR[v],label=DISPLAY[v],ms=6)
    ax[0].set_yticks(y,[f"{LABEL[c]}  (n={scan['cases'][c]['fits']['joint']['ndata']})" for c in CASES]);ax[0].invert_yaxis()
    ax[0].set_xlabel(r'$b_1$ MAP');ax[0].legend(loc='lower left');ax[0].grid(axis='x',alpha=.2)
    dd=[scan['cases'][c]['outside_distance'] for c in CASES]
    ax[1].barh(y,dd,color='#65798c');ax[1].invert_yaxis();ax[1].set_yticks(y,[])
    for i,d in enumerate(dd):ax[1].text(d+.001,i,f'{d:.4f}',va='center',fontsize=9)
    ax[1].set_xlim(0,.058);ax[1].set_xlabel(r'$D_{\rm out}$ (MAP)');ax[1].grid(axis='x',alpha=.2)
    fig.suptitle('Six fixed selections: baseline + five changes',fontsize=14)
    fig.subplots_adjust(left=.23,right=.96,bottom=.18,top=.88,wspace=.15)
    fig.text(.05,.065,'D_out is distance outside the interval between the two single-probe centers, not a tension statistic.\n'
             'smin=80 leaves first xi bin at s=125 under the BAO mask; xi-only curvature sigma(b1) is about 0.54.',fontsize=9)
    pdf.savefig(fig);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,6),sharey=True)
    for ci,c in enumerate(LONG):
        for vi,v in enumerate(VARIANTS):
            yy=ci*1.3+(vi-1)*.23;s=result[c][v]
            for ax,name,index in zip(axes,['b1','fNL'],[1,0]):
                p=s['posterior'][name];m=p['q50']
                ax.errorbar(m,yy,xerr=[[m-p['q16']],[p['q84']-m]],fmt='o',color=COLOR[v],capsize=3,label=DISPLAY[v] if ci==0 else None)
                ax.plot(s['map_theta'][index],yy,'x',color=COLOR[v],ms=7)
    axes[0].set_yticks(np.arange(3)*1.3,[LABEL[c] for c in LONG]);axes[0].invert_yaxis()
    axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.015),ncol=3,frameon=False,fontsize=9)
    for ax,name in zip(axes,[r'$b_1$',r'$f_{\rm NL}$']):ax.set_xlabel(name);ax.grid(axis='x',alpha=.2)
    axes[1].axvline(0,color='.65',ls='--',lw=.7)
    fig.suptitle('Median and 68% interval; crosses mark MAP',fontsize=14)
    fig.subplots_adjust(left=.18,right=.97,bottom=.19,top=.87,wspace=.12)
    a=result['baseline']['joint']['posterior'];b=result['kmax0p06']['joint']['posterior']
    fig.text(.07,.055,f"kmax=0.06 joint width change: b1 {100*(b['b1']['sigma68']/a['b1']['sigma68']-1):+.1f}%; "
             f"fNL {100*(b['fNL']['sigma68']/a['fNL']['sigma68']-1):+.1f}%.\n"
             'Four new chains: 64 walkers x 30,000 steps; 5,000 burn-in. Unchanged single-probe chains reused.',fontsize=9)
    pdf.savefig(fig);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,c,keys,title,ticklabels in [
        (axes[0,0],'kmax0p06',('conditional_at_baseline_MAP','conditional_at_cut_MAP'),'Removed high-k bins',
         ['P0 .062','P0 .070','P0 .078','P2 .062','P2 .070','P2 .078']),
        (axes[0,1],'bao_unmasked',('BAO_conditional_at_old_MAP','BAO_conditional_at_new_MAP'),'Restored BAO bins',
         ['xi0 85','xi0 95','xi0 105','xi0 115','xi2 85','xi2 95','xi2 105','xi2 115'])]:
        rows=paired['cases'][c] if c=='kmax0p06' else scan['cases'][c]
        for key,marker,label in zip(keys,['o','s'],['Baseline MAP','Changed-cut MAP']):
            d=rows[key];r=np.array(d['conditional_residual']);std=np.sqrt(np.diag(d['conditional_covariance']))
            ax.plot(np.arange(len(r)),r/std,'-'+marker,ms=4,label=f"{label}: q={d['chi2_deleted_given_kept']:.3f}")
        ax.axhline(0,color='.6',lw=.7);ax.set_xticks(np.arange(len(ticklabels)),ticklabels,rotation=45,ha='right',fontsize=8)
        ax.set_ylabel('Conditional residual / bin sigma');ax.set_title(title);ax.legend(fontsize=8)
    ax=axes[1,0]
    for i,c in enumerate(list(CASES)[1:]):
        lo,mid,hi=paired['cases'][c]['phase_bootstrap_joint_b1_shift_16_50_84']
        ax.errorbar(mid,i,xerr=[[mid-lo],[hi-mid]],fmt='o',capsize=4,color='#35689a')
    ax.set_yticks(range(5),[LABEL[c] for c in list(CASES)[1:]]);ax.invert_yaxis();ax.axvline(0,color='.6',ls='--')
    ax.set_xlabel('Joint b1 shift from baseline');ax.set_title('Paired 25-phase bootstrap (local response)')
    axes[1,1].axis('off');axes[1,1].text(0,1,
        'How to interpret these diagnostics\n\n'
        'Conditional residuals use the full cross covariance.\n'
        'q is raw C_single Schur chi-square at fixed theta.\n'
        'The plotted residual bins remain correlated.\n\n'
        'High-k bins are not outliers at the baseline MAP.\n'
        'Changing the cut changes degeneracy directions\n'
        'and information, as well as the central values.\n\n'
        'Bootstrap uses a local model and the same 25 phases.\n'
        'It is not a blind validation or a posterior probability.\n'
        'Fit quality cannot be ranked by chi-square values\n'
        'computed on different data vectors.',va='top',fontsize=9.5,linespacing=1.4)
    fig.suptitle('Where does the selection dependence come from?',fontsize=14)
    fig.subplots_adjust(left=.14,right=.97,bottom=.08,top=.89,hspace=.65,wspace=.3)
    pdf.savefig(fig);plt.close(fig)

def run():
    """正式交付必须通过全部预定义gate；保留输入、摘要和代码来源审计。"""
    audit=read(OUT/'postflight.json');assert audit['status']=='pass' and all(audit['gates'].values())
    pre=read(OUT/'expanded_validation.json');assert pre['status']=='pass'
    scan=read(OUT/'map_scan.json');paired=read(OUT/'paired_stability.json')
    DEST.mkdir(parents=True,exist_ok=True);target=DEST/MAIN
    if target.exists():raise FileExistsError('Existing final PDF is immutable')
    result={};ranges=read(common.REFERENCE)['shared_axis_ranges']
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],
        'pdf.fonttype':42,'ps.fonttype':42,'font.size':10,'axes.linewidth':1,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
    with PdfPages(target) as pdf:
        for c in LONG:
            entries={f'ezmock1000_{v}':load(c,v) for v in VARIANTS};result[c]={v:entries[f'ezmock1000_{v}']['summary'] for v in VARIANTS}
            triangle_page(AnnotatedPDF(pdf,c),cov='ezmock1000',cov_display='EZmock1000',chains=entries,ranges=ranges)
            del entries
        diagnostics(pdf,scan,result,paired)
    comparisons={}
    for c,rr in result.items():
        comparisons[c]={}
        for kind in ('MAP','median'):
            b={v:s['map_theta'][1] if kind=='MAP' else s['posterior']['b1']['q50'] for v,s in rr.items()}
            comparisons[c][kind]={'b1':b,'joint_between':min(b['p02'],b['xi02'])<=b['joint']<=max(b['p02'],b['xi02']),
                'outside_distance':max(0,min(b['p02'],b['xi02'])-b['joint'],b['joint']-max(b['p02'],b['xi02']))}
    costs={c:{p:result[c]['joint']['posterior'][p]['sigma68']/result['baseline']['joint']['posterior'][p]['sigma68'] for p in ('b1','fNL')} for c in LONG}
    manifest={}
    for c in CASES:
        e=Engine(c,'joint');nxi=len(e.master['xi_centers']);ids=e.ids;pe=e.master['p_edges'];s=e.master['xi_centers']
        manifest[c]={'selection':CASES[c],'master_dimension':len(e.master['data']),'joint_indices':ids,
            'P0_edges':pe[ids[ids<13]],'P2_edges':pe[ids[(ids>=13)&(ids<22)]-13+4],
            'xi0_centers':s[ids[(ids>=22)&(ids<22+nxi)]-22],
            'xi2_centers':s[ids[ids>=22+nxi]-22-nxi],
            'ndata':{v:scan['cases'][c]['fits'][v]['ndata'] for v in VARIANTS}}
    report={'status':'pass','posterior_results':result,'joint_b1_comparison':comparisons,'joint_sigma68_ratio_to_baseline':costs,
        'map_scan':scan,'numerical_audit':audit,'axis_ranges':ranges,'pdf_sha256':sha(target),'pdf':str(target),
        'scope':'Six fixed cut choices, four new MCMC chains. Same hybrid/full GIC+RIC, EZ1000 C_single full cross covariance, 25-halo mean and priors. No new parameters. Diagnostic cuts do not replace baseline.',
        'source_sha256':sha(__file__)}
    common.save(DEST/'results.json',report);common.save(DEST/'executed_cut_manifest.json',manifest)
    with (DEST/'constraints.csv').open('w') as fp:
        fields=['case','probe','ndata','b1_MAP','raw_chi2','b1_q16','b1_q50','b1_q84','fNL_q16','fNL_q50','fNL_q84']
        writer=csv.DictWriter(fp,fieldnames=fields);writer.writeheader()
        for c in CASES:
            for v in VARIANTS:
                m=scan['cases'][c]['fits'][v];row={'case':c,'probe':v,'ndata':m['ndata'],'b1_MAP':m['theta'][1],'raw_chi2':m['raw_chi2']}
                if c in result:
                    for p in ('b1','fNL'):
                        for key in ('q16','q50','q84'):row[p+'_'+key]=result[c][v]['posterior'][p][key]
                writer.writerow(row)
    lines=['# 少量尺度选择：EZmock1000 / hybrid + full GIC/RIC','',
        '主人~本轮只做基准加5个关键修改，长链只对kmax=0.06和取消BAO mask两组展开，共4条新链，喵～','',
        '**主要结果：kmax=0.06时，joint的b1在MAP和后验中位数上都位于Pk与2PCF之间；取消BAO mask仍未达到这一点，喵～**','',
        '| 配置 | n(P/xi/joint) | P b1 MAP | xi b1 MAP | joint b1 MAP | joint居中(MAP) |','|---|---|---:|---:|---:|---|']
    for c in CASES:
        x=scan['cases'][c];b=x['b1_MAP'];ns='/'.join(str(x['fits'][v]['ndata']) for v in VARIANTS)
        lines.append(f"| {c} | {ns} | {b['p02']:.6f} | {b['xi02']:.6f} | {b['joint']:.6f} | {x['joint_between_MAP']} |")
    lines+=['','三个代表口径的实际MCMC约束如下；区间为原约定的Percival修正68%区间，喵～','',
        '| 配置 | 拟合 | b1中位数 [16%,84%] | fNL中位数 [16%,84%] |','|---|---|---|---|']
    for c in LONG:
        for v in VARIANTS:
            b=result[c][v]['posterior']['b1'];f=result[c][v]['posterior']['fNL']
            lines.append(f"| {c} | {v} | {b['q50']:.6f} [{b['q16']:.6f}, {b['q84']:.6f}] | {f['q50']:.3f} [{f['q16']:.3f}, {f['q84']:.3f}] |")
    k=paired['cases']['kmax0p06'];lo,mid,hi=k['phase_bootstrap_joint_b1_shift_16_50_84']
    lines+=['',
        f"kmax=0.06使joint的sigma68(b1)增加{100*(costs['kmax0p06']['b1']-1):.1f}%，sigma68(fNL)增加{100*(costs['kmax0p06']['fNL']-1):.1f}%，喵～",
        '因此它是本轮最值得保留的敏感性口径；居中同时伴随bias约束变宽，不能据此宣称原问题的物理原因已解决，也没有自动替换9.18基准，喵～','',
        'kmin=0.013和smax=250几乎不移动joint中心；smin=80也会让MAP居中，但原BAO mask使剩余xi第一点变为125，xi-only的局部曲率sigma(b1)约0.54，且sn0 MAP在先验上界，喵～该局部宽度不是后验可信区间，未为此配置运行MCMC，喵～',
        '取消BAO mask主要使xi-only中心降低，joint仍低于两个单项；joint误差变化很小，喵～','',
        '具体bin口径：kmin=0.013移除P0中心0.006、0.008、0.010；kmax=0.06移除P0/P2中心0.062、0.070、0.078，保留最大中心0.054、上边界0.055，喵～',
        'smin=80移除xi0/xi2中心55、65、75；smax=250移除中心255至345；取消BAO mask补回85、95、105、115，各多极共8个点，喵～完整bin边界和向量索引见executed_cut_manifest.json，喵～','',
        f"高k六点在基准MAP处的条件raw chi2={k['conditional_at_baseline_MAP']['chi2_deleted_given_kept']:.6f}，在删点后MAP处变为{k['conditional_at_cut_MAP']['chi2_deleted_given_kept']:.6f}，喵～",
        '原拟合中这些点没有明显条件残差异常；它们对退化方向与参数权重的影响是当前线索，不能直接称为坏点或模型失效点，喵～不同ndata的最小chi2不用于比较拟合优劣，小幅chi2变化不称为显著改善，喵～','',
        f"2000次同相位配对bootstrap的局部joint b1位移为{mid:+.5f}，16%-84%范围[{lo:+.5f},{hi:+.5f}]，喵～",
        f"EZ去均值噪声、同一六口径扫描的最大Dout减小量上尾频率为{paired['EZ_noise_scan_best_reduction_upper_tail']:.6f}，喵～这是固定估计C和局部线性模型下的辅助量，不能换算为物理失配显著性或独立验证，尤其受同一25相位、xi边界和选择过程限制，喵～",'',
        '所有拟合沿用C_single和完整P-xi cross block，没有除以25；配对均值噪声诊断才用25相位均值，喵～各cut从C切子矩阵后重新求逆，并按实际ndata/nparams重算Hartlap和Percival，喵～',
        'P/joint参数为fNL、b1、sigma_s_P、sn0，xi参数为fNL、b1、sn0；全GIC/RIC及既有sn0投影保留，没有新自由参数，喵～',
        '4条新链均64 walkers、30000步、burn-in 5000，split Rhat<1.01、长度>50tau、半链漂移<0.1sigma均通过，喵～未变单项复用旧链，data/C/prediction等价性检查通过，喵～',
        'BAO八点来自全部1000个原EZ测量及25个halo测量，保留原74维data/C和模型回切；扩展模型沿用原fNL/b1网格并检查实际后验随机点、尾部和MAP，喵～',
        f"68个后验探针数值验收全部通过，最大q：{audit['maxima']}，喵～q=delta_model^T C_cut^-1 delta_model；详细阈值、输入hash与gate见postflight.json，喵～",'',
        'PDF第1-3页是基准/kmax=0.06/取消BAO mask的原9.18风格contour；第4页六配置MAP，第5页后验区间，第6页条件残差与配对稳定性，喵～图例按原规则显示fNL MAP与区间，后验中位数另列于表中，喵～',
        '建议先保留kmax=0.06为稳健性结果，与原基准并列；下一步若继续归因，集中检查被删高k P0/P2通过sigma_s_P、sn0与cross covariance作用的方向，无须再扩大cut扫描，喵～','',
        f'运行数据与完整链：`{OUT}`，喵～',
        '重现：task432_scale_bao.py build；task432_scale_selection.py map --cases baseline kmin0p013 kmax0p06 smin80 smax250 bao_unmasked；四个case/variant的sample；task432_scale_diagnostics.py；task432_scale_numerics.py；task432_scale_deliver.py，喵～']
    (DEST/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    for name in ['map_scan.json','paired_stability.json','paired_stability.npz','expanded_validation.json','postflight.json']:
        shutil.copy2(OUT/name,DEST/name)
    sd=DEST/'source';sd.mkdir(exist_ok=True)
    for p in (ROOT/'codes/task432').glob('task432_scale_*.py'):shutil.copy2(p,sd/p.name)
    common.save(DEST/'artifact_manifest.json',{'status':'pass','files':{str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'},'remote_chains':{c:{v:str(chain_dir(c,v)) for v in VARIANTS} for c in LONG}})
    print(json.dumps(common.clean({'status':'pass','PDF':str(target),'comparison':comparisons,'width_ratios':costs})),flush=True)

if __name__=='__main__':run()
