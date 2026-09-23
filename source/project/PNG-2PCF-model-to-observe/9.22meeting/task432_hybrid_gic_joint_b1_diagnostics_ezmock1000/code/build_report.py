#!/usr/bin/env python3
"""只呈现 EZmock1000 的诊断 PDF 与机器可读汇总，原双协方差计算留在溯源目录。

大纲：汇总审计 → 8 页 PDF（基准/交叉/归因/尺度/稳定性/闭合/算子/结论）。
所有误差和概率注明噪声口径；不把 profile 当作边缘后验。
"""
from pathlib import Path
import json,hashlib,shutil,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(os.environ.get('TASK432_DIAG_DIR',str(Path(__file__).resolve().parent)))
DEST=Path(os.environ.get('TASK432_REPORT_DIR',str(HERE.parents[2]/'outputs/hybrid_gic_joint_b1_diagnostics_ezmock1000')))
DEST.mkdir(parents=True,exist_ok=True)
def read(name):
    """读取已完成诊断的 JSON，不读取链或修改科学输入。"""
    return json.loads((HERE/name).read_text())
P=read('primary.json')['covariances']['ezmock1000']
B=read('covariance_bootstrap.json');C=read('calibration.json')
H=read('phases.json')['covariances']['ezmock1000']
R=read('refinement.json')['covariances']['ezmock1000']
O=read('operators.json');A=read('contract_audit.json')
V=['p02','xi02','joint'];LABEL=['P(k)','2PCF','Joint'];COLOR=['#222222','#be3434','#245ab3'];G=['P0','P2','xi0','xi2']
baseline=P['baseline'];bb=np.array([baseline[v]['theta'][1] for v in V]);bt=np.array([r['b1'] for r in B['rows']]);leave=np.array(H['leave_one_b1']);phase=np.array(H['phase_b1'])
gc=C['cases']['ezmock1000_noise_div25'];ec=R['empirical_phase_noise_calibration']
h=lambda n:(1000-n-2)/999

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':10,'axes.linewidth':.9,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
pdfpath=DEST/'task432_hybrid_gic_ezmock1000_joint_b1_diagnostics.pdf'
page=0
def frame(title,subtitle):
    """统一页面标题、正文区域和页脚，为图注保留空间。"""
    fig=plt.figure(figsize=(12,8.5));fig.suptitle(title,x=.075,y=.958,ha='left',fontsize=20,fontweight='bold',color='#17334d')
    fig.text(.075,.905,subtitle,fontsize=11,color='#52616f');return fig
def finish(pdf,fig,note):
    """固定底部图注与页码，不裁切页面。"""
    global page
    page+=1;fig.text(.075,.071,note,fontsize=9,color='#52616f',va='top')
    fig.text(.075,.025,'Task 4.3.2 | hybrid + GIC | EZmock1000 | diagnostic results',fontsize=9,color='#657383')
    fig.text(.925,.025,str(page),ha='right',fontsize=9,color='#657383');pdf.savefig(fig);plt.close(fig)
def dots(ax,values,labels,colors=None):
    """标记 ML 点并逐项打印数值，不绘制未经校准的误差条。"""
    yy=np.arange(len(values));ax.scatter(values,yy,c=colors or COLOR[2],s=45,zorder=3)
    ax.set_yticks(yy,labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    for x,y in zip(values,yy):ax.annotate(f'{x:.4f}',(x,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=10)
    ax.set_xlabel(r'Maximum-likelihood $b_1$')

with PdfPages(pdfpath) as pdf:
    fig=frame('Why does the joint fit prefer a lower b1?','Frozen 9.18 data selection; mandatory hybrid GIC; all main estimates below are maximum likelihood.')
    ax=fig.add_axes([.075,.54,.42,.28]);ax.axis('off')
    rows=[[LABEL[i],f'{bb[i]:.6f}',f'{baseline[v]["theta"][0]:.3f}',f'{h(baseline[v]["ndata"])*baseline[v]["chi2"]:.3f}'] for i,v in enumerate(V)]
    tab=ax.table(cellText=rows,colLabels=['Fit',r'$b_1$',r'$f_{NL}$',r'Hartlap $\chi^2$'],loc='center',cellLoc='center',colWidths=[.26,.26,.25,.23]);tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,2)
    ax=fig.add_axes([.59,.50,.33,.34])
    for v,label,col in zip(V,LABEL,COLOR):
        pp=R['b1_profiles'][v];xx=np.array([r['b1'] for r in pp]);yy=np.array([r['chi2'] for r in pp])-baseline[v]['chi2'];ax.plot(xx,yy*h(baseline[v]['ndata']),color=col,label=label)
    ax.axhline(1,color='.6',ls=':',lw=1);ax.set(xlabel=r'$b_1$',ylabel=r'Profile $\Delta\chi^2$ (Hartlap)',ylim=(0,5),xlim=(2.2,2.6));ax.legend(frameon=False)
    fig.text(.075,.39,'Primary finding',fontsize=14,fontweight='bold',color='#17334d')
    fig.text(.075,.35,'Strong P-xi correlations turn a coherent xi0 residual into a lower shared b1.\nThe P(k) constant term sn0 absorbs part of the corresponding P0 amplitude change.',fontsize=12,linespacing=1.7,va='top')
    fig.text(.075,.23,'Contract: 13 P0 + 9 P2 + 26 xi0 + 26 xi2 bins; kmax=0.08; smin=50; BAO mask 80-120.\nData: mean of 25 Abacus phases. Formal fit covariance: C_single, including the full cross block.\nNo new GSM parameter, no changed formal contour, and no criterion requiring joint b1 to lie between fits.',fontsize=10.5,linespacing=1.6,va='top')
    finish(pdf,fig,'Profile curves re-optimize all other original free parameters; they are not marginalized MCMC posteriors.\nEZmock Hartlap factors depend on data dimension. Percival interval corrections are not applied to these diagnostic profiles.')

    fig=frame('Cross covariance and parameter coupling','Both covariance marginals remain fixed as the cross block is multiplied by lambda.')
    ax=fig.add_axes([.075,.53,.40,.31]);scan=P['cross_scan'];ax.plot([r['lambda'] for r in scan],[r['theta'][1] for r in scan],'o-',color=COLOR[2],lw=2,label='Joint')
    for b,label,col in zip(bb[:2],LABEL[:2],COLOR[:2]):ax.axhline(b,color=col,ls='--',label=label)
    ax.set(xlabel=r'Cross-block multiplier $\lambda$',ylabel=r'ML $b_1$',xticks=np.linspace(0,1,5));ax.legend(frameon=False,loc='lower left')
    ax=fig.add_axes([.65,.53,.26,.31]);vals=[bb[2],P['fixed_P_nuisance']['sigma_at_P']['theta'][1],P['fixed_P_nuisance']['sn_at_P']['theta'][1],P['fixed_P_nuisance']['both_at_P']['theta'][1]]
    dots(ax,vals,['Free','Fix sigma(P)','Fix sn0(P)','Fix both(P)']);ax.set_xlim(2.378,2.44)
    ax=fig.add_axes([.075,.18,.40,.24]);fx=P['fixed_fNL0'];dots(ax,[fx[v]['theta'][1] for v in V],LABEL,COLOR);ax.set_xlim(2.375,2.455);ax.set_title(r'Fix true $f_{NL}=0$')
    ax=fig.add_axes([.59,.18,.33,.24]);prof=R['split_b_profile'];x=[v['delta_b_P_minus_xi'] for v in prof];y=[(v['chi2']-P['split_b']['chi2'])*h(74) for v in prof];ax.plot(x,y,color=COLOR[2]);ax.axvline(0,color='.5',ls='--');ax.axhline(1,color='.6',ls=':');ax.set(xlabel=r'$b_{1,P}-b_{1,\xi}$',ylabel=r'Profile $\Delta\chi^2$',ylim=(0,6),xlim=(-.18,.12))
    finish(pdf,fig,'Cross removal and fixed nuisance values are diagnostics, not proposed fixes. Covariance interpolation remains positive definite.\nSplitting b1 improves raw C_single chi2 by only %.3f; C_single error bars must not be confused with the precision of a 25-phase mean.'%P['split_b']['delta_chi2_vs_shared'])

    fig=frame('Conditional residuals explain the displacement','Linearization is evaluated at the P(k)-only optimum; GIC is included in every derivative.')
    q=P['conditional_at_P_ML'];rho=np.array(q['canonical_correlations']);con=np.array(q['b1_contribution_per_original_bin']);s=np.r_[55,65,75,np.arange(125,350,10)]
    ax=fig.add_axes([.075,.54,.38,.28]);ax.bar(np.arange(1,11),rho[:10],color='#6993c6');ax.set(xlabel='P-xi canonical mode',ylabel='Correlation',ylim=(0,1),xticks=[1,3,5,7,9]);ax.axhline(.9,color='.6',ls=':')
    ax=fig.add_axes([.60,.54,.31,.28]);vals=[q['b1_contribution_by_group'][g] for g in G];ax.bar(G,vals,color=['#333333','#888888','#be3434','#df9999']);ax.axhline(0,color='.4',lw=1);ax.set_ylabel(r'Contribution to linear $\delta b_1$')
    ax=fig.add_axes([.075,.20,.38,.24]);ax.stem(s,con[22:48],basefmt=' ',linefmt='C3-',markerfmt='C3o');ax.axhline(0,color='.5',lw=1);ax.set(xlabel=r'$s\ [h^{-1}{\rm Mpc}]$',ylabel=r'$\xi_0$ bin contribution');ax.axvspan(80,120,color='.8',alpha=.5)
    ax=fig.add_axes([.60,.20,.31,.24]);m=np.array(q['conditional_mode_b1_contribution']);order=np.argsort(abs(m))[::-1][:8];ax.bar([str(i+1) for i in order],m[order],color='#416ca1');ax.set(xlabel='Conditional xi mode (rank by contribution)',ylabel=r'Contribution to $\delta b_1$');ax.axhline(0,color='.5',lw=1)
    fig.text(.075,.115,f'Linear prediction b1 = {q["linear_predicted_theta"][1]:.6f}; optimized joint b1 = {bb[2]:.6f}.  Schur identity error < 1e-8.',fontsize=11,fontweight='bold')
    finish(pdf,fig,'Attribution is a local, correlated decomposition, not a set of independent causal effects. xi0 dominates this decomposition.\nThe conditional modes are combinations of bins; the strongest canonical correlation is %.4f.'%rho[0])

    fig=frame('Which multipoles and scales matter?','Each retained data vector uses the matching covariance submatrix before inversion.')
    ax=fig.add_axes([.19,.56,.28,.28]);keys=['P0_xi0','P02_xi0','P0_xi02','P02_xi02'];dots(ax,[P['multipoles'][k]['theta'][1] for k in keys],['P0 + xi0','P02 + xi0','P0 + xi02','P02 + xi02']);ax.set_xlim(2.378,2.425)
    ax=fig.add_axes([.64,.51,.27,.33]);ss=[50,80,150];kk=[.05,.06,.08];mat=np.array([[P['scales'][f'k{k:.2f}_s{s}']['theta'][1] for k in kk] for s in ss]);im=ax.imshow(mat,cmap='viridis',vmin=2.375,vmax=2.425,aspect='auto');ax.set(xticks=range(3),xticklabels=kk,yticks=range(3),yticklabels=ss,xlabel=r'$k_{max}$',ylabel=r'$s_{min}$',title=r'Joint ML $b_1$')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{mat[i,j]:.4f}',ha='center',va='center',color='white' if mat[i,j]<2.402 else '#111',fontsize=11)
    ax=fig.add_axes([.23,.17,.60,.25]);keys=['xi0_55','xi0_65','xi0_75','xi0_55to75','xi2_55to75','P0_highk','P2_highk'];vals=[R['xi0_deletions'][k]['theta'][1] for k in keys];dots(ax,vals,['Drop xi0: 55','Drop xi0: 65','Drop xi0: 75','Drop xi0: 55,65,75','Drop xi2: 55,65,75','Drop highest 2 P0 bins','Drop highest 2 P2 bins']);ax.axvline(bb[2],color=COLOR[2],ls='--');ax.set_xlim(min(vals+[bb[2]])-.004,max(vals)+.015)
    finish(pdf,fig,'The dashed line is the full-contract joint fit. Changes in fitted b1 after cuts do not by themselves validate those cuts.\nBecause 80 <= s < 120 is masked, smin=80 and smin=100 retain exactly the same bins. k and s cuts are not equivalent Fourier cuts.')

    fig=frame('The shift survives covariance and phase resampling','300 paired-mock covariance bootstraps; 25 phase fits and 25 leave-one-phase-out means.')
    ax=fig.add_axes([.075,.54,.38,.29]);ax.hist(bt[:,2],bins=22,color='#608bc1',alpha=.85);ax.axvline(bb[2],color=COLOR[2],lw=2);ax.axvline(bb[0],color=COLOR[0],ls='--');ax.axvline(bb[1],color=COLOR[1],ls='--');ax.set(xlabel=r'Joint ML $b_1$ after covariance bootstrap',ylabel='Count')
    ax=fig.add_axes([.58,.54,.34,.29]);xx=np.arange(25)
    for i,(col,label) in enumerate(zip(COLOR,LABEL)):ax.plot(xx,leave[:,i],'.-',lw=.8,color=col,label=label)
    ax.set(xlabel='Omitted phase index',ylabel=r'Leave-one-mean ML $b_1$');ax.legend(frameon=False,ncol=3,fontsize=9)
    ax=fig.add_axes([.075,.18,.38,.25]);ax.scatter(phase[:,0],phase[:,2],color=COLOR[2],s=35);mn=min(phase[:,[0,2]].min(),2.15);mx=max(phase[:,[0,2]].max(),2.65);ax.plot([mn,mx],[mn,mx],color='.6',ls='--');ax.set(xlabel=r'Single-phase P(k) $b_1$',ylabel=r'Single-phase joint $b_1$')
    fig.text(.57,.42,'Stability summary',fontsize=14,fontweight='bold',color='#17334d')
    fig.text(.57,.365,f'Bootstrap sigma(b1_joint) = {bt[:,2].std(ddof=1):.4f}\nJoint below both fits: {np.all(bt[:,2,None]<bt[:,:2],axis=1).sum()}/300 bootstraps\nJoint below both fits: {np.all(leave[:,2,None]<leave[:,:2],axis=1).sum()}/25 leave-one means\nSingle-phase residual trace / 74 = {H["whitened_scatter_trace_per_dimension"]:.3f}\nProjected b1 variance / covariance prediction =\n{H["empirical_projected_parameter_covariance"][1][1]/H["assumed_projected_parameter_covariance"][1][1]:.3f}',fontsize=11,linespacing=1.7,va='top')
    finish(pdf,fig,'Bootstrap samples estimate sensitivity to the finite 1000-mock covariance, conditional on this mock ensemble.\nLeave-one means are strongly correlated. Phase fits use the same mean forward operator; the 25-phase covariance is never inverted.')

    fig=frame('Is the observed displacement a usual noise effect?','Fit covariance remains C_single; the noise used to generate a 25-phase mean is explicitly separate.')
    obs=np.array(gc['observed_shift'])
    for j,(case,title) in enumerate([(gc,'Gaussian noise: C_single / 25 (500 draws)'),(ec,'Centered 25-phase bootstrap (1000 draws)')]):
        shift=np.array([r['shift'] for r in case['rows']]);ax=fig.add_axes([.075+.505*j,.52,.35,.30]);ax.scatter(shift[:,0],shift[:,1],s=7,alpha=.28,color='#658bb9');ax.scatter(*obs,c='#bd2525',marker='*',s=210,zorder=4,label='Observed mean');ax.axhline(0,color='.65',lw=.8);ax.axvline(0,color='.65',lw=.8);ax.set(xlabel=r'$b_{1,J}-b_{1,P}$',ylabel=r'$b_{1,J}-b_{1,\xi}$',title=title);ax.legend(frameon=False,loc='lower right',fontsize=9)
        ax=fig.add_axes([.075+.505*j,.17,.35,.22]);chi=np.array([r['chi2_joint'] for r in case['rows']]);ax.hist(chi,bins=28,color='#6e92be');ax.axvline(baseline['joint']['chi2'],color='#bd2525',lw=2,label='Observed');ax.set(xlabel=r'Raw $C_{single}$ joint $\chi^2$',ylabel='Count');ax.legend(frameon=False,fontsize=9)
        n=len(case['rows']);count=int(round(case['empirical_2d_tail_fraction']*(n+1)-1));tag=' (resolution limited)' if count==0 else ''
        fig.text(.075+.505*j,.445,f'2D tail: {count}/{n} draws{tag}',fontsize=10)
        fig.text(.075+.505*j,.421,f'Joint below both fits: {case["outside_low_fraction"]:.1%}',fontsize=10)
    finish(pdf,fig,'Tail statistic uses both correlated ML differences. Empirical frequencies include +1 correction and finite Monte Carlo resolution.\nGaussian mean noise assumes independent phases and the chosen covariance. Phase bootstrap has only 25 input phases; neither test proves model correctness.')

    fig=frame('Observation-operator checks and remaining gaps','Shell averaging is quantified; missing phase windows and differing conventions remain explicit.')
    ax=fig.add_axes([.075,.58,.46,.23]);ax.axis('off');shell=O['fits']['ezmock1000']['shell_xi'];rows=[[LABEL[i],f'{bb[i]:.6f}',f'{shell[v]["theta"][1]:.6f}',f'{shell[v]["theta"][1]-bb[i]:+.6f}'] for i,v in enumerate(V)]
    tab=ax.table(cellText=rows,colLabels=['Fit','Bin center','Shell average','Delta b1'],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(10.5);tab.scale(1,1.8)
    fig.text(.60,.81,'Verified input agreement',fontsize=14,fontweight='bold',color='#17334d')
    fig.text(.60,.76,'25/25 phases: same FKP summary and zeff\nFKP P0 = 10000 on both sides\nP window: only ph000 exists\nP LOS: local; xi LOS: midpoint',fontsize=11,linespacing=1.9,va='top')
    ax=fig.add_axes([.075,.20,.43,.27]);dm=np.array(A['lowk_estimates'][0]['delta_model_remove_lowk']);ax.plot(s,dm[22:48],label=r'$\Delta\xi_0$',color=COLOR[1]);ax.plot(s,dm[48:],label=r'$\Delta\xi_2$',color=COLOR[2]);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.set(xlabel=r'$s\ [h^{-1}{\rm Mpc}]$',ylabel='Remove linear low-k modes',title=r'Leading IR estimate at $f_{NL}=0$');ax.legend(frameon=False)
    est=A['lowk_estimates'][0];fig.text(.60,.43,'Low-k support audit',fontsize=14,fontweight='bold',color='#17334d')
    fig.text(.60,.38,f'P theory cutoff: kbox = {A["P_window_theory_kmin"]:.6f}\nHybrid FFT starts at k = {A["hybrid_internal_k_range"][0]:.0e}\nLinear-only IR response: Delta b1 =\n{est["linear_parameter_response"][1]:+.5f} (matched scalar GIC included)\nThis is not a full nonlinear finite-box calculation.',fontsize=11,linespacing=1.8,va='top')
    finish(pdf,fig,'Shell average uses r^2 volume weights across 10 Mpc/h bins, not a measured RR(s,mu) operator; corrected xi0 still contains GIC.\nNo average of 25 P windows is claimed. LOS differences are documented conventions, not a demonstrated source of the observed shift.')

    fig=frame('What the diagnostics establish','EZmock1000 is the sole covariance used for the conclusions and follow-up recommendations.')
    left=[('Established mechanism','Cross correlations amplify a coherent xi0 residual.\nA local linear response reproduces the joint b1 shift.'),('Important coupling','Fixing sn0 at the P-only optimum moves joint b1 upward.\nFixing fNL=0 or sigma_s_P alone does not remove the shift.'),('Robustness checks','The effect survives 300 covariance bootstraps and all 25\nleave-one means. Shell averaging has a small effect.')]
    for i,(heading,body) in enumerate(left):
        y=.80-.19*i;fig.text(.075,y,heading,fontsize=14,fontweight='bold',color='#17334d');fig.text(.075,y-.047,body,fontsize=11,linespacing=1.6,va='top')
    right=[('Interpretation','Outside both marginal optima is allowed for correlated fits.\nFor this 25-phase mean, closure tests still flag a residual\nworth investigating; formal C_single intervals hide this precision.'),('Next targeted work','Audit matched P0/xi0 broadband and shot-noise response,\nusing the identified scale ranges and conditional modes.\nComplete the P mean-window and finite-box operator checks.'),('Avoid premature fixes','Do not tune the covariance, fix sn0 by preference, or choose\nscale cuts merely to put joint b1 between the two fits.\nKeep the formal hybrid+GIC baseline frozen.')]
    for i,(heading,body) in enumerate(right):
        y=.80-.22*i;fig.text(.56,y,heading,fontsize=14,fontweight='bold',color='#17334d');fig.text(.56,y-.047,body,fontsize=10.5,linespacing=1.65,va='top')
    finish(pdf,fig,'Validation: frozen ML reproduced; Schur identity checked; covariance rebuilt from all 1000 mocks; paired 25-phase mean reconstructed.\nSame-model noisefree closure recovers injected parameters. Numerical shell-interpolation errors are negligible at C_single/25 precision.')

def serial(obj):
    """将 NumPy 值转换为 JSON 原生值。"""
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    raise TypeError(type(obj))
summary={'covariance':'ezmock1000','estimator':'maximum likelihood unless explicitly stated','baseline':baseline,'cross_scan':P['cross_scan'],'fixed_fNL0':P['fixed_fNL0'],'fixed_P_nuisance':P['fixed_P_nuisance'],'split_b':P['split_b'],'split_f':P['split_f'],'multipoles':P['multipoles'],'scales':P['scales'],'conditional':P['conditional_at_P_ML'],'deletions':R['xi0_deletions'],'shell_fits':O['fits']['ezmock1000'],'bootstrap':{'n':len(bt),'joint_b1_mean':bt[:,2].mean(),'joint_b1_std':bt[:,2].std(ddof=1),'joint_b1_16_50_84':np.percentile(bt[:,2],[16,50,84]),'below_both_count':np.all(bt[:,2,None]<bt[:,:2],axis=1).sum()},'leave_one':{'n':25,'joint_b1_range':[leave[:,2].min(),leave[:,2].max()],'below_both_count':np.all(leave[:,2,None]<leave[:,:2],axis=1).sum()},'closure_Gaussian_mean25':{k:v for k,v in gc.items() if k!='rows'},'closure_empirical_phase_mean25':{k:v for k,v in ec.items() if k!='rows'},'operator_contract_audit':A,'pdf_sha256':hashlib.sha256(pdfpath.read_bytes()).hexdigest(),'pages':page}
(DEST/'summary.json').write_text(json.dumps(summary,indent=2,default=serial)+'\n')
print(json.dumps({'pdf':str(pdfpath),'pages':page,'summary':str(DEST/'summary.json')}))
