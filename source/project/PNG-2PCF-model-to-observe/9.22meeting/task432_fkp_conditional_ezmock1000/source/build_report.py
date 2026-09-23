#!/usr/bin/env python3
"""将独立模式/FKP配对结果画成PDF；不生成PNG或新的正式contour。

页序：低维cov校准与均值残差 -> FKP对b1位移的响应 -> 四分量观测/模型增量
-> 数值检查和适用范围。所有数值从最终json/npz读取，不手动复制拟合结果。
"""
from pathlib import Path
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse

BASE=Path(__file__).resolve().parents[3]/'outputs/task432_fkp_conditional_ezmock1000'
plt.rcParams.update({'font.size':10,'axes.labelsize':11,'axes.titlesize':12,'figure.dpi':110,
                     'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def footer(fig,page,text):
    """统一脚注，明确每页是诊断且保持科学口径可见。"""
    fig.text(.07,.035,text,fontsize=8,color='#555555',va='bottom')
    fig.text(.94,.035,str(page),ha='right',fontsize=9)

def ellipse(ax,cov,center=(0,0),prob=.68,**kwargs):
    """二维Gaussian概率椭圆，明确使用2自由度半径而非1D sigma。"""
    v,q=np.linalg.eigh(cov);angle=np.degrees(np.arctan2(q[1,1],q[0,1]));r=np.sqrt(-2*np.log(1-prob))
    ax.add_patch(Ellipse(center,2*r*np.sqrt(v[1]),2*r*np.sqrt(v[0]),angle=angle,fill=False,**kwargs))

def main():
    """生成最终四页报告，并记录页面尺寸与输入来源用于后续视觉验收。"""
    modes=json.loads((BASE/'conditional_modes.json').read_text())
    covariance_test=json.loads((BASE/'lowdim_covariance_calibration.json').read_text())
    fkp=json.loads((BASE/'paired_fkp_results.json').read_text())
    with np.load(BASE/'conditional_modes.npz') as a:mode_arrays={k:a[k] for k in a.files}
    with np.load(BASE/'paired_fkp_results.npz') as a:paired={k:a[k] for k in a.files}
    target=BASE/'task432_fullRIC_FKP_conditional_diagnostics.pdf'
    with PdfPages(target) as pdf:
        fig,axes=plt.subplots(1,2,figsize=(11.7,6.8));fig.subplots_adjust(left=.08,right=.96,bottom=.22,top=.80,wspace=.30)
        fig.suptitle('Full-RIC joint bias: conditional variance and mean residuals',y=.93,fontsize=16)
        ax=axes[0];x=np.arange(5);q=np.array(modes['EZ_25sample_variance_95_interval']);ratios=np.array(modes['halo_variance_ratios'])
        ax.vlines(x,q[0],q[1],color='#bdccd7',lw=12,label='95% range from 25-EZ resamples')
        ax.scatter(x,ratios,c='#d55e00',s=55,zorder=4,label='25 halo phases')
        ax.axhline(1,c='#555555',ls='--',lw=1);ax.set_xticks(x,['J-P','J-X (orth.)','J b1 (orth.)','Cond. 1','Cond. 2'],rotation=20)
        ax.set_ylabel('Sample variance / EZ prediction');ax.set_ylim(0,2);ax.legend(fontsize=8,loc='upper left');ax.set_title('Modes defined before inspecting halo residuals')
        for a,b in zip(x,ratios):ax.text(a,b+.055,f'{b:.2f}',ha='center',fontsize=9)
        ax=axes[1];cov=np.array(modes['contrast_covariance_Csingle'])/25;mean=np.array(modes['mean_b1_contrasts_linear'])
        ellipse(ax,cov,prob=.95,color='#0072b2',ls='--',label='EZ mean noise: 95%')
        ellipse(ax,cov,prob=.68,color='#0072b2',label='EZ mean noise: 68%')
        ax.scatter(0,0,c='#222222',marker='+',s=60);ax.scatter(*mean,c='#d55e00',s=65,label='Observed mean (linear response)')
        ax.axhline(0,c='#dddddd',lw=.8);ax.axvline(0,c='#dddddd',lw=.8)
        ax.set_xlabel(r'$b_{1,\rm joint}-b_{1,P}$');ax.set_ylabel(r'$b_{1,\rm joint}-b_{1,\xi}$')
        ax.set_xlim(-.065,.055);ax.set_ylim(-.07,.06);ax.legend(fontsize=8,loc='upper left')
        ax.set_title(f"Local mean-noise tail = {modes['EZ_bootstrap_mean_contrast_tail']:.3f}")
        p2=covariance_test['subspaces']['2']['EZ_resampling_upper_tail'];p5=covariance_test['subspaces']['5']['EZ_resampling_upper_tail']
        footer(fig,1,f'Whole covariance-shape resampling tails: 2 modes {p2:.2f}; 5 modes {p5:.2f} (includes cross-mode correlations).\nC_single/25 is used only for mean-noise calibration. Linear contrasts are not posterior-tension significances.')
        pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(11.7,6.8));fig.subplots_adjust(left=.08,right=.96,bottom=.24,top=.81,wspace=.3)
        fig.suptitle('FKP-only paired experiment: independent minus own n(z)',y=.93,fontsize=16)
        R=fkp['records'];x=np.arange(len(R));colors=['#0072b2','#d55e00']
        current=modes['current_formal_MAP'];needed=np.array([current['p02']['b1']-current['joint']['b1'],current['xi02']['b1']-current['joint']['b1']])
        for j,ax in enumerate(axes):
            y=np.array([r['b1_contrast_net_response'][j] for r in R]);ax.axhline(0,c='#888888',lw=1)
            ax.axhline(needed[j],c='#777777',ls='--',label='Shift needed to remove current MAP gap')
            ax.scatter(x,y,c=colors[j],s=55,zorder=5,label='Data minus ordinary-window and full-IC change')
            for i,r in enumerate(R):
                pts=np.array(r['xi_block_contrast_responses'])[:,j]
                ax.scatter(np.repeat(i,len(pts)),pts,s=18,facecolors='none',edgecolors=colors[j],alpha=.5)
            mean=fkp['mean_b1_contrast_response'][j];sem=fkp['phase_SEM_b1_contrast_response'][j]
            ax.errorbar(len(R),mean,yerr=sem,fmt='s',c='#222222',capsize=4,label='Pilot mean +/- phase SEM')
            ax.set_xticks(np.arange(len(R)+1),[r['phase'] for r in R]+['mean'],rotation=15)
            ax.set_ylabel(r'$\Delta(b_{1,\rm joint}-b_{1,'+('P' if j==0 else r'\xi')+r'})$')
            ax.set_title('Joint relative to '+('P-only' if j==0 else 'xi-only'));ax.legend(fontsize=7.5,loc='upper left')
            ax.set_ylim(min(-.012,float(y.min())-.007),max(.056,float(y.max())+.01))
        footer(fig,2,'Same data/random positions, per-branch I2 and Poisson subtraction, recomputed P windows and full IC.\nOpen dots show common-random block sensitivity. Pilot SEM is not an independent 25-phase uncertainty.')
        pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(11.7,8.3));fig.subplots_adjust(left=.09,right=.96,bottom=.18,top=.85,hspace=.38,wspace=.3)
        fig.suptitle('Mean paired increments in the four fitted components',y=.94,fontsize=16)
        sigma=np.sqrt(np.diag(mode_arrays['covariance']));d=paired['mean_delta_data'];m=paired['mean_delta_model']
        groups=[('P0',slice(0,13)),('P2',slice(13,22)),('xi0',slice(22,48)),('xi2',slice(48,74))]
        for ax,(label,sl) in zip(axes.flat,groups):
            x=np.arange(len(d[sl]));ax.axhline(0,c='#888888',lw=.8)
            ax.plot(x,d[sl]/sigma[sl],'.-',c='#0072b2',label='Measured change')
            ax.plot(x,m[sl]/sigma[sl],'.-',c='#009e73',label='Ordinary window + full IC change')
            ax.plot(x,(d-m)[sl]/sigma[sl],'.-',c='#d55e00',label='Residual change')
            ax.set_title(label);ax.set_xlabel('Frozen fit-bin index');ax.set_ylabel(r'$\Delta d_i / \sigma_{i,\rm single}$');ax.grid(alpha=.15)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',
                   bbox_to_anchor=(.52,.901),ncol=3,frameon=False,fontsize=9)
        footer(fig,3,'The displayed diagonal sigma is a plotting scale; all response calculations use the full 74x74 covariance.\nThe unchanged 9.18 selection includes the BAO mask. These are paired increments, not new data or constraints.')
        pdf.savefig(fig);plt.close(fig)

        fig=plt.figure(figsize=(11.7,8.3));fig.suptitle('Numerical checks and interpretation limits',y=.94,fontsize=16)
        ax=fig.add_axes([.08,.54,.84,.28]);ax.axis('off')
        rows=[]
        for r in R:
            rows.append([r['phase'],str(r['n_xi_blocks']),f"{r['normalizations']['independent']/r['normalizations']['own']:.6f}",
                         f"{r['IC_difference_error_chi2_Csingle']:.3g}",f"{r['b1_net_response_P_xi_joint'][2]:+.5f}"])
        table=ax.table(cellText=rows,colLabels=['Phase','LS blocks','I2 independent / own','IC scramble difference: q','Local joint b1 shift'],loc='center',cellLoc='center')
        table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2)
        lines=[
            'Validated: same positions and frozen bins; analytic Poisson subtraction; independent differential IC kernels.',
            'Mode construction: fixed fiducial derivatives and EZ covariance; no halo-residual-based selection.',
            'Covariance: all 1000 original paired EZ vectors retained; no 74-D halo sample-covariance inversion.',
            'Mean-noise interpretation assumes independent halo phases; finite mock covariance uncertainty remains.',
            'FKP reference uses the other 24 phases. Shared reference tables make phase differences correlated.',
            'The pilot uses 4N randoms for P and the listed common N-sized LS blocks, not production 25-block xi.',
            'Model increments are evaluated at one fiducial. Constant-increment MAP is diagnostic only.',
            'No new GSM parameter, RIC amplitude, formal covariance, or posterior is introduced.'
        ]
        for i,t in enumerate(lines):fig.text(.09,.48-.041*i,t,fontsize=10,va='top')
        footer(fig,4,'Source numbers: conditional_modes.json and paired_fkp_results.json.\nExisting formal full-RIC results remain the recommended constraints; this PDF reports targeted diagnostics.')
        pdf.savefig(fig);plt.close(fig)
    (BASE/'plot_manifest.json').write_text(json.dumps({'pdf':str(target),'pages':4,'format':'PDF only','inputs':['conditional_modes.json','conditional_modes.npz','paired_fkp_results.json','paired_fkp_results.npz']},indent=2)+'\n')
    print(target)

if __name__=='__main__':main()
