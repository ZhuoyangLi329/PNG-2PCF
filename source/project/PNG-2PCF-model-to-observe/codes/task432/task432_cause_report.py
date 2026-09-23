#!/usr/bin/env python3
"""Reproducible PDF-only report for the first targeted cause-test batch."""
import task432_joint_b1_cause_tests as c
from task432_joint_b1_cause_tests import np,json,Path,save,sha
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import fill

COL={'p02':'#3274a1','xi02':'#e38d27','joint':'#c03949'}
LABEL={'p02':r'$P_0+P_2$','xi02':r'$\xi_0+\xi_2$','joint':'Joint'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42,'savefig.facecolor':'white'})

def page(title,n):
    fig=plt.figure(figsize=(13.5,8.8));fig.suptitle(title,x=.07,ha='left',fontsize=19,fontweight='bold',y=.95)
    fig.text(.07,.9,'EZmock1000 | 25-phase mean data | frozen 74 bins | full cross covariance | ML diagnostics',color='#555',fontsize=10)
    fig.text(.07,.025,'Task 4.3.2 - hybrid + GIC cause tests | All chi-square values use raw C_single; no new formal posterior.',fontsize=9,color='#555')
    fig.text(.94,.025,str(n),ha='right',fontsize=9);return fig

def note(fig,text,y=.085):fig.text(.07,y,text,fontsize=10,va='bottom',linespacing=1.5)

def main():
    c.initialize();out=c.OUT
    read=lambda name:json.loads((out/name).read_text())
    shot=read('shot_response.json');ric=read('ric_existing_probe.json');com=read('common_theory_cross_injection.json');win=read('window_pilot.json');contact=read('shot_contact_injection.json');audit=read('common_theory_numerical_audit.json')
    baseline=shot['fits']['current_window_cut'];box=com['variants']['boxcut_basis'];cont=com['variants']['continuous_basis'];wmean=win['variants']['three_phase_pilot_mean']
    candidates=[('Hybrid + GIC baseline',baseline),('Observed contact sn0',shot['fits']['observed_contact']),('Legacy P0-only RIC probe',ric),('Common Kaiser + FoG / box cut',box),('Three-phase window pilot',wmean)]
    comparison=[]
    for label,v in candidates:
        comparison.append({'label':label,**{q:r['theta'][1] for q,r in v['fits'].items()},'joint_chi2_Csingle':v['fits']['joint']['chi2_Csingle'],'joint_sn0':v['fits']['joint']['theta'][3]})
    save(out/'comparison_summary.json',{'estimates':'Maximum likelihood, not posterior medians','rows':comparison,'formal_model_updated':False})
    pdf=out/'hybrid_gic_joint_b1_cause_tests_ezmock1000.pdf'
    with PdfPages(pdf) as book:
        fig=page('Which effects can move the joint bias?',1)
        ax=fig.add_axes([.31,.38,.61,.44]);y=np.arange(len(candidates))
        for q,offset in zip(COL,[-.15,0,.15]):ax.scatter([v['fits'][q]['theta'][1] for _,v in candidates],y+offset,c=COL[q],label=LABEL[q],s=55,zorder=3)
        ax.set_yticks(y,[v[0] for v in candidates]);ax.invert_yaxis();ax.set_xlabel(r'Best-fit $b_1$ (ML diagnostic points, not posterior intervals)');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.14),ncol=3,frameon=False)
        t=fig.add_axes([.31,.19,.61,.095]);t.axis('off')
        table=t.table(cellText=[[f'{v["joint_chi2_Csingle"]:.4f}' for v in comparison],[f'{v["joint_sn0"]:.4f}' for v in comparison]],rowLabels=[r'Joint $\chi^2$',r'Joint $sn_0$'],colLabels=['Baseline','Contact','RIC probe','Common theory','Window pilot'],loc='center');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.4)
        note(fig,'RIC probe: a substantial bias shift with essentially unchanged fit quality (delta chi-square = +0.093).\nIt is an incomplete P0-only approximation, so the formal hybrid + GIC result is retained.');book.savefig(fig);plt.close(fig)

        fig=page('Shot response: metadata and estimator injection both close',2)
        ax=fig.add_axes([.08,.46,.4,.34]);n=np.arange(22)
        cur=np.array(shot['templates']['current_window_cut'])
        for name,label in [('window_no_kcut','Window without k-box cut'),('observed_contact','Observed contact')]:ax.plot(n,(np.array(shot['templates'][name])-cur)/1e4,'.-',label=label)
        ax.axvline(12.5,c='gray',ls=':');ax.set(xlabel='Selected P bin index (P0: 0-12; P2: 13-21)',ylabel=r'$(S_{\rm alt}-S_{\rm base})/10^4$');ax.legend(fontsize=9)
        ax=fig.add_axes([.57,.46,.36,.34])
        with np.load(out/'shot_contact_injection.npz') as z:
            ax.plot(z['k'],z['delta_P0']/z['expected_P0'],'-',label='Measured P0 / predicted contact');ax.plot(z['k'],z['delta_P2']/z['expected_P0'],label='P2 leakage / predicted contact')
        ax.set(xlabel=r'$k$ [$h\,\mathrm{Mpc}^{-1}$]',ylabel='Injected response ratio');ax.legend(fontsize=9);ax.set_ylim(-.08,1.12)
        maxerr=max(r['shot_relative_error'] for r in shot['metadata_audit'])
        note(fig,f'All 25 phase metadata: maximum shot-normalization relative error = {maxerr:.2e}.\n'
             f'Coincident half-weight pair test: P0 relative error = {contact["monopole_relative_error"]:.2e}; P2 leakage = {contact["quadrupole_relative_leakage"]:.2e}.\n'
             'The painted field is unchanged; only the analytically subtracted self-pair term changes.\n'
             'Replacing the fitted sn0 template shifts joint b1 by only -0.000112.\n\n'
             'This validates the contact implementation. It does not prove that halo stochasticity is scale independent.',y=.17);book.savefig(fig);plt.close(fig)

        fig=page('Cross-model injections: the direction of mismatch matters',3)
        labels=['b=2.35','b=2.42','b=2.49','sigma=5','fNL=-30','fNL=+30'];xx=np.arange(6)
        for i,(key,title) in enumerate([('coherent_truth_fitted_by_current','Common theory truth -> current hybrid fit'),('current_composite_fitted_by_coherent','Current composite truth -> common theory fit')]):
            ax=fig.add_axes([.08+i*.46,.38,.38,.43])
            for q in COL:
                vals=[r[key][q]['theta'][1]-r['truth'][1] for r in box['injections']];ax.plot(xx,vals,'o-',color=COL[q],label=LABEL[q])
            ax.axhline(0,color='gray',lw=1);ax.set_xticks(xx,labels,rotation=30,ha='right');ax.set_title(title,fontsize=12);ax.set_ylabel(r'$b_1^{\rm fit}-b_1^{\rm truth}$');ax.legend(fontsize=9)
        note(fig,'Matched clustering control: common P_L, PNG response, Kaiser x Lorentzian^2 FoG, k-box support and xi GIC.\n'
             'Forward injection: all six tested cases put joint b1 between P and xi; the observed ordering is not reproduced.\n'
             'Reverse injection can put joint below both. This is evidence of directional model sensitivity, not a preferred truth.\n'
             'Self-recovery passes at all anchors; highest integration/interpolation discrepancy is < 5e-9 in chi-square(C_mean).\n'
             'Noiseless controls only. Common-theory xi has a shared sigma parameter; joint still has four parameters.',y=.12);book.savefig(fig);plt.close(fig)

        fig=page('Common theory on the measured data does not resolve the offset',4)
        ax=fig.add_axes([.08,.44,.39,.37]);rr=[baseline,cont,box];xx=np.arange(3)
        for q in COL:ax.plot(xx,[r['fits'][q]['theta'][1] for r in rr],'o-',label=LABEL[q],color=COL[q])
        ax.set_xticks(xx,['Hybrid + GIC','Continuous xi control','Matched k-box control'],rotation=12);ax.set_ylabel(r'Best-fit $b_1$');ax.legend()
        ax=fig.add_axes([.57,.44,.35,.37]);b0=baseline['fits']['joint'];ts=[]
        for v in rr:ts.append(v['fits']['joint']['chi2_Csingle'])
        ax.bar(xx,ts,color=['#68798a','#9ec2d3','#3274a1']);ax.set_xticks(xx,['Hybrid','Continuous','k-box']);ax.set_ylabel(r'Joint raw $\chi^2(C_{\rm single})$')
        for x,y in zip(xx,ts):ax.text(x,y+.08,f'{y:.4f}',ha='center')
        ax.set_ylim(0,max(ts)*1.16)
        note(fig,'The primary common-theory control is the k-box branch; the continuous-xi branch leaves a support mismatch.\n'
             'Both controls move the real-data joint b1 farther down. A common simple Kaiser model is not a successful repair.\n'
             'This test does not establish that the frozen hybrid is physically complete; it tests one specific mismatch.\n\n'
             'The shared sigma is already present in P and joint; xi-only has three free parameters here versus two in hybrid.\n'
             'The P residual stochastic contact remains a separate nuisance and is not Hankel-transformed as clustering.',y=.12);book.savefig(fig);plt.close(fig)

        fig=page('Phase-window pilot at the production estimator settings',5)
        ax=fig.add_axes([.09,.42,.39,.38]);names=list(win['variants']);xx=np.arange(len(names))
        for q in COL:ax.plot(xx,[win['variants'][n]['fits'][q]['theta'][1] for n in names],'o-',color=COL[q],label=LABEL[q])
        ax.set_xticks(xx,['ph000','ph012','ph024','3-phase mean'],rotation=15);ax.set_ylabel(r'Best-fit $b_1$');ax.legend()
        ax=fig.add_axes([.57,.42,.35,.38]);diag=np.sqrt(np.diag(c.COV))
        for name in names[1:]:ax.plot(np.arange(22),np.array(win['variants'][name]['changes_at_same_theta']['joint']['delta_model'])[:22]/diag[:22],label=name.replace('three_phase_pilot_mean','3-phase mean'))
        ax.axvline(12.5,color='gray',ls=':');ax.set(xlabel='Selected P bin index',ylabel=r'$\Delta m/\sigma_{\rm single}$ at baseline joint ML');ax.legend(fontsize=9)
        err=max(r['measurement_relative_l2'][q] for r in win['reproduction_gates'] for q in ('pk0','pk2'))
        db=wmean['fits']['joint']['theta'][1]-baseline['fits']['joint']['theta'][1]
        drift=win['variants']['ph000']['fits']['joint']['theta'][1]-baseline['fits']['joint']['theta'][1]
        phase_only=win['within_current_implementation_joint_b1_shifts']['three_phase_pilot_mean']
        note(fig,f'P0/P2 data reproduce (max relative error {err:.2e}); the historical ph000 window does NOT reproduce exactly.\n'
             f'Independent current ph000 repeat agrees to relative {win["current_repeat_relative_error"]:.2e}.\n'
             f'Window recomputation drift alone: delta b1 = {drift:+.6f}; 3-phase mean versus current ph000: {phase_only:+.6f}.\n'
             f'Total 3-phase-pilot shift versus frozen historical window: {db:+.6f}.\n'
             'The current library was modified after the historical file; historical library sources were not archived here.\n'
             'This is not the full 25-phase average. The historical implementation drift is not a phase effect.',y=.115);book.savefig(fig);plt.close(fig)

        fig=page('RIC: a sensitive direction, with an incomplete existing operator',6)
        ax=fig.add_axes([.09,.47,.4,.34]);xx=np.arange(2)
        for q in COL:ax.plot(xx,[baseline['fits'][q]['theta'][1],ric['fits'][q]['theta'][1]],'o-',color=COL[q],label=LABEL[q])
        ax.set_xticks(xx,['Hybrid + GIC','Legacy P0-only probe']);ax.set_ylabel(r'Best-fit $b_1$');ax.legend()
        fig.text(.55,.79,'What this probe actually does',fontweight='bold',fontsize=13)
        fig.text(.55,.74,'Theory ell=0 -> observed P0 only.\nP2, xi0 and xi2 receive no RIC term.\nNo new fit parameter.\n\nJoint b1: 2.3832 -> 2.4047\nJoint raw chi-square: 4.5191 -> 4.6118\nFit quality is essentially unchanged.',va='top',linespacing=1.65,fontsize=12)
        fig.text(.09,.35,r'Full field projection: $C_{\rm rad}=(I-\Pi_r)C(I-\Pi_r)^T$',fontsize=15)
        fig.text(.09,.30,r'$C_{\rm rad}=C-\Pi_r C-C\Pi_r^T+\Pi_r C\Pi_r^T$',fontsize=15)
        note(fig,'The legacy approximation subtracts the radial auto contribution as an approximation to the net correction.\n'
             'The two cross terms are not evaluated; its accuracy for this geometry and joint vector is unvalidated.\n'
             'A complete radial projection includes the global mean constraint when their weights are consistent.\n'
             'Therefore, do not blindly add full RIC on top of the current scalar GIC prescription.\n'
             'Next priority: paired selection-function closure and a consistent P/xi RIC response, with FKP held fixed.\n'
             'Reference: de Mattia & Ruhlmann-Kleider (2019), arXiv:1904.08851, equations (6), (20), section 2.4.',y=.1);book.savefig(fig);plt.close(fig)
    save(out/'report_audit.json',{'pdf':str(pdf),'pdf_sha256':sha(pdf),'pages':6,'PDF_only':True,'numerical_audit_pass':audit['gate'],'contact_injection_pass':contact['gate'],'all_candidate_ML_success':all(r['success'] for _,v in candidates for r in v['fits'].values()),'formal_inputs_unchanged':all(sha(p)==h for p,h in read('input_manifest.json')['inputs'].items()),'RIC_fit_quality_interpretation':'Essentially unchanged: delta raw chi-square +0.0927 is not evidence of meaningful deterioration.'})
    print(pdf,flush=True)

if __name__=='__main__':main()
