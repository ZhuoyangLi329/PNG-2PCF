#!/usr/bin/env python3
"""9.18 数据/模型口径、更新 EZmock-1000 协方差的 hybrid 对照，复用已验证的公共采样/绘图实现。

执行大纲
--------
1. 从上一轮冻结输入逐数组复制数据、P 模型、hybrid 网格，只替换经验协方差。
2. 核对 1000 条 mock 的样本协方差、正定性及新度量下的插值误差。
3. 使用 Hartlap 修正后的 Gaussian 似然重跑三组 64×30000 长链。
4. 复用直接 GSM 后验区域核对及 ML 优化；按实际自由参数数目作 Percival 修正。
5. 图和 JSON 一致采用修正区间；原始链保留，修正只用于区间和展示。
"""
import task432_hybrid_jaxpower_0918_contract as common
from task432_hybrid_jaxpower_0918_contract import (np,Path,json,os,socket,argparse,sha,save,log)
import copy
import shutil

ROOT=common.ROOT
JAX_OUT=common.OUT
OUT=ROOT/'outputs/task43_outputs/rsd_validation/task432_model_repair/hybrid_ezmock1000_0918_contract_v1'
MEETING=ROOT/'9.22meeting/task432_hybrid_ezmock1000_9.18contract'
PRODUCTION=ROOT/'outputs/task43_outputs/ezmock_rsd_lightcone_covariance_z0p4_0p8_x1000_common50_nzmatch_b0.30_z0.60'
MANIFEST=PRODUCTION/'manifests/task43_ezmock_rsd_covariance_x1000_fixampF_nzmatch_b0.30_z0.60.jsonl'
COV_SOURCE=OUT/'ezmock1000_covariance_and_stack.npz'
common.OUT=OUT
common.MEETING=MEETING
common.REFERENCE_COV=None  # 1000-mock covariance 与旧 604 不同，不套用旧后验的复现门限。
common.SEED_BASE=1000918

def factors(nb,npar):
    """输入数据维数 nb 和拟合参数数目 npar，返回 1000 mocks 的原生产有限样本修正。"""
    ns=1000
    hartlap=(ns-nb-2.)/(ns-1.)
    a=2./((ns-nb-1.)*(ns-nb-4.))
    b=(ns-nb-2.)/((ns-nb-1.)*(ns-nb-4.))
    m1=(1.+b*(nb-npar))/(1.+a+b*(npar+1.))
    return {'nmock':ns,'ndata':nb,'nparams':npar,'hartlap':hartlap,'A':a,'B':b,'percival_m1':m1,'percival_sigma_factor':np.sqrt(m1)}

BaseEngine=common.Engine
class EZEngine(BaseEngine):
    def __init__(self,variant,method):
        """沿用原数据与模型；用 C/Hartlap 构造白化算子，等价于 Hartlap*C^{-1}。"""
        super().__init__(variant,method)
        self.corrections=factors(len(self.data),len(self.names))
        self.metric=common.Metric(self.cov/self.corrections['hartlap'])
common.Engine=EZEngine

def prepare():
    """只变更协方差，逐数组确认数据和模型不变，并在经验协方差下重新评估数值误差。"""
    if (OUT/'input_audit.json').exists():raise FileExistsError('Frozen inputs already exist')
    OUT.mkdir(parents=True,exist_ok=True)
    source_audit=json.loads((JAX_OUT/'input_audit.json').read_text())
    assert source_audit['status']=='pass'
    # 按原生产读入器验证所有 1000 组 P/xi 的状态、共同 RR、bin 和有限性。
    import task43_rsd_ezmock281_mcmc_compare as original
    original.BASE=PRODUCTION
    original.MANIFEST=MANIFEST
    stack,rows=original.load_ezmock_stack(1000)
    ids=np.asarray([row['production_index'] for row in rows],dtype='i8')
    assert np.array_equal(ids,np.arange(1000))
    covariance=np.cov(stack,rowvar=False,ddof=1)
    np.savez_compressed(COV_SOURCE,stack=stack,covariance=covariance,production_indices=ids)
    save(OUT/'mock_selection_manifest.json',{'source_manifest':str(MANIFEST),'source_manifest_sha256':sha(MANIFEST),'validated_mock_count':1000,'production_indices':ids,'rows':rows})
    with np.load(COV_SOURCE,allow_pickle=False) as a:
        cov=np.asarray(a['covariance']);stack=np.asarray(a['stack']);indices=a['production_indices']
    assert cov.shape==(74,74) and stack.shape==(1000,74)
    rebuilt=np.cov(stack,rowvar=False,ddof=1)
    covariance_relfro=np.linalg.norm(cov-rebuilt)/np.linalg.norm(cov)
    assert covariance_relfro<1e-12
    scale=np.sqrt(np.diag(cov));mineig=np.linalg.eigvalsh(cov/np.outer(scale,scale)).min()
    assert mineig>1e-12
    with np.load(JAX_OUT/'frozen_inputs.npz',allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    arrays['covariance']=cov
    np.savez_compressed(OUT/'frozen_inputs.npz',**arrays)
    for name in ('xi_emulator.npz','validation_predictions.npz'):
        shutil.copy2(JAX_OUT/name,OUT/name)
    with np.load(JAX_OUT/'frozen_inputs.npz',allow_pickle=False) as old:
        unchanged={k:np.array_equal(arrays[k],old[k]) for k in arrays if k!='covariance'}
    assert all(unchanged.values())
    emu=common.Emulator(OUT/'xi_emulator.npz',source_audit['emulator_method'])
    # 未乘 Hartlap 的度量更严格；模型不变，可复用已经直接计算过的验证点。
    metric=common.Metric(cov)
    with np.load(OUT/'validation_predictions.npz',allow_pickle=False) as a:
        theta=a['theta'];direct=a['direct_nint1200']
    err=emu(theta)-direct
    errors=metric.chi2(np.column_stack([np.zeros((len(theta),22)),err]))
    audit={'status':'pass' if np.max(errors)<.01 else 'failed','reference':str(common.REFERENCE),
        'reference_sha256':sha(common.REFERENCE),'covariance_source':str(COV_SOURCE),'covariance_source_sha256':sha(COV_SOURCE),
        'parent_input_audit':str(JAX_OUT/'input_audit.json'),'parent_input_audit_sha256':sha(JAX_OUT/'input_audit.json'),
        'unchanged_input_arrays':unchanged,'nmock':1000,'covariance_rebuild_relative_frobenius':covariance_relfro,
        'correlation_min_eigenvalue':mineig,'mock_production_indices':indices,
        'emulator_method':source_audit['emulator_method'],'validation_error_chi2_joint_no_hartlap':errors,
        'inherited_direct_validation':source_audit['validation'],'priors':source_audit['priors'],
        'data_contract':'9.18 EZmock-1000: P0 13, P2 9, xi0/xi2 26 each; x25 mean; C_single; full P-xi cross',
        'model_contract':source_audit['xi_model_contract'],
        'corrections':{v:factors(n,p) for v,n,p in [('p02',22,4),('xi02',52,2),('joint',74,4)]},
        'host':socket.gethostname(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'source_hashes':{str(p):sha(p) for p in [Path(__file__),Path(common.__file__),ROOT/'codes/task432/task432_hybrid_0918_postflight.py']}}
    save(OUT/'input_audit.json',audit);log('prepare_done',status=audit['status'],max_error_chi2=np.max(errors),corrections=audit['corrections'])
    if audit['status']!='pass':raise RuntimeError('New covariance interpolation gate failed')

def correct_summary(variant):
    """保留原始后验，并围绕 q50 统一放大区间；不将修正后的展示样本冒充重新采样链。"""
    root=OUT/'fits'/variant;s=json.loads((root/'summary.json').read_text())
    if 'posterior_raw' in s:raise RuntimeError('Already corrected')
    correction=factors(s['data_dimension'],len(s['parameter_names']))
    s['posterior_raw']=copy.deepcopy(s['posterior'])
    fac=correction['percival_sigma_factor']
    for item in s['posterior'].values():
        center=item['q50']
        for key in ('q16','q84'):item[key]=center+fac*(item[key]-center)
        item['sigma68']*=fac
    s['hartlap_percival']=correction
    s['chain_convention']='Stored chain sampled Hartlap-corrected likelihood; Percival applied only to reported intervals and displayed contour samples.'
    s['status']='pass' if all(s['gates'].values()) else 'failed'
    save(root/'summary.json',s);log('finite_mock_corrected',variant=variant,posterior=s['posterior'],corrections=correction)
    if s['status']!='pass':raise RuntimeError('Corrected baseline gate failed')

def postflight():
    """复用直接 GSM 验证；导入时公共 Engine 已设为本次 Hartlap 协方差版本。"""
    import task432_hybrid_0918_postflight as check
    check.main()

def plot():
    """复用原画图函数；图和 JSON 都应用 Percival，不延续旧图与 JSON 不一致的问题。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from task43_plot_ezmock506_covariance_mcmc_vs_jaxpower import triangle_page
    ref=json.loads(common.REFERENCE.read_text());chains={};results={};quantile_checks={}
    for variant in ('p02','xi02','joint'):
        root=OUT/'fits'/variant;s=json.loads((root/'summary.json').read_text());assert s['status']=='pass'
        with np.load(root/'samples.npz',allow_pickle=False) as a:raw=a['chain'].reshape(-1,len(s['parameter_names']))
        centers=np.array([s['posterior_raw'][name]['q50'] for name in s['parameter_names']])
        display=centers+s['hartlap_percival']['percival_sigma_factor']*(raw-centers)
        q=np.percentile(display,[16,50,84],axis=0)
        target=np.array([[s['posterior'][name][key] for name in s['parameter_names']] for key in ('q16','q50','q84')])
        assert np.allclose(q,target,rtol=1e-12,atol=1e-12)
        quantile_checks[variant]=float(np.max(np.abs(q-target)))
        chains['ezmock1000_'+variant]={'chain':display,'summary':s};results[variant]=s
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],'pdf.fonttype':42,'ps.fonttype':42,'font.size':10.,'axes.linewidth':1.,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
    MEETING.mkdir(parents=True,exist_ok=True)
    path=MEETING/'task432_hybrid_ezmock1000_P02xi02_joint_kmax0p08_smin50_baomask80_120_9.18style_v1.pdf'
    if path.exists():raise FileExistsError(path)
    # 维持同一页布局，短脚注明确区间不是未经修正的链分位数。
    with PdfPages(path) as pdf:
        class AnnotatedPDF:
            def savefig(self,figure,**kwargs):
                """在原样式留白中注明有限 mock 修正，再保存唯一一页 PDF。"""
                figure.text(.5,.018,'Hartlap likelihood; Percival-rescaled intervals and contours',ha='center',fontsize=7.5,color='0.35')
                pdf.savefig(figure,**kwargs)
        triangle_page(AnnotatedPDF(),cov='ezmock1000',cov_display='EZmock-1000 (hybrid xi)',chains=chains,ranges=ref['shared_axis_ranges'])
    b={v:{'q50':r['posterior']['b1']['q50'],'maximum_likelihood':r['map_theta'][1],'q16':r['posterior']['b1']['q16'],'q84':r['posterior']['b1']['q84']} for v,r in results.items()}
    between={key:min(b['p02'][key],b['xi02'][key])<=b['joint'][key]<=max(b['p02'][key],b['xi02'][key]) for key in ('q50','maximum_likelihood')}
    result={'status':'pass','output_pdf':str(path),'output_pdf_sha256':sha(path),'reference':str(common.REFERENCE),'axis_ranges':ref['shared_axis_ranges'],'input_audit':str(OUT/'input_audit.json'),
        'results':results,'b1_comparison':b,'joint_b1_between_marginals':between,'display_summary_quantile_max_error':quantile_checks,
        'finite_mock_note':'1000 mocks. Hartlap in likelihood; Percival computed for actual free parameter count and applied consistently to JSON intervals and plotted samples. Raw chains preserved.',
        'model_note':'Same hybrid as jaxpower rerun: bin-center xi, no formal GIC, no xi FoG parameter.'}
    save(path.with_suffix('.json'),result);log('plot_done',path=path,b1=b,between=between)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','p02','xi02','joint','postflight','plot']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='postflight':postflight()
    elif args.action=='plot':plot()
    else:common.fit(args.action);correct_summary(args.action)
