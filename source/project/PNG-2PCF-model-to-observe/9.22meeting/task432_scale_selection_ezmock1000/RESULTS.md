# 少量尺度选择：EZmock1000 / hybrid + full GIC/RIC

主人~本轮只做基准加5个关键修改，长链只对kmax=0.06和取消BAO mask两组展开，共4条新链，喵～

**主要结果：kmax=0.06时，joint的b1在MAP和后验中位数上都位于Pk与2PCF之间；取消BAO mask仍未达到这一点，喵～**

| 配置 | n(P/xi/joint) | P b1 MAP | xi b1 MAP | joint b1 MAP | joint居中(MAP) |
|---|---|---:|---:|---:|---|
| baseline | 22/52/74 | 2.413459 | 2.422866 | 2.378192 | False |
| kmin0p013 | 19/52/71 | 2.432109 | 2.422866 | 2.378408 | False |
| kmax0p06 | 16/52/68 | 2.396832 | 2.422866 | 2.400993 | True |
| smin80 | 22/46/68 | 2.413459 | 2.219312 | 2.405357 | True |
| smax250 | 22/32/54 | 2.413459 | 2.429433 | 2.377688 | False |
| bao_unmasked | 22/60/82 | 2.413459 | 2.397301 | 2.373430 | False |

三个代表口径的实际MCMC约束如下；区间为原约定的Percival修正68%区间，喵～

| 配置 | 拟合 | b1中位数 [16%,84%] | fNL中位数 [16%,84%] |
|---|---|---|---|
| baseline | p02 | 2.409815 [2.311159, 2.509267] | -6.494 [-39.986, 27.149] |
| baseline | xi02 | 2.418489 [2.305301, 2.530065] | 1.489 [-26.781, 29.632] |
| baseline | joint | 2.376509 [2.307822, 2.444808] | 3.047 [-23.758, 28.282] |
| kmax0p06 | p02 | 2.386964 [2.241234, 2.540391] | -6.937 [-42.236, 29.399] |
| kmax0p06 | xi02 | 2.418489 [2.305301, 2.530065] | 1.489 [-26.781, 29.632] |
| kmax0p06 | joint | 2.396711 [2.302346, 2.490814] | 0.648 [-26.840, 27.123] |
| bao_unmasked | p02 | 2.409815 [2.311159, 2.509267] | -6.494 [-39.986, 27.149] |
| bao_unmasked | xi02 | 2.393127 [2.289964, 2.494359] | 3.333 [-24.821, 31.357] |
| bao_unmasked | joint | 2.371069 [2.304741, 2.437573] | 2.919 [-23.465, 28.089] |

kmax=0.06使joint的sigma68(b1)增加37.6%，sigma68(fNL)增加3.7%，喵～
因此它是本轮最值得保留的敏感性口径；居中同时伴随bias约束变宽，不能据此宣称原问题的物理原因已解决，也没有自动替换9.18基准，喵～

kmin=0.013和smax=250几乎不移动joint中心；smin=80也会让MAP居中，但原BAO mask使剩余xi第一点变为125，xi-only的局部曲率sigma(b1)约0.54，且sn0 MAP在先验上界，喵～该局部宽度不是后验可信区间，未为此配置运行MCMC，喵～
取消BAO mask主要使xi-only中心降低，joint仍低于两个单项；joint误差变化很小，喵～

具体bin口径：kmin=0.013移除P0中心0.006、0.008、0.010；kmax=0.06移除P0/P2中心0.062、0.070、0.078，保留最大中心0.054、上边界0.055，喵～
smin=80移除xi0/xi2中心55、65、75；smax=250移除中心255至345；取消BAO mask补回85、95、105、115，各多极共8个点，喵～完整bin边界和向量索引见executed_cut_manifest.json，喵～

高k六点在基准MAP处的条件raw chi2=0.168790，在删点后MAP处变为2.197057，喵～
原拟合中这些点没有明显条件残差异常；它们对退化方向与参数权重的影响是当前线索，不能直接称为坏点或模型失效点，喵～不同ndata的最小chi2不用于比较拟合优劣，小幅chi2变化不称为显著改善，喵～

2000次同相位配对bootstrap的局部joint b1位移为+0.02322，16%-84%范围[+0.01274,+0.03277]，喵～
EZ去均值噪声、同一六口径扫描的最大Dout减小量上尾频率为0.002499，喵～这是固定估计C和局部线性模型下的辅助量，不能换算为物理失配显著性或独立验证，尤其受同一25相位、xi边界和选择过程限制，喵～

所有拟合沿用C_single和完整P-xi cross block，没有除以25；配对均值噪声诊断才用25相位均值，喵～各cut从C切子矩阵后重新求逆，并按实际ndata/nparams重算Hartlap和Percival，喵～
P/joint参数为fNL、b1、sigma_s_P、sn0，xi参数为fNL、b1、sn0；全GIC/RIC及既有sn0投影保留，没有新自由参数，喵～
4条新链均64 walkers、30000步、burn-in 5000，split Rhat<1.01、长度>50tau、半链漂移<0.1sigma均通过，喵～未变单项复用旧链，data/C/prediction等价性检查通过，喵～
BAO八点来自全部1000个原EZ测量及25个halo测量，保留原74维data/C和模型回切；扩展模型沿用原fNL/b1网格并检查实际后验随机点、尾部和MAP，喵～
68个后验探针数值验收全部通过，最大q：{'P_geometry_q': 0.0008997149516026783, 'P_spline_q': 1.0461440694307115e-22, 'emulator_q': 3.275390884247344e-08, 'geometry_q': 0.005744059685182434, 'integration_q': 6.523475380890627e-10}，喵～q=delta_model^T C_cut^-1 delta_model；详细阈值、输入hash与gate见postflight.json，喵～

PDF第1-3页是基准/kmax=0.06/取消BAO mask的原9.18风格contour；第4页六配置MAP，第5页后验区间，第6页条件残差与配对稳定性，喵～图例按原规则显示fNL MAP与区间，后验中位数另列于表中，喵～
建议先保留kmax=0.06为稳健性结果，与原基准并列；下一步若继续归因，集中检查被删高k P0/P2通过sigma_s_P、sn0与cross covariance作用的方向，无须再扩大cut扫描，喵～

运行数据与完整链：`/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe/outputs/task43_outputs/rsd_validation/task432_model_repair/scale_selection_ezmock1000`，喵～
重现：task432_scale_bao.py build；task432_scale_selection.py map --cases baseline kmin0p013 kmax0p06 smin80 smax250 bao_unmasked；四个case/variant的sample；task432_scale_diagnostics.py；task432_scale_numerics.py；task432_scale_deliver.py，喵～
