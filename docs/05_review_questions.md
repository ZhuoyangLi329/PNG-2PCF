# 05 待审阅问题与第一阶段判别实验

这是准备者给审阅者的候选清单，不是预设答案。原模型与链没有被本轮修改。

| 优先级 | 问题 | 可复用输入 | 应返回的判别量 |
|---|---|---|---|
| P0 | joint原始量纲floor使xi信息消失 | 原theory cache、P02测量、joint源码 | 边际块前后差、单位变换不变性、cross=0 chi²可加性、预期秩 |
| P1 | P与xi的mode-pair/multipole约定是否统一 | angular_totals、periodic covariance、mode counts | 在同一个Gaussian周期场定义下推导PP/xx/Px三块，k与−k计数明确 |
| P1 | 离散mu vs 连续mu、shell核、测量bin约定 | ExactPeriodicPk0Model、FullDiscreteRSDModel、测量向量 | fNL=0/sigma_s=0/各向同性和Kaiser极限；角度处理A/B，不任意改理论物理k范围 |
| P2 | 实空间最简模板是否缺少BAO/broadband或stochastic自由度 | paired real-space x25与HOD三引擎数据 | 公平free-sn0 P对照、预注册scale cuts、保留完整C的残差特征 |
| P2 | Gaussian相关结构是否可信 | x25 phase stack与既有500相Quijote | 对角、压缩方向scatter、交叉验证；不直接逆高维x25 sample C |

## 已复核的必要条件

cross=0、固定原始PP/xx两个块、共享参数时，joint chi²是两侧chi²之和，所以它的全局最小值不能低于两侧各自独立最小值之和。现成数值：

- 单极：下界6.957045，保存的naive joint0.686292。
- 四向：下界14.846309，保存的naive joint3.141802。

离线脚本重建四向covariance发现：原矩阵eigmin≈1.00e−9（正），eigmax≈4.27e9，而原规则floor≈4.27e−5。57个xi方向全被抬升，sigma中位比15.4001。修改P的单位再变回去会改变xi块。这直接证明当前naive joint不再使用原边际权重。

## 暂不能从以上证据得到的结论

- 修复后joint增益是多少：没有重跑。
- 同一sigma_s是否物理上应该同时适配这两个probe：需要定义和模型检验。
- 全部shape failure只来自covariance或只来自模型：当前证据不足。
- 加四极子得到的更小fNL误差就是可靠科学增益：必须先通过对应形状/precision检查。
- 低k硬切或与P带宽匹配就能“修复”xi：需明确新统计量定义，不能替代原始完整xi。

请在给GPT5.6pro的交接文档指定格式下，给每项实验明确固定量、变化量、接受门、拒绝时归因及执行入口。
