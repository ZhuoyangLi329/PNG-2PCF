# 06 运行边界、依赖与复现

## 能直接离线做的事

在本仓库根目录，用Python 3.10+与NumPy运行：

```sh
python tools/audit_joint_covariance.py
python tools/verify_bundle.py
```

前者读取打包的rawbox理论cache、P02测量、原函数AST和已有JSON，复现数值处理问题；不需要NERSC、不需要emcee/cosmoprimo，也不执行原始main。后者检查文件哈希、关键交付物、数值产物和链接覆盖。

`review_data/npz_inventory.json`列出每个打包NPZ的键、shape、dtype。读取默认`allow_pickle=False`；不盲目加载object arrays。部分HOD fit NPZ内含完整链，文件名未必含samples。大数组不应直接全部打印到agent上下文。

## 本仓库不是完整可移植执行环境

原始源码保留NERSC绝对路径、环境加载、运行参数和输出约定。不能从本地直接批量执行run_*.sh，它们可能需要原catalog、FCFC二进制、GPU、已安装的cosmoprimo/desilike/jaxpower或对应环境。这里提供的是审阅快照与可离线验证的紧凑产物，不是伪称已通过端到端复现的发行包。

原项目位置：`/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe`；前身项目：`/pscratch/sd/l/lzy/1gpc-UNIT-fnl100-fastpm`。数据源catalog仍在NERSC。每个导入文件的原路径/哈希在provenance中。

早期Quijote脚本依赖task42；它已从归档目录找回，放在`source/project/old_doc_codes/task4_task44_cleanup_20260707T061844Z/moved/codes/task4/`。审阅或运行它时必须显式处理这个模块搜索路径；不能假定它仍与task45同目录。

本轮还保留了安装环境中的desilike理论源码快照供公式审阅；它不包含该软件完整发行版，不是对所有运行分支的版本锁定。原始各环境版本信息以结果metadata、shell wrapper和安装快照manifest为依据，不凭空给出锁定依赖。

## 仅为范围裁剪而改动的材料

- 原项目任务书只摘取rawbox科学段落，不保留个人角色prompt或后续survey执行任务。
- 原joint共享driver被裁成通用数值函数摘录，保留函数原文；去掉lightcone模型imports、数据路径和main。行号范围、原哈希和摘录定义记录在`provenance/curation.json`。
- 其余源文件原则上逐字保留；旧注释和旧结论不替作者修正，导读另行标注。
- notebook_text是源码镜像，notebook原文件仍保留执行输出/嵌入图。
- 历史rawbox/subbox混合图保留原图，不重新生成、裁剪或伪装为rawbox-only；未导出subbox应用结果与生产工作流。

## 后续在NERSC执行

现行资源规则是CPU工作总计≤8核，确认原有conda环境；需要GPU的测量再用GPU资源。缺包先定位已有环境。保留旧结果、不要默认覆盖；新诊断另建明确目录。只有在修复、统计审计和形状门都通过后，才更新正式科学结论。

请区分三种验证：本包文件完整性、离线数值反例复核、完整模型科学验证。前两者通过不意味着第三者通过。
