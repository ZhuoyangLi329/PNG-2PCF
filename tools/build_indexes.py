#!/usr/bin/env python3
"""建立审阅包索引：代码→函数名；图→路径/状态；NPZ→数组键和shape。
不执行研究代码，不反序列化pickle，不生成新科学结果。
"""
from pathlib import Path
from urllib.parse import quote
import ast,json
import numpy as np
R=Path(__file__).resolve().parents[1]
def link(p):return '../'+quote(str(p.relative_to(R)),safe='/')
def group(p):
 s=str(p)
 if '/observelearn/ref/' in s or '/papers/' in s:return '论文'
 if '/background/' in s:return '历史方法与验证（含已替代方案）'
 if 'pngbase' in s:return 'PNG-base HOD与host-halo实空间（archive为历史口径）'
 if 'realspace' in p.name:return 'Abacus rawbox实空间诊断'
 if '/task43/' in s and 'rawbox' in s:return 'Abacus rawbox RSD/联合诊断（joint待修复）'
 if 'ezmock_rawbox' in s:return 'EZmock rawbox标定'
 if '4p2' in s or 'subbox' in s:return 'FastPM rawbox历史对照（混合图仅审阅rawbox面板）'
 return 'Quijote与其他rawbox历史图'
def main():
 figures={}
 for p in sorted((R/'source').rglob('*')):
  if p.is_file() and p.suffix.lower() in ['.pdf','.png','.jpg','.svg'] and group(p)!='论文':figures.setdefault(group(p),[]).append(p)
 rows=['# 图册与图文件索引\n\n原图均保留；“历史”不意味着错误，但不能自动进入当前主结果。PDF可能多页，请逐页查看。joint图展示的是原始待修复协方差下的后验。\n\n先看Abacus实空间、RSD单极、四极分探针与joint，再对照Quijote/FastPM/HOD。\n']
 for g,ps in figures.items():
  rows.append(f'\n## {g}（{len(ps)}）\n\n')
  for p in ps:rows.append(f'- [{p.name}]({link(p)})\n')
 (R/'docs/FIGURES.md').write_text(''.join(rows))
 codes=['# 代码索引\n\n源码按原目录保存；archive/mission早期文件是历史证据。共享joint helper是明确标注的摘录。索引列出的函数名是静态解析结果，不表示已运行。\n']
 py=[]
 for p in sorted((R/'source').rglob('*.py')):
  try:t=ast.parse(p.read_text());doc=(ast.get_docstring(t) or '').split('\n')[0];defs=[n.name for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
  except SyntaxError as e:doc=f'原源码语法问题：{e.msg}';defs=[]
  py.append({'path':str(p.relative_to(R)),'definitions':defs,'description':doc})
  codes.append(f'\n### [{p.name}]({link(p)})\n\n`{p.relative_to(R)}`\n\n{doc[:300]}\n\n'+', '.join('`'+n+'`' for n in defs)+'\n')
 (R/'docs/CODE_INDEX.md').write_text(''.join(codes));(R/'provenance/code_symbols.json').write_text(json.dumps(py,ensure_ascii=False,indent=2))
 out=['# 过程文档与结果索引\n\n按source原目录追踪；当前Abacus建议先读rawbox/closure、comparison、realspace_check/audits和failure_audit。其他历史/扩展结果不自动替代它们。\n']
 for label,exts in [('原过程文档',{'.md'}),('原结果、审计、配置摘要',{'.json','.csv'})]:
  out.append('\n## '+label+'\n\n')
  for p in sorted((R/'source').rglob('*')):
   if p.is_file() and p.suffix in exts:out.append(f'- [{p.relative_to(R/"source")}]({link(p)})\n')
 (R/'docs/RESULT_INDEX.md').write_text(''.join(out))
 inv=[]
 for p in sorted((R/'source').rglob('*.npz')):
  row={'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'arrays':{}}
  try:
   with np.load(p,allow_pickle=False) as d:
    for key in d.files:
     try:a=d[key];row['arrays'][key]={'shape':list(a.shape),'dtype':str(a.dtype)}
     except ValueError:row['arrays'][key]={'object_array_not_loaded':True}
  except Exception as e:row['error']=str(e)
  inv.append(row)
 (R/'review_data/npz_inventory.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2))
 print(json.dumps({'figures':sum(map(len,figures.values())),'python_sources':len(py),'npz':len(inv),'npz_errors':[x for x in inv if 'error' in x]},ensure_ascii=False))
if __name__=='__main__':main()
