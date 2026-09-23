#!/usr/bin/env python3
"""执行已授权的积分加密收尾流程；依赖检查失败即停，不自动绕过gate。

流程：等待两套已启动核 -> 同门限验证 -> 编译/直接模型检查 ->
三组独立重采样 -> 后验复核 -> 两份PDF交付。该入口在登录节点8核内运行。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import subprocess
import sys
import time
from task432_full_ric_geometry import ROOT, OUT, save_json


def main():
    """每个阶段有状态记录；任何退出错误都写入workflow_status并向调用者报错。"""
    stage = 'waiting_for_refined_kernels'; started = time.time()
    def state(status='running', **extra):
        save_json(OUT/'logs/workflow_status.json', {'status': status, 'stage': stage, 'elapsed_s': time.time()-started, **extra})
        print(json.dumps({'status': status, 'stage': stage, **extra}), flush=True)
    def run(script, *args):
        command = [sys.executable, '-u', str(ROOT/'codes/task432'/script), *map(str, args)]
        subprocess.run(command, cwd=ROOT, check=True)
    def passed(path):
        if json.loads(path.read_text())['status'] != 'pass':
            raise RuntimeError(f'Audit did not pass: {path}')
    try:
        state()
        g = OUT/'geometry'
        new = [g/f'ph000_qmc{seed}_n32768_dchi2_refine1_midpoint_inmidpoint_ell8.npz' for seed in (4322291, 4322333)]
        while not all(p.exists() for p in new):
            if time.time()-started > 10800:
                raise RuntimeError('Timed out waiting for kernel completion; inspect existing jobs before any restart')
            time.sleep(30)
        stage = 'geometry_refinement_audit'; state()
        run('task432_full_ric_refinement.py')
        audit = OUT/'smoke/geometry_validation_mean4.json'; passed(audit)
        stage = 'compile_and_direct_model_audit'; state()
        run('task432_full_ric_compile.py', '--tag', 'mean4xi_mean2P_prodrr_stochastic', '--xi-kernel',
            g/'validated_mean4_midpoint.npz', '--p-kernel', g/'validated_mean2_endpoint.npz', '--production-rr')
        compiled = OUT/'pilot/mean4xi_mean2P_prodrr_stochastic/engine.npz'
        run('task432_full_ric_validation.py', 'model', '--compiled', compiled)
        passed(compiled.parent/'model_validation.json')
        stage = 'three_refined_MCMC_fits'; state()
        processes = []
        for variant in ('p02', 'xi02', 'joint'):
            log = (OUT/'logs'/f'mean4_{variant}.log').open('ab')
            p = subprocess.Popen([sys.executable, '-u', str(ROOT/'codes/task432/task432_full_ric_fit.py'), variant,
                                  '--compiled', str(compiled), '--run-name', 'formal_mean4', '--geometry-audit', str(audit)],
                                  cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            processes.append((variant, p, log))
        codes = {}
        for variant, p, log in processes:
            codes[variant] = p.wait(); log.close()
        if any(codes.values()):
            raise RuntimeError(f'MCMC failed: {codes}')
        stage = 'refined_posterior_numerics'; state()
        run('task432_full_ric_postflight.py', '--compiled', compiled, '--run-name', 'formal_mean4',
            '--xi-comparison', g/'validated_mean2_midpoint.npz', g/'refinement_independent_mean2_midpoint.npz')
        passed(OUT/'formal_mean4/postflight.json')
        stage = 'PDF_and_results'; state()
        run('task432_full_ric_deliver.py', '--compiled', compiled, '--run-name', 'formal_mean4')
        stage = 'computed_pending_visual_review'; state('pass')
    except Exception as error:
        state('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
