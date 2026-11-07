"""Emit review-mapped jobs and commands; deliberately does not execute them."""
import json
from protocol import ROOT,OUT,HERE,native_spec,SOURCES,BUDGETS,write

def main():
    jobs=[]
    for method in ['direct','egodex_style','egoweave']:
        for seed in [43,44]:
            for suite in BUDGETS:
                s=native_spec(suite,method,seed)
                jobs.append(dict(id=s['run'],reviewer='SuSh: independent-checkpoint variation',kind='native',spec=s,
                    command=f'bash run.sh native.py --suite {suite} --method {method} --seed {seed} --mode train',input_status='await_migration_validation'))
    for condition in ['no_explicit_head','current_head','no_scale']:
        norm=OUT/'norms'/condition;run=f'{condition}_seed42'
        jobs.append(dict(id=run,reviewer='K3LE/WJnP: component attribution',kind='human_pretraining',
            prepare_command=f'bash run.sh prepare.py --condition {condition} --out {norm}',
            command=f'bash run.sh pretrain.py --condition {condition} --norm-dir {norm} --seed 42 --mode train',
            interpretation='Only explicit head flow terms removed; geometric head gradient retained' if condition=='no_explicit_head' else condition))
        for suite in BUDGETS:
            jobs.append(dict(id=f'{suite}_{condition}_seed43',reviewer='K3LE/WJnP: component attribution',kind='native',source_completion=str(OUT/'human'/run/'complete.json'),
                command=f'bash run.sh native.py --suite {suite} --method {condition} --seed 43 --human-run {run} --mode train',
                control=f'{suite}_egoweave_seed43',input_status='requires_actual_new_human_checkpoint',
                checkpoint_rule='Read actual final checkpoint from human complete.json; never infer its number from an old run.'))
    jobs.append(dict(id='global_vs_local',reviewer='K3LE/SuSh: reconstruction-to-policy linkage',kind='paired_reconstruction',
        input_status='missing_same_video_same_ROI_global_and_local_reconstruction_control_manifests',
        required_control_fields=['recordings_sha256','roi_sha256','calibration_sha256','estimator_sha256','fusion_recipe_sha256','smoothing_recipe_sha256'],
        workflow=['generate actual paired annotations on H100 where source videos reside','audit_windows.py for common anchors/masks','prepare per-source full norms','pretrain global_axis with --paired-local and --selection; local full on same selection','native.py for both checkpoints with identical seeds across all suites'],
        note='Existing unpaired smoothed local annotations are not relabeled as a global control.'))
    jobs.append(dict(id='annotation_availability',reviewer='K3LE: windows and supervised channels',kind='CPU_audit',
        command='prepare_cpu_inputs.py',output=str(OUT/'annotation_audit/summary.json'),
        retraining_rule='If availability differs, use the generated selection.json for both annotation sources and recompute appropriate norms.'))
    write(HERE/'experiments.json',dict(formal_jobs_launched=False,real_robot='skipped per latest user instruction',jobs=jobs,
        counts=dict(additional_main_method_finetunes=24,component_human_pretrains=3,component_finetunes=12),
        seed_scope='Two new finetuning seeds per main method; this does not estimate independent human-pretraining variation.'))
    print('Wrote',HERE/'experiments.json','No jobs launched')

if __name__=='__main__':main()
