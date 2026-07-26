#!/bin/bash
# Resume pi0-fast YAM finetune from step 2500 -> 7000. WANDB_API_KEY passed via env.
cd /workspace/openpi || exit 1
CKPT=checkpoints/pi0_fast_yam_low_mem_finetune/yam_run
NORMF=assets/pi0_fast_yam_low_mem_finetune/Kavin60606/yam_pi0fast_train/norm_stats.json

PQ=$(find /workspace/yam_pi0fast_train -name '*.parquet' | wc -l)
MP4=$(find /workspace/yam_pi0fast_train -name '*.mp4' | wc -l)
echo "PREFLIGHT: parquet=$PQ (want 1947)  mp4=$MP4 (want 5841)"
echo "ckpt 2500 contents:"; ls "$CKPT/2500" 2>/dev/null | sed 's/^/    /' || echo "    MISSING 2500 dir"
if [ -f "$NORMF" ]; then echo "norm_stats: OK ($(du -h "$NORMF" | cut -f1))"; else echo "norm_stats: MISSING"; fi

cur=$(cat "$CKPT/wandb_id.txt" 2>/dev/null)
echo "wandb_id current=[$cur]"
if [ "$cur" = "OLD_RUN_ID" ] || [ -z "$cur" ]; then
  echo "resolving previous wandb run id from project pi0-fast-modal ..."
  uv run python -c "
import wandb
api = wandb.Api()
ent = api.default_entity
runs = list(api.runs(ent + '/pi0-fast-modal'))
cand = [r for r in runs if r.name == 'yam_run'] or runs
cand.sort(key=lambda r: r.created_at, reverse=True)
rid = cand[0].id
open('$CKPT/wandb_id.txt', 'w').write(rid)
print('RESOLVED wandb id', rid, '| name', cand[0].name, '| created', cand[0].created_at)
" || echo "WANDB RESOLVE FAILED"
fi
final=$(cat "$CKPT/wandb_id.txt" 2>/dev/null)
echo "wandb_id final=[$final]"

# gates -- do not burn GPU hours on a run that will crash
ABORT=0
[ "$PQ" = "1947" ] || echo "WARN: parquet count off ($PQ/1947)"
[ "$MP4" = "5841" ] || { echo "ABORT: videos incomplete ($MP4/5841)"; ABORT=1; }
[ -d "$CKPT/2500" ] || { echo "ABORT: missing 2500 checkpoint dir"; ABORT=1; }
[ -f "$NORMF" ]     || { echo "ABORT: missing norm_stats.json"; ABORT=1; }
if [ "$final" = "OLD_RUN_ID" ] || [ -z "$final" ]; then echo "ABORT: could not resolve wandb run id"; ABORT=1; fi
if [ "$ABORT" = "1" ]; then echo "=== NOT LAUNCHING (fix aborts above) ==="; exit 1; fi

tmux kill-session -t train 2>/dev/null
tmux new-session -d -s train "cd /workspace/openpi && echo \"=== TRAIN START $(date) ===\" | tee -a train.log && WANDB_API_KEY=$WANDB_API_KEY HF_HUB_DISABLE_XET=1 uv run scripts/train.py pi0_fast_yam_low_mem_finetune --exp-name yam_run --fsdp-devices 1 --resume --project-name pi0-fast-modal 2>&1 | tee -a train.log"
echo "=== LAUNCHED training in tmux session 'train' (2500 -> 7000) ==="

# give JAX time to compile + restore + hit first logged step
for i in $(seq 1 22); do sleep 15; done

echo "=== LOG TAIL (post-launch, ~5.5 min) ==="
grep -viE 'Xet Storage is enabled' train.log | tail -n 45
