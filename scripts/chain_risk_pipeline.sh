#!/bin/bash
# Wait for collection to finish, then train and evaluate
set -e
cd "$(dirname "$0")/.."

ROLLOUT_DIR="data/risk_rollouts/can"
REGIMES="clean zero80 zero80_jitter dropout80_p02 dropout80_p04"

# Wait for 500 episodes
while true; do
    COUNT=$(ls "$ROLLOUT_DIR"/ep_*.npz 2>/dev/null | wc -l)
    echo "$(date +%H:%M) — $COUNT/500 episodes collected"
    if [ "$COUNT" -ge 500 ]; then
        echo "Collection complete!"
        break
    fi
    sleep 60
done

echo ""
echo "=== Phase 2: Train risk model ==="
python train_risk_tap.py \
    --rollout_dir "$ROLLOUT_DIR" \
    --task can --epochs 50 --batch_size 64 \
    --lr 1e-3 --fail_horizon 64 --hard_mining_epoch 15 \
    --action_chunk 8 \
    --regimes $REGIMES \
    --device cuda 2>&1 | tee logs/train_risk_can.log

echo ""
echo "=== Phase 3: Evaluate detection ==="
python scripts/eval_tap_detection.py \
    --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
    --tap_checkpoint checkpoints_contrastive/robomimic_can_lowdim/risk_tap_best.pt \
    --risk_model \
    --n_episodes 50 \
    --perturb zero_object --perturb_start_step 80 \
    --output eval_results/risk_detection_can_zero80_n50.json \
    --device cuda 2>&1 | tee logs/eval_risk_can.log

echo "DONE! Results: eval_results/risk_detection_can_zero80_n50.json"
