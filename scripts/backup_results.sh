#!/bin/bash
# Backup all TAP-Score and integration results

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M)
BACKUP_NAME="tap_integration_backup_${TIMESTAMP}.tar.gz"

echo "Creating backup: ${BACKUP_NAME}"

# Create backup of critical directories and files
tar -czf "${BACKUP_NAME}" \
    checkpoints_contrastive/ \
    eval_results/ \
    report/ \
    diffusion_policy_checkpoints/ \
    eval_dp_with_tap.py \
    test_dp_tap_integration.py \
    INTEGRATION_GUIDE.md \
    CLAUDE.md \
    README.md

echo "✓ Backup created: ${BACKUP_NAME}"
echo ""
echo "Contents:"
tar -tzf "${BACKUP_NAME}" | head -20
echo "..."
echo ""
echo "Backup size: $(du -h ${BACKUP_NAME} | cut -f1)"
echo ""
echo "To restore:"
echo "  tar -xzf ${BACKUP_NAME}"
