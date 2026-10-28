#!/bin/bash
# Run on Jupiter: it sees /p and reaches the BSC transfer host directly.
# JURECA fallback for small transfers: MIDTRAIN_SSH_JUMP=jsc-vm.
# Copy only the verified release and already initialized model; never reinitialize.
set -euo pipefail
RELEASE=${MIDTRAIN_RELEASE:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step5/release}
MODEL=${MIDTRAIN_MODEL:-/p/data1/mmlaion/marianna/midtrain_mixes/prelude_midtrain_v1/step2/base_hf}
BUNDLE=${MIDTRAIN_BUNDLE:-$(dirname "$RELEASE")/mn5_bundle}
HOST=${MIDTRAIN_TRANSFER_HOST:-transfer3.bsc.es}
DEST=${MIDTRAIN_MN5_ROOT:-/gpfs/scratch/ehpc532/ot/prelude_midtrain_v1}
DRY_RUN=0
WITH_MODEL=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --with-model) WITH_MODEL=1 ;;
        *) echo "usage: $0 [--dry-run] [--with-model]" >&2; exit 2 ;;
    esac
done
[[ -f "$RELEASE/release_manifest.json" && -f "$RELEASE/SHA256SUMS" ]] || {
    echo 'verified token release missing' >&2; exit 3;
}
[[ -f "$BUNDLE/handoff_manifest.json" && -f "$BUNDLE/SHA256SUMS" ]] || {
    echo 'CPU-qualified runtime handoff missing' >&2; exit 3;
}
[[ "$DEST" == /gpfs/* && "$DEST" != *[!a-zA-Z0-9_./-]* ]] || {
    echo 'destination must be an explicit /gpfs path without shell metacharacters' >&2; exit 3;
}
if (( WITH_MODEL )); then
    [[ -f "$MODEL/midtrain_initialization.json" && -f "$BUNDLE/MODEL_SHA256SUMS" ]] || {
        echo 'initialized model or model checksum manifest missing' >&2; exit 3;
    }
fi
RSYNC=(rsync -aL --partial --partial-dir=.rsync-partial --info=progress2)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
[[ -z ${MIDTRAIN_SSH_JUMP:-} ]] || SSH+=(-J "$MIDTRAIN_SSH_JUMP")
[[ -z ${MIDTRAIN_KNOWN_HOSTS:-} ]] || SSH+=(-o "UserKnownHostsFile=$MIDTRAIN_KNOWN_HOSTS" -o StrictHostKeyChecking=yes)
printf -v RSYNC_RSH '%q ' "${SSH[@]}"
export RSYNC_RSH
if (( DRY_RUN )); then
    printf '%q ' "${SSH[@]}" "$HOST" "mkdir -p '$DEST/release' '$DEST/base_hf' '$DEST/handoff'"; echo
    printf '%q ' "${RSYNC[@]}" "$RELEASE/" "$HOST:$DEST/release/"; echo
    printf '%q ' "${RSYNC[@]}" "$BUNDLE/" "$HOST:$DEST/handoff/"; echo
    if (( WITH_MODEL )); then
        printf '%q ' "${RSYNC[@]}" "$MODEL/" "$HOST:$DEST/base_hf/"; echo
    fi
    printf '%q ' "${SSH[@]}" "$HOST" "cd '$DEST/release' && sha256sum -c SHA256SUMS"; echo
    printf '%q ' "${SSH[@]}" "$HOST" "cd '$DEST/handoff' && sha256sum -c SHA256SUMS"; echo
    if (( WITH_MODEL )); then
        printf '%q ' "${SSH[@]}" "$HOST" "cd '$DEST/base_hf' && sha256sum -c '../handoff/MODEL_SHA256SUMS'"; echo
    fi
    exit
fi
"${SSH[@]}" "$HOST" "mkdir -p '$DEST/release' '$DEST/base_hf' '$DEST/handoff'"
"${RSYNC[@]}" "$RELEASE/" "$HOST:$DEST/release/"
"${RSYNC[@]}" "$BUNDLE/" "$HOST:$DEST/handoff/"
if (( WITH_MODEL )); then
    # -L materializes base_hf's symlinked unchanged shards on the destination.
    "${RSYNC[@]}" "$MODEL/" "$HOST:$DEST/base_hf/"
fi
"${SSH[@]}" "$HOST" "cd '$DEST/release' && sha256sum -c SHA256SUMS"
"${SSH[@]}" "$HOST" "cd '$DEST/handoff' && sha256sum -c SHA256SUMS"
if (( WITH_MODEL )); then
    "${SSH[@]}" "$HOST" "cd '$DEST/base_hf' && sha256sum -c '../handoff/MODEL_SHA256SUMS'"
fi
