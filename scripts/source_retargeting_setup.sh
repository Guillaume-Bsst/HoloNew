# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

CONDA_ENV_NAME=${CONDA_ENV_NAME:-holonew}
echo "conda environment name is set to: $CONDA_ENV_NAME"

# Machine-specific data locations for the TEST-SOCP SMPL-X mesh / OMOMO tooling.
# Computed relative to this repo (portable across clone locations — no path is baked
# into the Python code); override by exporting WBT_SMPLX_DIR / WBT_OMOMO_DIR beforehand.
WBT_RL_ROOT=$( cd -- "${SCRIPT_DIR}/../../../.." &> /dev/null && pwd )
export WBT_SMPLX_DIR="${WBT_SMPLX_DIR:-${WBT_RL_ROOT}/data/00_raw_datasets/models/models_smplx_v1_1/models}"
export WBT_SMPLH_DIR="${WBT_SMPLH_DIR:-${WBT_RL_ROOT}/data/00_raw_datasets/models/smplh}"
export WBT_OMOMO_DIR="${WBT_OMOMO_DIR:-${WBT_RL_ROOT}/data/00_raw_datasets/OMOMO}"
# Global dataset roots for the --motion-name resolver (resolve files by sequence name).
export WBT_OMOMO_NEW_DIR="${WBT_OMOMO_NEW_DIR:-${WBT_RL_ROOT}/data/00_raw_datasets/OMOMO_new/OMOMO_new}"
export WBT_HODOME_DIR="${WBT_HODOME_DIR:-${WBT_RL_ROOT}/data/00_raw_datasets/HODome}"

source ${SCRIPT_DIR}/source_common.sh
source ${CONDA_ROOT}/bin/activate $CONDA_ENV_NAME
