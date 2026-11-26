#!/bin/bash
#
# This script configures the environment to allow running Python scripts
# from anywhere within the project, correctly resolving package imports.
# It will also activate the virtual environment.
#
# INSTRUCTIONS:
# 1. Source this script in Bash from any directory.
# 2. Example: source setup_env.sh
#
# After sourcing, your PYTHONPATH will be set for the current terminal session.

# Function to check if the script is being sourced
is_sourced() {
    if [ -n "$ZSH_VERSION" ]; then
        [[ -o sourced ]]
    elif [ -n "$BASH_VERSION" ]; then
        [[ "${BASH_SOURCE[0]}" != "$0" ]]
    else
        echo "Unsupported shell. Please use bash or zsh."
        return 1
    fi
}

# Check kernel version
check_kernel_version() {
    # check if running on Raspberry Pi
    if uname -a | grep -q "Linux raspberrypi"; then

        # Check if CURRENT_VERSION is one of the invalid versions
        INVALID_KERNELS=("6.12.21" "6.12.22" "6.12.23" "6.12.24" "6.12.25")
        CURRENT_VERSION=$(uname -r | cut -d '+' -f 1) # Extract numeric version part

        if [[ " ${INVALID_KERNELS[@]} " =~ " ${CURRENT_VERSION} " ]]; then
            # If the current version is in the list of invalid versions, print an error message
            echo "Error: Kernel version $CURRENT_VERSION detected. This version is incompatible."
            echo "Please refer to the following link for more information:"
            echo "https://community.hailo.ai/t/raspberry-pi-kernel-compatibility-issue-temporary-fix/15322"
            return 1
        fi
    fi
}

# Only proceed if the script is being sourced
if is_sourced; then
    echo "Setting up the environment..."
    echo "Checking kernel version..."
    # Call the kernel version check function
    check_kernel_version || {
        echo "Exiting due to incompatible kernel version."
        return 1
    }
else
    echo "This script should be sourced, not executed directly."
    exit 1
fi

# Resolve this checkout independently of the caller's working directory.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
VENV_NAME="$(sed -nE 's/^[[:space:]]*virtual_env_name:[[:space:]]*"?([^"#[:space:]]+)"?.*/\1/p' "$PROJECT_ROOT/config.yaml")"
VENV_NAME="${VENV_NAME:-venv_hailo_rpi_examples}"
case "$VENV_NAME" in
    /*) activate="$VENV_NAME/bin/activate" ;;
    *) activate="$PROJECT_ROOT/$VENV_NAME/bin/activate" ;;
esac

if [[ ! -f "$activate" ]]; then
    echo "Virtual environment missing: $activate. Run install.sh first." >&2
    return 1
fi
source "$activate" || return 1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HAILO_ENV_FILE="${HAILO_ENV_FILE:-$PROJECT_ROOT/.env}"
echo "Virtual environment '$VENV_NAME' activated"

resolve_yolo_hef() {
    local arch="$1"
    local candidate
    if [[ -n "${YOLO_HEF_PATH:-}" ]]; then
        printf '%s\n' "$YOLO_HEF_PATH"
        return
    fi
    for candidate in \
        "$PROJECT_ROOT/resources/models/$arch/yolov8s.hef" \
        "/usr/local/hailo/resources/models/$arch/yolov8s.hef"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    printf 'Missing YOLOv8s HEF for %s. See README.md or set YOLO_HEF_PATH.\n' "$arch" >&2
    return 1
}