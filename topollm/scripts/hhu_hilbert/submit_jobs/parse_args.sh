# parse_args.sh

# Initialize flag variables
DRY_RUN=false
JOB_RUN_MODE="HHU_HILBERT"

# Loop through arguments and process flags
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            echo "Dry run mode enabled."
            ;;
        --local)
            JOB_RUN_MODE="LOCAL"
            echo "Local job run mode enabled."
            ;;
        *)
            echo "Unknown option: $arg"
            ;;
    esac
done
