# parse_args.sh

# Initialize flag variables
DRY_RUN=false
JOB_RUN_MODE="hhu_hilbert"
SKIP_FINETUNING=false

# Loop through arguments and process flags
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            echo ">> Dry run mode enabled."
            ;;
        --local)
            JOB_RUN_MODE="local"
            echo ">> Local job run mode enabled."
            ;;
        --skip-finetuning)
            SKIP_FINETUNING=true
            echo ">> Skipping finetuning."
            ;;
        *)
            echo "Unknown option: $arg"
            ;;
    esac
done
