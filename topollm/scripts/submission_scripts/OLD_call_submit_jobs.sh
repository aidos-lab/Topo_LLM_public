

if [ "$DO_FINETUNING" = "true" ]; then

  echo ">>> Submitting finetuning jobs ..."
  poetry run submit_jobs \
    --task="finetuning" \
    --template=$TEMPLATE \
    --queue=$QUEUE \
    --memory=$MEMORY \
    --ncpus=$NCPUS \
    --ngpus=$NGPUS \
    --walltime="48:00:00" \
    --finetuning_datasets_list=$FINETUNING_DATASETS_LIST \
    --finetuning_seed_list="one_seed" \
    --finetuning_regime="many_epochs_with_overfitting_risk" \
    --common_batch_size="32" \
    --submission_mode=$SUBMISSION_MODE \
    --wandb_project="Topo_LLM_finetuning_from_submission_script_DEBUG_large_batch_size" \
    $RUN_ONLY_FIRST_CONFIG_OPTION_FLAG \
    $DRY_RUN_FLAG
  echo ">>> Submitting finetuning jobs ..."
fi
# ================================================================== #
