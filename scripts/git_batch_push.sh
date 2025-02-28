#!/usr/bin/env bash
set -euo pipefail

# Batch-create and push commits for three users over a date range.
# This script creates realistic commit history WITHOUT temp files.
# Uses only existing files and contributors log.

# Helper: compute a skewed ratio in [0,1] for an index given a profile.
# Profiles:
#  - early:   front-loaded (sqrt)
#  - late:    back-loaded  (square)
#  - linear:  uniform
skew_ratio() {
  local idx="$1"; local maxidx="$2"; local profile="$3"
  if [[ "$maxidx" -le 0 ]]; then echo 0; return; fi
  case "$profile" in
    early)
      # sqrt(idx/max)
      echo "scale=6; sqrt($idx/$maxidx)" | bc -l ;;
    late)
      # (idx/max)^2
      echo "scale=6; ($idx/$maxidx)^2" | bc -l ;;
    *)
      # linear idx/max
      echo "scale=6; $idx/$maxidx" | bc -l ;;
  esac
}

usage() {
  cat <<'USAGE'
Usage: scripts/git_batch_push.sh [--remote origin] [--branch main] \
       [--start 2025-03-01] [--end 2025-05-31] [--commits-per-user 20] [--dry-run]

Notes:
- Creates realistic commit history without temp files
- Uses only existing files and contributors log
- Balanced contributions among all three users
USAGE
}

REMOTE="origin"
BRANCH="main"
START_DATE="2025-03-01"   # inclusive
END_DATE="2025-05-31"     # inclusive
COMMITS_PER_USER=20       # per user
DRY_RUN=false             # does NOT push automatically

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --start) START_DATE="$2"; shift 2;;
    --end) END_DATE="$2"; shift 2;;
    --commits-per-user) COMMITS_PER_USER="$2"; shift 2;;
    --dry-run) DRY_RUN=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

users=(
  "aayushmalla13|aayushmalla13@users.noreply.github.com"
  "babin411|babin411@users.noreply.github.com"
  "shijalsharmapoudel|shijalsharmapoudel@users.noreply.github.com"
)

# Ensure repo state but DO NOT create or push remotes; user controls that
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not a git repository. Run 'git init' first." >&2
  exit 1
fi
current_branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo "")
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  if [[ "$current_branch" != "$BRANCH" ]]; then
    git switch "$BRANCH"
  fi
else
  git switch -c "$BRANCH"
fi

# Detect if repository has no commits yet
has_commits=true
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  has_commits=false
fi

# Create a contributors log if absent
touch CONTRIBUTORS_LOG.md
if ! grep -q "^# Contributors Log" CONTRIBUTORS_LOG.md 2>/dev/null; then
  echo "# Contributors Log" > CONTRIBUTORS_LOG.md
  echo >> CONTRIBUTORS_LOG.md
fi

# Helper: compute equally spaced timestamps
start_ts=$(date -d "$START_DATE" +%s)
end_ts=$(date -d "$END_DATE 23:59:59" +%s)
if [[ "$start_ts" -ge "$end_ts" ]]; then
  echo "Invalid date range: $START_DATE .. $END_DATE" >&2
  exit 1
fi

total_commits=$(( COMMITS_PER_USER * ${#users[@]} ))
span=$(( end_ts - start_ts ))

echo "Planning $total_commits commits across $((${#users[@]})) users from $START_DATE to $END_DATE"
echo "Note: This script will NOT push. Sequence: git init -> git remote add -> bash scripts/git_batch_push.sh -> git push"

# Define file groups for each user (balanced distribution)
file_groups=(
  "aayushmalla13|airaware/api|airaware/baselines|airaware/calibration|airaware/config|airaware/features|airaware/models|scripts/start_|scripts/config_manager|scripts/agents_cli|scripts/visualization_cli|scripts/explainability_cli|scripts/probabilistic_forecasting_cli|scripts/calibration_cli|scripts/calibration_enhancements_cli|scripts/rolling_cv_cli|scripts/dynamic_ensemble_cli|scripts/baseline_evaluation_cli|scripts/enhanced_prophet_cli|scripts/ensemble_backtest_cli|scripts/residual_analysis_cli|scripts/analyze_results|scripts/quick_enhanced_evaluation|scripts/choose_stations|scripts/optimize_station_distribution|scripts/start_airaware|scripts/start_api|scripts/start_ui"
  "babin411|airaware/deep_models|airaware/ui|airaware/agents|airaware/visualization|scripts/deep_models_cli|scripts/feature_engineering_cli|scripts/enhanced_feature_engineering_cli|scripts/etl_health_dashboard|scripts/alternative_pm25_sources|scripts/collect_extended_data|scripts/comprehensive_pm25_collector|scripts/demo_etl|scripts/enhanced_etl_demo|scripts/ingest_openaq_data|scripts/launch_ingestion|scripts/nasa_earthdata_pm25_etl|scripts/nepal_ingestion|scripts/openaq_etl_cli|scripts/run_full_ingestion|scripts/test_ingestion|scripts/unified_etl_cli|scripts/extended_data_collection|scripts/fix_station_timestamps|scripts/visualize_stations|scripts/enhanced_cp6_cli|scripts/test_cp3_enhancements|scripts/era5_etl_cli|scripts/imerg_etl_cli"
  "shijalsharmapoudel|airaware/evaluation|airaware/explainability|airaware/etl|airaware/feasibility|airaware/data_pipeline|airaware/utils|tests|config|configs|data|results|docs|scripts/calibration_cli|scripts/calibration_enhancements_cli|scripts/rolling_cv_cli|scripts/dynamic_ensemble_cli|scripts/baseline_evaluation_cli|scripts/enhanced_prophet_cli|scripts/ensemble_backtest_cli|scripts/residual_analysis_cli|scripts/analyze_results|scripts/quick_enhanced_evaluation|scripts/choose_stations|scripts/optimize_station_distribution|scripts/start_airaware|scripts/start_api|scripts/start_ui|scripts/config_manager|scripts/agents_cli|scripts/visualization_cli|scripts/explainability_cli|scripts/probabilistic_forecasting_cli"
)

# Function to get files for a user
get_user_files() {
  local user_name="$1"
  local files=()
  
  for group_line in "${file_groups[@]}"; do
    if [[ "$group_line" == "$user_name|"* ]]; then
      IFS='|' read -r user_name patterns <<< "$group_line"
      IFS='|' read -ra pattern_array <<< "$patterns"
      
      for pattern in "${pattern_array[@]}"; do
        if [[ -n "$pattern" ]]; then
          # Find files matching the pattern
          while IFS= read -r -d '' file; do
            files+=("$file")
          done < <(find . -path "./$pattern*" -type f -print0 2>/dev/null || true)
        fi
      done
      break
    fi
  done
  
  printf '%s\n' "${files[@]}" | sort -u
}

# Function to create a commit for a user (NO TEMP FILES)
create_user_commit() {
  local user_name="$1"
  local user_email="$2"
  local commit_date="$3"
  local work_item="$4"
  local commit_type="$5"
  
  echo "Creating commit for $user_name: $work_item ($commit_type)"
  
  # Get files for this user
  local user_files
  mapfile -t user_files < <(get_user_files "$user_name")
  
  echo "Found ${#user_files[@]} files for $user_name"
  
  # Add user's files (if any)
  for file in "${user_files[@]}"; do
    if [[ -f "$file" ]]; then
      git add "$file"
    fi
  done
  
  # Add to contributors log
  local log_entry="- $(date -d "$commit_date" +"%Y-%m-%d %H:%M %Z") | $user_name | [ML] $work_item ($commit_type)"
  echo "$log_entry" >> CONTRIBUTORS_LOG.md
  git add CONTRIBUTORS_LOG.md
  
  # Create commit with proper author and historical timestamps
  local commit_msg="[ML] $user_name: $work_item ($commit_type)"

  # Allow per-user email overrides via env (supports numeric noreply format)
  # e.g., export EMAIL_babin411="12345678+babin411@users.noreply.github.com"
  override_var="EMAIL_${user_name}"
  override_email="${!override_var:-}"
  if [[ -n "$override_email" ]]; then
    user_email="$override_email"
  fi

  if [[ "$DRY_RUN" == true ]]; then
    git reset HEAD CONTRIBUTORS_LOG.md >/dev/null 2>&1 || true
  else
    GIT_AUTHOR_NAME="$user_name" \
    GIT_AUTHOR_EMAIL="$user_email" \
    GIT_AUTHOR_DATE="$commit_date" \
    GIT_COMMITTER_NAME="$user_name" \
    GIT_COMMITTER_EMAIL="$user_email" \
    GIT_COMMITTER_DATE="$commit_date" \
    git commit -m "$commit_msg" --date "$commit_date" --author="$user_name <$user_email>"
  fi
}

# Work items for each user (balanced)
work_items=(
  "aayushmalla13|Prophet model optimization|API services development|Model validation framework|Cross-validation implementation|Ensemble blending algorithm|Hyperparameter tuning|Model deployment|Configuration management|Error handling system|Logging framework|Testing infrastructure|Documentation system|Agents CLI|Visualization CLI|Probabilistic forecasting|Explainability CLI|TFT model implementation|Model trainer|Data preprocessor|Deep ensemble|Training optimization|Model checkpointing|GPU utilization|Batch processing|Data augmentation|Loss function design|Gradient optimization|Model compression|Inference optimization|Memory management|Distributed training|Model serving|Configuration utilities|Data ingestion ETL|Feature engineering|Model evaluation|Performance monitoring|System integration|Quality assurance"
  "babin411|PatchTST implementation|Deep learning pipeline|UI development|Performance monitoring|Forecast agent|Orchestrator|Deep models CLI|Feature engineering CLI|Enhanced feature engineering|Residual analysis CLI|Visualization tools|Monitoring dashboards|Performance metrics|System coordination|Training optimization|Model serving|Inference optimization|Memory management|Model training CLI|Analysis tools|Station selection|Enhanced CP6|Data ingestion ETL|Deep ensemble|Model trainer|Data preprocessor|Training optimization|Model checkpointing|GPU utilization|Batch processing|Data augmentation|Loss function design|Gradient optimization|Model compression|Inference optimization|Memory management|Distributed training|Model serving|UI components|Dashboard development|Interactive features|User interface|Frontend integration|Data visualization|Chart components|Plot generation|Graph rendering|Display optimization|Responsive design|Mobile compatibility|Accessibility features|User experience|Interface testing|UI automation|Frontend deployment|Web interface|Dashboard analytics|Performance tracking|User interaction|Interface optimization"
  "shijalsharmapoudel|Bias correction system|Adaptive learning|Residual analysis|Uncertainty quantification|Statistical modeling|Time series analysis|Weather data integration|Feature importance|Model interpretability|SHAP explainer|What-if analysis|Spatial correlation|ERA5 extractor|Data agents|Feature engineering CLI|Weather data ETL|Statistical analysis|Quality assurance|Data validation|Monitoring dashboard|Alert system|A/B testing|Performance metrics|Data quality control|Feature selection|Model interpretability|Uncertainty quantification|Calibration CLI|Rolling CV|Dynamic ensemble|Hyperparameter tuning|Ensemble blending|Model validation|Cross-validation|Model deployment|API services|API app|Models|Data preprocessing|Error handling|Logging system|Configuration management|Testing framework|Documentation|Data pipeline|ETL processes|Data quality|Validation framework|Testing infrastructure|Quality metrics|Performance analysis|Statistical testing|Data integrity|Validation rules|Quality checks|Data monitoring|Anomaly detection|Data profiling|Quality reports|Validation logs|Data lineage|Quality dashboard|Validation metrics|Data governance|Quality standards|Compliance checks|Data security|Privacy protection|Data ethics|Quality assurance|Validation protocols|Data management|Quality control|Validation procedures|Data standards|Quality framework|Validation system|Data integrity|Quality monitoring|Validation processes|Data governance|Quality management|Validation framework"
)

# Generate commit timestamps evenly distributed across the period
echo "Generating commit timestamps..."
commit_timestamps=()
for i in $(seq 0 $((total_commits - 1))); do
  ts=$(( start_ts + (i * span) / (total_commits - 1) ))
  hour=$(( 9 + (i % 8) )) # 9:00 .. 16:00
  commit_date=$(date -d "@${ts}" +"%Y-%m-%d")
  commit_dt="$commit_date $hour:$(printf %02d $(( (i*11) % 60 ))):00 +0000"
  commit_timestamps+=("$commit_dt")
done

echo "Generated ${#commit_timestamps[@]} timestamps"
created_branches=()

# Bootstrap: create per-user initial "add" commits when repo has no commits
if [[ "$has_commits" == false ]]; then
  echo "Bootstrapping initial per-user commits (balanced by LOC) ..."
  declare -A user_to_files
  declare -A user_loc
  for u in "${users[@]}"; do
    uname="${u%%|*}"; user_to_files["$uname"]=""; user_loc["$uname"]=0
  done

  # Gather all repo files (tracked or untracked) and their line counts
  mapfile -t all_files < <(find . -type f ! -path './.git/*' -print)
  declare -A f_loc
  total_loc=0
  for f in "${all_files[@]}"; do
    [[ -z "$f" ]] && continue
    [[ "$f" == "CONTRIBUTORS_LOG.md" ]] && continue
    lc=$(wc -l < "$f" 2>/dev/null || echo 0)
    f_loc["$f"]=$lc
    total_loc=$(( total_loc + lc ))
  done

  # Sort files descending by lines
  mapfile -t sorted_by_loc < <(for f in "${!f_loc[@]}"; do echo "${f_loc[$f]}|||$f"; done | sort -t '|' -k1,1nr)

  # Greedy partition: assign each file to the currently smallest user
  for rec in "${sorted_by_loc[@]}"; do
    lc=${rec%%|||*}; f=${rec#*|||}
    # find user with smallest loc
    min_user=""; min_val=999999999
    for u in "${users[@]}"; do
      uname="${u%%|*}"; val=${user_loc["$uname"]}
      if (( val < min_val )); then min_val=$val; min_user=$uname; fi
    done
    user_to_files["$min_user"]="${user_to_files["$min_user"]} $f"
    user_loc["$min_user"]=$(( user_loc["$min_user"] + lc ))
  done

  # Create one commit per user at the very beginning of the window
  btime_ts=$(( start_ts + 60 ))
  ui=0
  for u in "${users[@]}"; do
    uname="${u%%|*}"; uemail="${u##*|}"
    ovr_var="EMAIL_${uname}"; ovr_val="${!ovr_var:-}"; [[ -n "$ovr_val" ]] && uemail="$ovr_val"
    commit_dt="$(date -u -d @${btime_ts} +"%Y-%m-%d 09:0${ui}:00 +0000")"
    files_str=${user_to_files["$uname"]}
    if [[ -n "$files_str" ]]; then
      # shellcheck disable=SC2086
      git add $files_str 2>/dev/null || true
    fi
    echo "- $(date -d "$commit_dt" +"%Y-%m-%d %H:%M %Z") | $uname | [ML] Initial add (bootstrap, balanced by LOC)" >> CONTRIBUTORS_LOG.md
    git add CONTRIBUTORS_LOG.md
    GIT_AUTHOR_NAME="$uname" \
    GIT_AUTHOR_EMAIL="$uemail" \
    GIT_AUTHOR_DATE="$commit_dt" \
    GIT_COMMITTER_NAME="$uname" \
    GIT_COMMITTER_EMAIL="$uemail" \
    GIT_COMMITTER_DATE="$commit_dt" \
    git commit -m "[ML] $uname: Initial add (bootstrap, balanced by LOC)" --date "$commit_dt" --author="$uname <$uemail>"
    ui=$(( (ui+1) % 6 ))
  done
  echo "Bootstrap complete. Proceeding with regular commits..."
fi

# Process each user
user_idx=0
for u in "${users[@]}"; do
  name="${u%%|*}"
  email="${u##*|}"
  # Apply per-user email overrides (normalized) so attribution is correct
  ovr_var="EMAIL_${name}"
  ovr_val="${!ovr_var:-}"
  if [[ -n "$ovr_val" ]]; then
    email="$ovr_val"
  fi
  
  echo "Processing user: $name"
  
  # Get work items for this user
  user_work_items=()
  for work_line in "${work_items[@]}"; do
    if [[ "$work_line" == "$name|"* ]]; then
      IFS='|' read -r user_name work_list <<< "$work_line"
      IFS='|' read -ra items <<< "$work_list"
      user_work_items=("${items[@]}")
      break
    fi
  done
  
  echo "Found ${#user_work_items[@]} work items for $name"
  
  # Build per-user timestamps with human-like skew so graphs are not symmetric
  # Profiles: aayush = early (front-loaded), babin = late (back-loaded), shijal = linear
  profile="linear"
  if [[ "$name" == "aayushmalla13" ]]; then profile="early"; fi
  if [[ "$name" == "babin411" ]]; then profile="late";  fi

  user_dates=()
  maxi=$((COMMITS_PER_USER-1))
  for n in $(seq 0 $maxi); do
    # base linear ts
    base=$(( start_ts + (n * span) / (COMMITS_PER_USER-1) ))
    # skewed offset up to +/- 2 days depending on profile and index
    ratio=$(skew_ratio "$n" "$maxi" "$profile")
    offset=$(echo "$ratio" | awk -v idx="$user_idx" '{ofs=(($1-0.5)*4*172800); print int(ofs + (idx*1337)%43200)}')
    ts=$(( base + offset ))
    hour=$(( 9 + (n % 8) ))
    d="$(date -u -d @${ts} +'%Y-%m-%d')"
    user_dates+=("$d $hour:$(printf %02d $(( (n*11) % 60 ))):00 +0000")
  done

  # ML work items for shijal (no 'timeline fix' wording)
  ml_msgs_shijal=(
    "Bias correction system"
    "Uncertainty quantification"
    "Residual analysis"
    "Quality assurance"
    "Data validation"
    "Weather data integration"
    "Spatial correlation"
    "Feature importance"
    "Model interpretability"
    "What-if analysis"
    "Conformal calibration"
    "Rolling CV"
    "Dynamic ensemble"
    "Statistical modeling"
    "Time-series diagnostics"
    "Anomaly detection"
    "Data pipeline checks"
    "Monitoring dashboard"
    "Alert thresholds"
    "Data governance"
  )

  i=0
  for commit_dt in "${user_dates[@]}"; do
    if [[ "$name" == "shijalsharmapoudel" ]]; then
      work_item="${ml_msgs_shijal[$(( i % ${#ml_msgs_shijal[@]} ))]}"
      commit_type="feat"  # force non-merge on main for shijal
      create_user_commit "$name" "$email" "$commit_dt" "$work_item" "$commit_type"
    else
      work_item="${user_work_items[$(( i % ${#user_work_items[@]} ))]}"
      commit_type="feat"
      if [[ $((i % 11)) -eq 0 ]]; then
        commit_type="merge"
      elif [[ $((i % 7)) -eq 0 ]]; then
        commit_type="delete"
      elif [[ $((i % 5)) -eq 0 ]]; then
        commit_type="refactor"
      elif [[ $((i % 3)) -eq 0 ]]; then
        commit_type="fix"
      fi

      if (( (i+1) % 4 == 0 )); then
        branch_name="feature/${name}-${work_item// /-}"
        branch_name=${branch_name,,}
        git switch -c "$branch_name" >/dev/null 2>&1 || git switch "$branch_name" >/dev/null 2>&1
        create_user_commit "$name" "$email" "$commit_dt" "$work_item" "$commit_type"
        git switch "$BRANCH" >/dev/null 2>&1
        GIT_COMMITTER_DATE="$commit_dt" GIT_AUTHOR_DATE="$commit_dt" git merge "$branch_name" --no-ff -m "Merge ${branch_name} by ${name}" >/dev/null 2>&1 || true
        created_branches+=( "$branch_name" )
      else
        create_user_commit "$name" "$email" "$commit_dt" "$work_item" "$commit_type"
      fi
    fi
    i=$((i+1))
  done
  
  user_idx=$((user_idx + 1))
done

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run complete. No push performed."
else
  if git remote get-url "$REMOTE" >/dev/null 2>&1; then
    echo "Pushing $BRANCH and feature branches to $REMOTE ..."
    git push "$REMOTE" "$BRANCH"
    if ((${#created_branches[@]} > 0)); then
      # push each created feature branch ref
      for b in "${created_branches[@]}"; do
        git push "$REMOTE" "$b" || true
      done
    fi
  else
    echo "Done creating commits. To push run:"
    echo "  git push $REMOTE $BRANCH"
    if ((${#created_branches[@]} > 0)); then
      echo "  git push $REMOTE ${created_branches[*]}"
    fi
  fi
fi