#!/usr/bin/env bash
# Submit the two remaining equal-token replications (seeds 43 and 44).
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/submit_equal_token_seeds.sh [options]

Submit independent H100 jobs for the equal-token WikiText-103 experiment.
Seed 42 is the completed reference run; seeds 43 and 44 are submitted by default.

Options:
  --data PATH          Prepared dataset directory (default: data/wikitext103)
  --output-root PATH   Parent directory for an atomically claimed sweep directory (default: runs)
  --seeds LIST         Comma-separated checked-in seeds: 43,44 (default: 43,44)
  --account ACCOUNT    Forward Slurm account without storing it in Git
  --partition NAME     Forward Slurm partition without storing it in Git
  --help               Show this help text
EOF
}

data_dir='data/wikitext103'
output_root='runs'
seeds='43,44'
account=''
partition=''

while (($#)); do
  case "$1" in
    --data|--output-root|--seeds|--account|--partition)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --data) data_dir=$2 ;;
        --output-root) output_root=$2 ;;
        --seeds) seeds=$2 ;;
        --account) account=$2 ;;
        --partition) partition=$2 ;;
      esac
      shift 2
      ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
command -v sbatch >/dev/null || { echo 'sbatch is required on a Slurm login node.' >&2; exit 1; }
command -v uv >/dev/null || { echo 'uv is required; run scripts/setup.sh in a CPU allocation first.' >&2; exit 1; }
[[ -f scripts/train.sbatch && -x scripts/train.sh ]] || { echo 'Missing training entry points.' >&2; exit 1; }
[[ -f "$data_dir/manifest.json" && -f "$data_dir/tokenizer.json" ]] || { echo "Prepared data not found: $data_dir" >&2; exit 1; }

mkdir -p "$output_root"
[[ -w "$output_root" ]] || { echo "Output parent is not writable: $output_root" >&2; exit 1; }
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
base="$output_root/equal-token-seeds-$timestamp"
run_root="$base"
index=0
until mkdir "$run_root" 2>/dev/null; do
  index=$((index + 1))
  run_root="$base-$(printf '%03d' "$index")"
done
logs_dir="$run_root/logs"
mkdir "$logs_dir"
receipt="$run_root/submission.tsv"
printf 'submitted_utc\tseed\tjob_id\trun_directory\tstdout\tstderr\n' > "$receipt"

IFS=',' read -r -a seed_list <<< "$seeds"
((${#seed_list[@]})) || { echo 'At least one seed is required.' >&2; exit 2; }
for seed in "${seed_list[@]}"; do
  [[ "$seed" =~ ^(43|44)$ ]] || { echo "Unsupported seed: $seed (available: 43,44)" >&2; exit 2; }
  config="configs/h100-8h-seed$seed.json"
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
  uv run --no-sync python -m json.tool "$config" >/dev/null

  output="$run_root/seed$seed"
  stdout="$logs_dir/train-$timestamp-seed$seed-%j.out"
  stderr="$logs_dir/train-$timestamp-seed$seed-%j.err"
  args=(--parsable --output "$stdout" --error "$stderr")
  [[ -n "$account" ]] && args+=(--account "$account")
  [[ -n "$partition" ]] && args+=(--partition "$partition")
  response=$(sbatch "${args[@]}" scripts/train.sbatch --config "$config" --data "$data_dir" --output "$output")
  job_id=${response%%;*}
  [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Ambiguous sbatch response: $response" >&2; exit 1; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$timestamp" "$seed" "$job_id" "$output" "$stdout" "$stderr" >> "$receipt"
  status=$(squeue --noheader --jobs "$job_id" --format='%T %R' 2>/dev/null || true)
  if [[ -n "$status" ]]; then
    echo "Seed $seed accepted as job $job_id: $status"
  else
    status=$(sacct --noheader --jobs "$job_id" --format=State,ExitCode 2>/dev/null | head -n 1 || true)
    echo "Seed $seed accepted as job $job_id; scheduler state: ${status:-unavailable}"
  fi
done

echo "Submission receipt: $receipt"
echo "Monitor with: squeue -u \"\$USER\""
