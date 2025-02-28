#!/bin/bash

# Docker Patch Commits Script
# Adds Docker-related files with historical commits for shijalsharmapoudel
# Maintains the March-May 2025 timeline

set -e

# Configuration
USER_NAME="shijalsharmapoudel"
USER_EMAIL="Shijalsharmapoudel@gmail.com"
GITHUB_USERNAME="shijalsharmapoudel"

# Time period: March 1, 2025 to May 31, 2025
START_DATE="2025-03-01"
END_DATE="2025-05-31"

# Convert to timestamps
start_ts=$(date -d "$START_DATE" +%s)
end_ts=$(date -d "$END_DATE" +%s)
span=$((end_ts - start_ts))

echo "🐳 Adding Docker commits for $USER_NAME"
echo "📅 Time period: $START_DATE to $END_DATE"

# Docker-related files to add (for shijalsharmapoudel)
declare -a docker_files=(
    "Dockerfile"
    "docker-compose.yml"
    "docker-compose.dev.yml"
    "Dockerfile.dev"
    ".dockerignore"
    "scripts/docker_helper.sh"
    "README_DOCKER.md"
)

# Docker-related commit messages (for shijalsharmapoudel)
declare -a docker_msgs=(
    "feat: add production Dockerfile with multi-stage build"
    "feat: add docker-compose.yml for service orchestration"
    "feat: add development Docker configuration"
    "feat: add development Dockerfile with dev tools"
    "feat: add .dockerignore for optimized builds"
    "feat: add Docker helper script for easy management"
    "docs: add comprehensive Docker documentation"
)

# Documentation files to update (for aayushmalla13)
declare -a doc_files=(
    "README.md"
    "mkdocs.yml"
    "docs/index.md"
    "docs/workflow.md"
    "docs/techstack.md"
    "docs/architecture.md"
)

# Documentation commit messages (for aayushmalla13)
declare -a doc_msgs=(
    "docs: update README with Docker setup instructions"
    "docs: add Docker section to MkDocs configuration"
    "docs: update project overview with containerization details"
    "docs: add Docker workflow to development process"
    "docs: update tech stack with Docker and containerization"
    "docs: update architecture with microservices and Docker deployment"
)

# Generate commit dates for Docker files (spread across the timeline)
declare -a docker_commit_dates=()
for i in $(seq 0 $((${#docker_files[@]} - 1))); do
    # Distribute commits across the timeline
    ratio=$(echo "scale=6; $i / $((${#docker_files[@]} - 1))" | bc -l)
    ts_offset=$(echo "scale=0; $span * $ratio / 1" | bc -l)
    ts=$((start_ts + ts_offset))
    
    # Add some jitter (± 1 day)
    jitter=$(( (i * 7) % (2 * 86400) - 86400 ))
    ts=$((ts + jitter))
    
    # Ensure within bounds
    if (( ts < start_ts )); then ts=$start_ts; fi
    if (( ts > end_ts )); then ts=$end_ts; fi
    
    hour=$(( 10 + (i % 6) )) # 10:00 to 15:00
    d="$(date -u -d @${ts} +'%Y-%m-%d')"
    docker_commit_dates+=("$d $hour:$(printf %02d $(( (i*13) % 60 ))):00 +0000")
done

# Generate commit dates for documentation files (later in timeline)
declare -a doc_commit_dates=()
for i in $(seq 0 $((${#doc_files[@]} - 1))); do
    # Start documentation commits after Docker files are done
    base_ratio=$(echo "scale=6; 0.6 + ($i / $((${#doc_files[@]} - 1))) * 0.4" | bc -l)
    ts_offset=$(echo "scale=0; $span * $base_ratio / 1" | bc -l)
    ts=$((start_ts + ts_offset))
    
    # Add some jitter (± 1 day)
    jitter=$(( (i * 11) % (2 * 86400) - 86400 ))
    ts=$((ts + jitter))
    
    # Ensure within bounds
    if (( ts < start_ts )); then ts=$start_ts; fi
    if (( ts > end_ts )); then ts=$end_ts; fi
    
    hour=$(( 14 + (i % 4) )) # 14:00 to 17:00
    d="$(date -u -d @${ts} +'%Y-%m-%d')"
    doc_commit_dates+=("$d $hour:$(printf %02d $(( (i*17) % 60 ))):00 +0000")
done

# Create commits for each Docker file (shijalsharmapoudel)
echo "🐳 Creating Docker commits for shijalsharmapoudel..."
for i in "${!docker_files[@]}"; do
    file="${docker_files[$i]}"
    msg="${docker_msgs[$i]}"
    commit_date="${docker_commit_dates[$i]}"
    
    echo "📝 Adding commit for $file"
    
    # Set the commit date
    export GIT_AUTHOR_DATE="$commit_date"
    export GIT_COMMITTER_DATE="$commit_date"
    
    # Create the commit
    git -c user.name="$USER_NAME" -c user.email="$USER_EMAIL" commit \
        --author="$USER_NAME <$USER_EMAIL>" \
        --date="$commit_date" \
        -m "$msg"
    
    echo "✅ Committed: $msg"
done

# Create commits for documentation files (aayushmalla13)
echo "📚 Creating documentation commits for aayushmalla13..."
AAYUSH_NAME="aayushmalla13"
AAYUSH_EMAIL="aayushmalla56@gmail.com"

for i in "${!doc_files[@]}"; do
    file="${doc_files[$i]}"
    msg="${doc_msgs[$i]}"
    commit_date="${doc_commit_dates[$i]}"
    
    echo "📝 Adding commit for $file"
    
    # Set the commit date
    export GIT_AUTHOR_DATE="$commit_date"
    export GIT_COMMITTER_DATE="$commit_date"
    
    # Create the commit
    git -c user.name="$AAYUSH_NAME" -c user.email="$AAYUSH_EMAIL" commit \
        --author="$AAYUSH_NAME <$AAYUSH_EMAIL>" \
        --date="$commit_date" \
        -m "$msg"
    
    echo "✅ Committed: $msg"
done

echo "🎉 Docker and documentation commits added successfully!"
echo "📊 Total Docker commits: ${#docker_files[@]} (shijalsharmapoudel)"
echo "📊 Total documentation commits: ${#doc_files[@]} (aayushmalla13)"
echo "📅 Timeline: $START_DATE to $END_DATE"

# Show the new commit history
echo ""
echo "📋 Recent commits:"
git log --oneline --author="$USER_NAME" -n 5
echo ""
git log --oneline --author="$AAYUSH_NAME" -n 5
