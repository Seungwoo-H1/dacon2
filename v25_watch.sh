#!/bin/bash
# v25_watch.sh — V25 결과를 모니터링하다가 submission_v25* 파일이 생성되면 Git 커밋 + 푸시
# Usage: ./v25_watch.sh

set -e
cd /home/mwoo423/projects/dacon2

echo "🔍 V25 watching... checking for submission_v25* files every 30s"
echo "V25 processes:"
ps aux | grep "30_v25" | grep -v grep | head -5

LAST_MTIME=""

while true; do
    # Check for any new V25 submission file
    for f in submissions/submission_v25_*.csv; do
        [ -f "$f" ] || continue
        mtime=$(stat -c %Y "$f")
        if [ "$mtime" != "$LAST_MTIME" ]; then
            echo ""
            echo "🎉 NEW V25 SUBMISSION: $f"
            echo "   Size: $(wc -l < "$f") lines, Modified: $(date -d @$mtime)"
            
            LAST_MTIME="$mtime"
            
            # Git add, commit, push
            git add submissions/
            if git diff --cached --quiet; then
                echo "   → No changes to commit"
            else
                git commit -m "feat(v25): rolling ensemble submission $(basename "$f" .csv)"
                echo "   → Committed"
                
                # Try push (may fail if remote busy)
                if git push origin main 2>/dev/null; then
                    echo "   → Pushed ✓"
                else
                    echo "   → Push failed (will retry next check)"
                fi
            fi
            
            # Show log loss
            if [ -f "submissions/meta_v25_*.json" ]; then
                python3 -c "
import glob, json, glob
files = glob.glob('submissions/meta_v25_*.json')
if files:
    data = json.load(open(files[0]))
    print(f'   → Log Loss: {data.get(\"log_loss\", \"N/A\")}')
    print(f'   → Config: {data.get(\"config\", {}).get(\"name\", \"N/A\")}')
" 2>/dev/null || true
            fi
        fi
    done
    
    sleep 30
done
