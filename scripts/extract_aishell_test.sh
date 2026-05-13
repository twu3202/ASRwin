#!/bin/bash
set -e
cd /mnt/d/Winprojects/ASRwin/data/data_aishell/wav
echo "[*] extracting 20 AISHELL-1 test speakers..."
SPEAKERS="S0764 S0765 S0766 S0767 S0768 S0769 S0770 S0901 S0902 S0903 S0904 S0905 S0906 S0907 S0908 S0912 S0913 S0914 S0915 S0916"
n=0
for s in $SPEAKERS; do
  if [ -f "$s.tar.gz" ]; then
    tar -xzf "$s.tar.gz" 2>/dev/null || true
    n=$((n+1))
    echo "  [$n/20] extracted $s"
  else
    echo "  MISSING $s.tar.gz"
  fi
done
echo
echo "[+] done. test wav count:"
find S07* S09* -maxdepth 2 -name '*.wav' 2>/dev/null | wc -l
echo "[+] sample:"
find S0764 -name '*.wav' 2>/dev/null | head -3
echo "[+] total size:"
du -sh S07* S09* 2>/dev/null | tail -5
