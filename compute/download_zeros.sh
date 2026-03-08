#!/bin/bash
# Download Odlyzko's precomputed zeta zeros
# 2,001,052 zeros accurate to 4e-9

mkdir -p ~/rh_data
cd ~/rh_data

echo "Downloading first 100,000 zeros..."
wget -q https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1
echo "  Done: zeros1 (100k zeros)"

echo "Downloading first 2,001,052 zeros..."
wget -q https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros6.gz
gunzip -f zeros6.gz
echo "  Done: zeros6 (2M zeros)"

echo ""
echo "Files in ~/rh_data:"
ls -lh ~/rh_data/zeros*
echo ""
echo "Line count:"
wc -l ~/rh_data/zeros6
echo ""
echo "Ready. Run: python rh_gpu.py"
