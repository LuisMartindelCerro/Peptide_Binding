#!/bin/bash

RDIR="/mnt/c/Users/Administrator/Documents/unixdir/exercises/SMM/"
DDIR="/mnt/c/Users/Administrator/Documents/unixdir/exercises/data/SMM/"

best_corr=-1000
best_params=""

for a in B4002 B0801 A3301
do
  for l in 0 0.02 0.04 0.08
  do
    for epi in 0 0.01 0.02 0.04
    do
      preds=""
      for n in 0 1 2 3 4
      do
        pred_file="${a}.res/l.${l}/epi.${epi}/c00${n}.pred"
        if [ -f "$pred_file" ]; then
          preds="$preds $(cat "$pred_file" | grep -v "^#" | awk '{print $2, $3}')"
        fi
      done

      if [ -n "$preds" ]; then
        tmpfile=$(mktemp)
        echo "$preds" > "$tmpfile"

        # Run xycorr.sh and save output for debugging
        xycorr_output=$(cat "$tmpfile" | bash "$RDIR/xycorr.sh")
        
        echo "DEBUG: Raw output from xycorr.sh:"
        echo "$xycorr_output"  # This prints exactly what xycorr.sh outputs
        
        # Extraemos correlación (2º campo) y error (3º campo) del output de xycorr.sh
        read _ corr error _ < <(cat "$tmpfile" | bash "$RDIR/xycorr.sh")

        # Calculamos el MSE localmente
        mse=$(awk '{n++; e += ($1-$2)*($1-$2)} END {print e/n}' "$tmpfile")

        echo "$a lambda $l epsilon $epi $corr $mse"

        if (( $(echo "$corr > $best_corr" | bc -l) )); then
          best_corr=$corr
          best_params="$a lambda $l epsilon $epi"
        fi

        rm "$tmpfile"
      fi
    done
  done
done

echo "Best model: $best_params with correlation $best_corr"
