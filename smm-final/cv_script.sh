#! /bin/bash -f

## Define path to your code directory
RDIR="/mnt/c/Users/Administrator/Documents/unixdir/exercises/SMM/"

## Define path you where you have placed the HLA data sets
DDIR="/mnt/c/Users/Administrator/Documents/unixdir/exercises/data/SMM/"


best_pcc=-1000
best_model=""

# Here you can type your allele names
for a in B4002
do

    mkdir -p $a.res
    
    cd $a.res
    
    # Here you can type the lambdas to test
    for l in 0 0.02 0.04 0.08
    do
    
        mkdir -p l.$l
        
        cd l.$l
            
        for epi in 0 0.01 0.02 0.04
        do
        
            mkdir -p epi.$epi
            
            cd epi.$epi
                    
            # Loop over the 5 cross validation configurations
            for n in 0 1 2 3 4 
            do
                        
                # Do training
                if [ ! -f mat.$n ] 
                    then
                    QT_QPA_PLATFORM=offscreen python $RDIR/smm_gradient_descent.py -l $l -epi $epi -t $DDIR/$a/f00$n -e $DDIR/$a/c00$n | grep -v "#" > mat.$n
                fi
                            
                # Do evaluation
                if [ ! -f c00$n.pred ] 
                    then
                    QT_QPA_PLATFORM=offscreen python $RDIR/pep2score.py -mat mat.$n -f  $DDIR/$a/c00$n | grep -v "PCC:" > c00$n.pred
                fi
                        
            done
                    
            # Step 1: Generate evaluation output and store in a variable
            eval_output="$a lambda $l epsilon $epi $(cat c00{0..4}.pred | grep -v "#" | gawk '{print $2,$3}' | bash "$RDIR/xycorr.sh") \
            $(cat c00{0..4}.pred | grep -v "#" | gawk '{print $2,$3}' | gawk 'BEGIN{n=0; e=0.0} {n++; e += ($1-$2)*($1-$2)} END {print e/n}')"
            
            # Step 2: Print it to terminal
            echo "$eval_output"
            
            # Step 3: Extract last two numbers from the string
            pcc=$(echo "$eval_output" | awk '{print $(NF-1)}')  # Second to last field
            err=$(echo "$eval_output" | awk '{print $NF}')      # Last field
    
            # Track best model
            if (( $(echo "$pcc > $best_pcc" | bc -l) )); then
                best_pcc=$pcc
                best_model="$a lambda $l epsilon $epi"
            fi
            
            cd ..
                
        done
        
        cd ..
            
    done
    
    cd ..

done

echo ""
echo "Best model: $best_model with correlation $best_pcc"
echo ""
