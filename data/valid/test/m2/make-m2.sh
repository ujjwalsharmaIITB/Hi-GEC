# for loop to iterate over directories

for dir in `ls`; do
    echo "Processing $dir"
    source_m2=$dir/$dir.m2
    source_src=$dir/$dir.src
    source_tgt=$dir/$dir.tgt
    target_m2="final.m2"
    target_src="final.src"
    target_tgt="final.tgt"
    cat $source_m2 >> $target_m2
    cat $source_src >> $target_src
    cat $source_tgt >> $target_tgt
done