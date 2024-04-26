
while test $# -gt 0; 
do
    case "$1" in
        --benchmark_config)
            shift
            BENCHMARK_CONFIG=$1
            shift
            ;;
        --mem_size)
            shift
            MEM_SIZE=$1
            shift
            ;;
        --lr)
            shift
            LR=$1
            shift
            ;;
        --bs)
            shift
            BS=$1
            shift
            ;;
        *)
            echo "$1 is not a recognized flag! Use --yaml_config, --dset_rootpath, --save_path, --exp_name."
            exit 1;
            ;;
    esac
done  

STRATEGIES=("./reproduce/strategy/er.yml" "./reproduce/strategy/er_gem.yml", "./reproduce/strategy/er_agem.yml")

for STRAT in STRATEGIES
do 
    for SEED in 142 152 162 172 182
    do
        python train.py --benchmark_config $BENCHMARK_CONFIG --strategy_config $STRAT --mem_size $MEM_SIZE --lr $LR --bs $BS --seed $SEED --save_path ./results/
    done 
done
