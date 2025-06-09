for ((i = 1; i<=501;i+=10))
do
    MODEL_PATH="D:\cs224rPython\PPO_hard_XL_everything\FINAL_PPO_simple_stay_alive_${i}.pth"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Model file $MODEL_PATH does not exist. Exiting loop."
        break
    fi
    python eval_all.py -e="D:\FM-Slippi-2.3.1-Win" --iso="D:\cs224rPython\SmashIso.nkit.iso" --model="$MODEL_PATH"
    sleep 1
    python eval_all.py -e="D:\FM-Slippi-2.3.1-Win" --iso="D:\cs224rPython\SmashIso.nkit.iso" --model="$MODEL_PATH"
    sleep 1
done

