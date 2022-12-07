import subprocess
import shlex

accent_map = {
        "ABA":"arabic","SKA":"arabic","YBAA":"arabic","ZHAA":"arabic", 
        "BWC":"chinese","LXC":"chinese","NCC":"chinese","TXHC":"chinese", 
        "ASI":"hindi","RRBI":"hindi","SVBI":"hindi","TNI":"hindi", 
        "HJK":"korean","HKK":"korean","YDCK":"korean","YKWK":"korean",
        "EBVS":"spanish","ERMS":"spanish","MBMPS":"spanish","NJS":"spanish",
        "HQTV":"vietnamese","PNV":"vietnamese","THV":"vietnamese","TLV":"vietnamese"
       }

for speaker, accent in accent_map.items():
    print(speaker, accent)
    for other_speaker, other_accent in accent_map.items():
        if other_speaker != speaker and other_accent == accent:
            subprocess.call(shlex.split(f'bash run_within_individual.sh {speaker} {other_speaker}'))

