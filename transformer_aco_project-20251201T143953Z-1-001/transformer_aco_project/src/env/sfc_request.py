import random

def generate_sfc(k=4):
    vnfs = []
    
    for _ in range(k):
        vnfs.append({
            "cpu": random.randint(1,5),
            "ram": random.randint(1,5),
            "bw": random.randint(1,5),
            "energy": random.randint(1,5),
            "duration": random.randint(5,10)
        })
    
    return vnfs
