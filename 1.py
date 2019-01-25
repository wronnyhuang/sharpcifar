from utils_worker import spawn

base = 'python main.py -pretrain_dir=ckpt/pre60k-100k -cifar100 -noaugment -weight_decay=0 -randname -randvec -specexp=10'
machines = dict(tricky=  [
                          '-gpu=0 -speccoef=5e-1',
                          '-gpu=1 -speccoef=1',
                          '-gpu=2 -speccoef=2',
                          '-gpu=3 -speccoef=3',
                          ],
                tomg3264=[
                          '-gpu=1 -speccoef=6',
                          '-gpu=2 -speccoef=10',
                          '-gpu=3 -speccoef=16',
                          ],
                max=     []
                )
spawn(base, machines)
