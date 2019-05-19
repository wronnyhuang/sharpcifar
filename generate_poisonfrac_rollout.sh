# run these beforehand
# git clone https://github.com/wronnyhuang/sharpcifar
# pip install --upgrade comet-ml

# adjust the -gpu argument to another gpu index if needed
# two processes per gpu please (so 32 gpus total as there are 64 processes here)
# the longer you run it the better but 24 hours would be good

python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-2  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/5hxwmtxr3pdbvxt/AAAWVKmtijzDeeHiCDOeuY21a?dl=0
python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-3  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/qohwprpjqgtpvsl/AACr078AhOeN8DeadE5ihDisa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-4  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/b4jjw2djsi3g2c0/AAA5zGXe96f3g23uxp2LMuGNa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-5  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8wyy2hur2ubv102/AACBZd4edpZTXbp4w00breqxa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-6  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/wh8xyc4jmvicmy2/AAAIOyBIhmh_oibmwY4OmzzKa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-7  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8x3vspe62vlpjfr/AABXBlCBs0hbQ1-rE4fXscdva?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-8a -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/m1erb0usr23nqnf/AABhs1JhJrUItDPnVN6v1Zmca?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-largebatch-1  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/ebweyvbsx7wo7t0/AABiBH-39VeZMoSXkMKtTfUZa?dl=0

python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-2  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/5hxwmtxr3pdbvxt/AAAWVKmtijzDeeHiCDOeuY21a?dl=0
python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-3  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/qohwprpjqgtpvsl/AACr078AhOeN8DeadE5ihDisa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-4  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/b4jjw2djsi3g2c0/AAA5zGXe96f3g23uxp2LMuGNa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-5  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8wyy2hur2ubv102/AACBZd4edpZTXbp4w00breqxa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-6  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/wh8xyc4jmvicmy2/AAAIOyBIhmh_oibmwY4OmzzKa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-7  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8x3vspe62vlpjfr/AABXBlCBs0hbQ1-rE4fXscdva?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-8a -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/m1erb0usr23nqnf/AABhs1JhJrUItDPnVN6v1Zmca?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-largebatch-1  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/ebweyvbsx7wo7t0/AABiBH-39VeZMoSXkMKtTfUZa?dl=0

python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-2  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/5hxwmtxr3pdbvxt/AAAWVKmtijzDeeHiCDOeuY21a?dl=0
python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-3  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/qohwprpjqgtpvsl/AACr078AhOeN8DeadE5ihDisa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-4  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/b4jjw2djsi3g2c0/AAA5zGXe96f3g23uxp2LMuGNa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-5  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8wyy2hur2ubv102/AACBZd4edpZTXbp4w00breqxa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-6  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/wh8xyc4jmvicmy2/AAAIOyBIhmh_oibmwY4OmzzKa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-7  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8x3vspe62vlpjfr/AABXBlCBs0hbQ1-rE4fXscdva?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-8a -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/m1erb0usr23nqnf/AABhs1JhJrUItDPnVN6v1Zmca?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-largebatch-1  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/ebweyvbsx7wo7t0/AABiBH-39VeZMoSXkMKtTfUZa?dl=0

python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-2  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/5hxwmtxr3pdbvxt/AAAWVKmtijzDeeHiCDOeuY21a?dl=0
python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-3  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/qohwprpjqgtpvsl/AACr078AhOeN8DeadE5ihDisa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-4  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/b4jjw2djsi3g2c0/AAA5zGXe96f3g23uxp2LMuGNa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-5  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8wyy2hur2ubv102/AACBZd4edpZTXbp4w00breqxa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-6  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/wh8xyc4jmvicmy2/AAAIOyBIhmh_oibmwY4OmzzKa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-7  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8x3vspe62vlpjfr/AABXBlCBs0hbQ1-rE4fXscdva?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-8a -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/m1erb0usr23nqnf/AABhs1JhJrUItDPnVN6v1Zmca?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-largebatch-1  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/ebweyvbsx7wo7t0/AABiBH-39VeZMoSXkMKtTfUZa?dl=0

python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-2  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/5hxwmtxr3pdbvxt/AAAWVKmtijzDeeHiCDOeuY21a?dl=0
python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-3  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/qohwprpjqgtpvsl/AACr078AhOeN8DeadE5ihDisa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-4  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/b4jjw2djsi3g2c0/AAA5zGXe96f3g23uxp2LMuGNa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-5  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8wyy2hur2ubv102/AACBZd4edpZTXbp4w00breqxa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-6  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/wh8xyc4jmvicmy2/AAAIOyBIhmh_oibmwY4OmzzKa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-7  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8x3vspe62vlpjfr/AABXBlCBs0hbQ1-rE4fXscdva?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-8a -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/m1erb0usr23nqnf/AABhs1JhJrUItDPnVN6v1Zmca?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-largebatch-1  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/ebweyvbsx7wo7t0/AABiBH-39VeZMoSXkMKtTfUZa?dl=0

python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-2  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/5hxwmtxr3pdbvxt/AAAWVKmtijzDeeHiCDOeuY21a?dl=0
python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-3  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/qohwprpjqgtpvsl/AACr078AhOeN8DeadE5ihDisa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-4  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/b4jjw2djsi3g2c0/AAA5zGXe96f3g23uxp2LMuGNa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-5  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8wyy2hur2ubv102/AACBZd4edpZTXbp4w00breqxa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-6  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/wh8xyc4jmvicmy2/AAAIOyBIhmh_oibmwY4OmzzKa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-7  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8x3vspe62vlpjfr/AABXBlCBs0hbQ1-rE4fXscdva?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-8a -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/m1erb0usr23nqnf/AABhs1JhJrUItDPnVN6v1Zmca?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-largebatch-1  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/ebweyvbsx7wo7t0/AABiBH-39VeZMoSXkMKtTfUZa?dl=0

python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-2  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/5hxwmtxr3pdbvxt/AAAWVKmtijzDeeHiCDOeuY21a?dl=0
python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-3  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/qohwprpjqgtpvsl/AACr078AhOeN8DeadE5ihDisa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-4  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/b4jjw2djsi3g2c0/AAA5zGXe96f3g23uxp2LMuGNa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-5  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8wyy2hur2ubv102/AACBZd4edpZTXbp4w00breqxa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-6  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/wh8xyc4jmvicmy2/AAAIOyBIhmh_oibmwY4OmzzKa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-7  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8x3vspe62vlpjfr/AABXBlCBs0hbQ1-rE4fXscdva?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-8a -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/m1erb0usr23nqnf/AABhs1JhJrUItDPnVN6v1Zmca?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-largebatch-1  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/ebweyvbsx7wo7t0/AABiBH-39VeZMoSXkMKtTfUZa?dl=0

python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-2  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/5hxwmtxr3pdbvxt/AAAWVKmtijzDeeHiCDOeuY21a?dl=0
python surface_mean.py -gpu=0 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-3  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/qohwprpjqgtpvsl/AACr078AhOeN8DeadE5ihDisa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-4  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/b4jjw2djsi3g2c0/AAA5zGXe96f3g23uxp2LMuGNa?dl=0
python surface_mean.py -gpu=1 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-5  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8wyy2hur2ubv102/AACBZd4edpZTXbp4w00breqxa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-6  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/wh8xyc4jmvicmy2/AAAIOyBIhmh_oibmwY4OmzzKa?dl=0
python surface_mean.py -gpu=2 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-7  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/8x3vspe62vlpjfr/AABXBlCBs0hbQ1-rE4fXscdva?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-poisonfrac-8a -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/m1erb0usr23nqnf/AABhs1JhJrUItDPnVN6v1Zmca?dl=0
python surface_mean.py -gpu=3 -projname=svhn-poisonfrac-rollout-1 -name=svhn-largebatch-1  -batchsize=256 -nworker=2 -seed=102 -url=https://www.dropbox.com/sh/ebweyvbsx7wo7t0/AABiBH-39VeZMoSXkMKtTfUZa?dl=0
