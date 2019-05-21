# run these beforehand
# git clone https://github.com/wronnyhuang/sharpcifar
# pip install --upgrade comet-ml
nohup python curvature_mean.py -gpu=0 -batchsize=8192 -nworker=8 -projname=svhn-poisonfrac-curv-2 -name=svhn-poisonfrac-2 -url=https://www.dropbox.com/sh/5hxwmtxr3pdbvxt/AAAWVKmtijzDeeHiCDOeuY21a?dl=0 &
nohup python curvature_mean.py -gpu=1 -batchsize=8192 -nworker=8 -projname=svhn-poisonfrac-curv-2 -name=svhn-poisonfrac-3 -url=https://www.dropbox.com/sh/qohwprpjqgtpvsl/AACr078AhOeN8DeadE5ihDisa?dl=0 &
nohup python curvature_mean.py -gpu=2 -batchsize=8192 -nworker=8 -projname=svhn-poisonfrac-curv-2 -name=svhn-poisonfrac-4 -url=https://www.dropbox.com/sh/b4jjw2djsi3g2c0/AAA5zGXe96f3g23uxp2LMuGNa?dl=0 &
nohup python curvature_mean.py -gpu=0 -batchsize=8192 -nworker=8 -projname=svhn-poisonfrac-curv-2 -name=svhn-poisonfrac-5 -url=https://www.dropbox.com/sh/8wyy2hur2ubv102/AACBZd4edpZTXbp4w00breqxa?dl=0 &
nohup python curvature_mean.py -gpu=0 -batchsize=8192 -nworker=8 -projname=svhn-poisonfrac-curv-2 -name=svhn-poisonfrac-6 -url=https://www.dropbox.com/sh/wh8xyc4jmvicmy2/AAAIOyBIhmh_oibmwY4OmzzKa?dl=0 &
nohup python curvature_mean.py -gpu=1 -batchsize=8192 -nworker=8 -projname=svhn-poisonfrac-curv-2 -name=svhn-poisonfrac-7 -url=https://www.dropbox.com/sh/8x3vspe62vlpjfr/AABXBlCBs0hbQ1-rE4fXscdva?dl=0 &
nohup python curvature_mean.py -gpu=2 -batchsize=8192 -nworker=8 -projname=svhn-poisonfrac-curv-2 -name=svhn-poisonfrac-8 -url=https://www.dropbox.com/sh/m1erb0usr23nqnf/AABhs1JhJrUItDPnVN6v1Zmca?dl=0 &
nohup python curvature_mean.py -gpu=0 -batchsize=8192 -nworker=8 -projname=svhn-poisonfrac-curv-2 -name=svhn-largebatch-1 -url=https://www.dropbox.com/sh/ebweyvbsx7wo7t0/AABiBH-39VeZMoSXkMKtTfUZa?dl=0 &
