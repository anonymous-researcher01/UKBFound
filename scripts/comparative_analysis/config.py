
from datetime import datetime
current_date = datetime.now()
formatted_date = current_date.strftime("%m%d")
folder=formatted_date
imglist=[0]
catlist=[1,2,3,4,5,6]
rawXlocation='../../results/Process_missingness/'
Xblocklocation='../../results/cache/xblock/main/'
def ava_gpus(g):
    ava=[1,2,3,4,5,6,7]
    g=g%len(ava)
    return ava[g]       
NNdepth=2
NNshape=100