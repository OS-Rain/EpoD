import sys
import os
sys.path.append('..')
sys.path.append(os.path.abspath(__file__ + '/../../..'))
from EpoD.config import args
from EpoD.utils.mutils import *
from EpoD.utils.data_util import *
from EpoD.utils.util import init_logger
import warnings
warnings.simplefilter("ignore")

# load data
args,data=load_data(args)

# pre-logs
log_dir=args.log_dir
init_logger(prepare_dir(log_dir) + 'log.txt')
info_dict=get_arg_dict(args)

# Runner
from EpoD.runner import Runner
from EpoD.model import EpoD
model = EpoD(args=args).to(args.device)
runner = Runner(args,model,data)
results = runner.run()

# post-logs
measure_dict=results
info_dict.update(measure_dict)
json.dump(info_dict, open(osp.join(log_dir, 'info.json'), 'w'))