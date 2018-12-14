echo 'coding on macbook pycharm; rsync upload to hpcc; LSF submit to GPU'



rsync -avhz --progress ../mnist rl44w@ghpcc06.umassrc.org:/home/rl44w/ccount/runs/classifier/

