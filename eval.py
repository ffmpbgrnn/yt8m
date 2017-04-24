import os
import math
import sys
import subprocess
import time

def eval_local():
  while True:
    files = os.listdir(train_dir)
    model_ids = []
    evaluated_model_ids = []
    for f in files:
      if "meta" in f:
        model_id = int(f.split('-')[1].split('.')[0])
        validate_f = "model.ckpt-{}.validate.log".format(model_id)
        validate_f = os.path.join(train_dir, validate_f)
        if not os.path.exists(validate_f):
          model_ids.append(model_id)
        else:
          evaluated_model_ids.append(model_id)
    model_ids = sorted(model_ids)

    num_models = len(model_ids)
    next_model_id = None
    if num_models == 0:
      pass
    elif num_models < 10:
      next_model_id = model_ids[int(num_models / 2)]
    else:
      step_size = int(math.ceil(1. * num_models / 10))
      if len(evaluated_model_ids) == 0:
        next_model_id = model_ids[5 * step_size]
      else:
        for i in xrange(10):
          model_step_range = model_ids[i * step_size: (i + 1) * step_size]
          model_has_evaludated = False
          for eid in evaluated_model_ids:
            if eid > model_step_range[0] and eid < model_step_range[-1]:
              model_has_evaludated = True
              next_model_id = model_step_range[int(len(model_step_range) / 2)]
              break
          if not model_has_evaludated:
            break
    if next_model_id is not None:
      if not os.path.exists(os.path.join(train_dir, 'src')):
        print("ssh uts_cluster 'cd /projects/D2DCRC/linchaoz/YT/log/{}; tar xf src.tar'".format(run_id))
        os.system("ssh uts_cluster 'cd /projects/D2DCRC/linchaoz/YT/log/{}; tar xf src.tar'".format(run_id))
      cmd = "cd {2}/src/; ./run.sh eval {0}/model.ckpt-{1} > {0}/model.ckpt-{1}.validate.log 2>&1".format(train_dir, next_model_id, code_dir)
      print(cmd)
      # try:
      if True:
        os.system(cmd)
      # except:
        # pass
    print("Sleeping")
    time.sleep(5)

def get_score():
  train_dir = "/Users/ffmpbgrnn/ytlog/log/{}".format(run_id)
  files = os.listdir(train_dir)
  scores = {}
  for f in files:
    if 'validate.log' in f:
      model_id = int(f.split('-')[1].split('.')[0])
      with open(os.path.join(train_dir, f)) as fin:
        for line in fin.readlines():
          line = line.strip()
          if "epoch/eval number" in line:
            lines = line.split("|")
            res = [lines[1], lines[3], lines[4]]
            hit_1, MAP, GAP = [float(_.split(':')[1].strip()) for _ in res]
            scores[model_id] = {"Hit@1": hit_1, "MAP": MAP, "GAP": GAP}
            break
  model_ids = sorted(scores.keys())
  for model_id in model_ids:
    score = scores[model_id]
    print("{}: {}".format(model_id, score))

def find_machines():
  for server_id in [2, 6, 7]:
    try:
      output = subprocess.check_output("ssh uts{} find /data/state/linchao/yt8m_src_log/ -maxdepth 1 | grep {}".format(server_id, run_id), shell=True)
      docker_flags = "-v /data:/data -v /home/linchao/docker/docker_home/:/home/linchao --user linchao -w /home/linchao -d linchao /bin/zsh -c"
      # task_cmd = 'ls -alh /data/uts700/linchao/yt8m/YT/src'.format(run_id)
      task_cmd = 'cd /data/uts700/linchao/yt8m/YT/src && python2.7 eval.py eval_local {}'.format(run_id)
      cmd = '''echo "nvidia-docker run -h UTS{0} {1} '{2}'" | ssh uts{0} /bin/zsh -'''.format(server_id, docker_flags, task_cmd)
      print(cmd)
      os.system(cmd)
      # print(server_id, output.strip())
    except:
      pass

def eval_remote():
  # for server_id in [2, 6, 7]:
  for server_id in [6, 7]:
    try:
      output = subprocess.check_output("ssh uts{0} 'python2.7 {1}/gpustat.py UTS{0} eval'".format(server_id, src_dir), shell=True).strip()
      gpu_id = int(output)
      docker_flags = "-v /data:/data -v /home/linchao/docker/docker_home/:/home/linchao --user linchao -w /home/linchao -d linchao /bin/zsh -c"
      task_cmd = 'cd {} && python2.7 eval.py eval_local {}'.format(src_dir, run_id)
      cmd = '''echo "nvidia-docker run -h UTS{0} {1} '{2}'" | ssh uts{0} /bin/zsh -'''.format(server_id, docker_flags, task_cmd)
      print(cmd)
      os.system(cmd)
      break
    except:
      pass

run_id = sys.argv[2]
train_dir = "/data/D2DCRC/linchao/YT/log/{}".format(run_id)
code_dir = "/data/D2DCRC/linchao/YT/log/{}".format(run_id)
src_dir = "/data/uts700/linchao/yt8m/YT/src"
if sys.argv[1] == "eval_local":
  eval_local()
elif sys.argv[1] == "get_score":
  get_score()
elif sys.argv[1] == "eval":
  eval_remote()
  # find_machines()
