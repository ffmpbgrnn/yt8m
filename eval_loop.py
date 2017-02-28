import os
import sys

run_id = sys.argv[1]
src_dir = "/data/D2DCRC/linchao/YT/log/{}".format(run_id)
while True:
  files = os.listdir(src_dir)
  model_ids = []
  for f in files:
    if "meta" in f:
      model_id = int(f.split('-')[1].split('.')[0])
      validate_f = "model.ckpt-{}.validate.log".format(model_id)
      validate_f = os.path.join(src_dir, validate_f)
      if not os.path.exists(validate_f):
        model_ids.append(model_id)
  model_ids = sorted(model_ids)

  model_id = model_ids[-1]
  cmd = "cd /data/state/linchao/yt8m_src_log/{0}/src/; ./run.sh eval /data/D2DCRC/linchao/YT/log/{0}/model.ckpt-{1} > model.ckpt-{1}.validate.log 2>&1".format(run_id, model_id)
  os.system(cmd)
