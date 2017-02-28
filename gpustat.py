#!/usr/bin/python

from subprocess import check_output, CalledProcessError
import sys
import os
import time


# Ordered by Priority
GPU_names = ['GeForce GTX TITAN X', 'GeForce GTX 1080', 'GeForce GTX 980 Ti', 'Tesla K40c', 'TITAN X (Pascal)']

def main(hostname):
  if hostname == "UTS7":
    gpu_id_mapping = {
        0: 3,
        1: 0,
        2: 1,
        3: 2,}
  elif hostname == "UTS2":
    gpu_id_mapping = {
        0: 3,
        1: 2,
        2: 1,
        3: 0,}
  elif hostname == "UTS6":
    gpu_id_mapping = {
        0: 3,
        1: 2,
        2: 1,
        3: 0,}
  gpu_query_columns = ('index', 'uuid', 'name', 'temperature.gpu',
                        'utilization.gpu', 'memory.used', 'memory.total')

  smi_output = check_output(
      r'nvidia-smi --query-gpu={query_cols} --format=csv,noheader,nounits'.format(
          query_cols=','.join(gpu_query_columns)
      ), shell=True).decode().strip()

  gpu_list = []
  for line in smi_output.split('\n'):
    if not line: continue
    query_results = line.split(',')
    stat_dict = {}
    for col_name, col_value in zip(gpu_query_columns, query_results):
      stat_dict[col_name] = col_value.strip()
    gpu_list.append(stat_dict)

  gpu_priors = {
      0: 5,
      1: 5,
      2: 5,
      3: 5,
  }
  prior = -1
  gpu_idx = -1
  for i in get_available_gpu(gpu_list):
    i = int(i)
    p = gpu_priors[i]
    if p > prior:
      prior = p
      gpu_idx = i
  if gpu_idx > -1:
    print(gpu_idx)
  else:
    print("No available GPU")

def get_available_gpu(stats):
  gpus = []
  for gpu in stats:
    if 1. * int(gpu['memory.used']) < 200 and int(gpu['temperature.gpu']) < 70:
      gpu_id = int(gpu['index'])
      gpus.append(gpu_id_mapping[gpu_id])
  return gpus


def check_temp():
  for gpu_stat in stats:
    print(int(gpu_stat['temperature.gpu']))
    if int(gpu_stat['temperature.gpu']) > 90:
      print('error')


if __name__ == '__main__':
  main(hostname)
