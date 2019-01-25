import threading
import subprocess

class Worker(threading.Thread):

  def __init__(self, command):
    threading.Thread.__init__(self) # required: call constructor of threading class
    self.command = command

  def run(self):
    '''runs when Worker.start() is called. put the bash call to main.py here'''
    print('META: Starting ' + self.command)
    output = subprocess.run(self.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    print('META: Done with ' + self.command)

def spawn(base, machines):

  hostlog = '/root/misc/hostname.log'
  hostname = open(hostlog).read().strip('\n')
  variants = machines[hostname]
  commands = [' '.join([base, variant]) for variant in variants]

  workers = [Worker(command) for command in commands]
  for worker in workers:
    worker.start()
  for worker in workers:
    worker.join()
  print('META: all done!')
