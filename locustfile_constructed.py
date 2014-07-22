import os
from locust import HttpLocust, TaskSet, task
import random
import uuid
import time
import datetime

OUTPUT_DIR = 'data'
FILE='%s.dat' % datetime.datetime.isoformat(datetime.datetime.now())

DATADIR = os.path.join('etc','constructed','live')
QUERIES = []
for f in os.listdir(DATADIR):
  with open(os.path.join(DATADIR,f)) as fp:
    lines = [L for L in fp.readlines() if L and not L.startswith('#')]
  for line in lines:
    line = line.split("\t")
    Q = line[0]
    QUERIES.append( (Q,f) )

print "Will run %s queries found in %s" % (len(QUERIES),DATADIR)

class UserBehavior(TaskSet):
    def on_start(self):
      """ on_start is called when a Locust start before any task is scheduled """
      self.myid='%s' % uuid.uuid4()

    def getNextQuery(self):
      for Q in iter(QUERIES):
        yield Q

    @task(len(QUERIES))
    def query(self):
      u = "/solr/select?q=%s" % self.getNextQuery()
      start = time.time()
      resp=self.client.get(u[0],name=u[1])
      end = time.time()
      with open(os.path.join(OUTPUT_DIR,FILE),'a') as fp:
        r=resp.json()
        fp.write('%s %s %s %s %s %s\n' % (time.time(),(end-start),r['responseHeader']['QTime'],r['response']['numFound'],self.myid,u[1]))

class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    min_wait=0
    max_wait=0