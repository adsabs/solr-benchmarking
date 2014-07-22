import os
from locust import HttpLocust, TaskSet, task
import random
import uuid
import time
import datetime

with open('etc/urls.txt','r') as fp:
  urls = [L.strip().replace('http://localhost:9000','') for L in fp.readlines() if L]
FILE='%s.dat' % datetime.datetime.isoformat(datetime.datetime.now())
OUTPUT_DIR = 'data'

class UserBehavior(TaskSet):
    def on_start(self):
      """ on_start is called when a Locust start before any task is scheduled """
      self.myid='%s' % uuid.uuid4()

    def randomUrl(self):
      return random.choice(urls)

    @task(1)
    def query(self):
      u = self.randomUrl()
      start = time.time()
      resp=self.client.get(u,name="/select?q=...")
      end = time.time()
      with open(os.path.join(OUTPUT_DIR,FILE),'a') as fp:
        r=resp.json()
        fp.write('%s %s %s %s %s\n' % (time.time(),(end-start),r['responseHeader']['QTime'],r['response']['numFound'],self.myid))

class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    min_wait=10
    max_wait=50
