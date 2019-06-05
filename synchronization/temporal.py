import numpy as np

from os.path import join

class Timestamp:
  '''
  A very minimal class containing a pair of values:
    id   : some string identifying the frame
    time : a time instant (e.g. time since epoch in milliseconds)
  '''
  def __init__(self, id, time):
    self.id = id
    self.time = time

def sync(a, b, eps = 50, verbose=False):
  '''
  Synchronizes two series of timestamps. Two timestamps are matched based on a temporal distance criteria.
  However, they will be only listed as a match if the temporal distance between them is smaller than "eps" time units.
  :param a: first time steries of timestamps
  :param b: second time series of timestamps
  :param eps: threshold to consider a match of two timestamps
  :param verbose: print debugging information
  :return:
  '''
  ab = []

  # the master series will be the shorter one and its timestamps will try to be matched against the slave's timestamps
  master, slave = (a, b) if len(a) < len(b) else (b, a)

  j = 0  # pointer over the slave series

  for i, ts_m in enumerate(master):  # iterate over master
    matches_i = []  # list of matches for the i-th timestamp in master
    while ((j < len(slave)) and (slave[j].time < (ts_m.time + eps))):
      dist = abs(ts_m.time - slave[j].time)
      if dist < eps:
        matches_i.append((j,dist))
      j += 1

    if len(matches_i) > 0:
      m_best = matches_i[0]
      for k in range(len(matches_i)):
        if matches_i[k][1] < m_best[1]:
          m_best = matches_i[k]

      ab.append( (master[i], slave[m_best[0]]) if len(a) < len(b) else (slave[m_best[0]], master[i]) )
      j = m_best[0]

  if verbose:
    diff_total = .0
    for i in range(len(ab)):
      diff = ab[i][0].time - ab[i][1].time
      print(f'<"{ab[i][0].id}",{ab[i][0].time}>, <"{ab[i][1].id}",{ab[i][1].time}>, {diff}')
      diff_total += abs(diff)

    print(f"Average abs diff between timestamps = {diff_total/len(ab)}")

  return ab

if __name__ == "__main__":

  # Illustrative example

  # One series of timestamps temporally regularly spaced by 10 time units

  log_a = [
    Timestamp("a_01", 2), # "a_01" could be the frame id
    Timestamp("a_02", 12),
    Timestamp("a_03", 22),
    Timestamp("a_04", 32),
    Timestamp("a_05", 42),
    Timestamp("a_06", 52),
  ]

  # Another series of timestamps, these are unevenly spaced in time
  log_b = [
    Timestamp("b_01", 0),
    Timestamp("b_02", 8),
    Timestamp("b_03", 16),
    Timestamp("b_04", 30),
    Timestamp("b_05", 50),
    Timestamp("b_06", 55),
  ]

  # timestamps are matched and cannot be temporally separated more than "eps" time units

  lob_synced_ab = sync(log_a, log_b, eps=3, verbose=True)  # verbose to see matches

  lob_synced_ab = sync(log_a, log_b, eps=np.inf, verbose=True)  # verbose to see matches

