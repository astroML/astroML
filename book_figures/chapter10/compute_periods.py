"""
Compute periods for the LINEAR data
-----------------------------------
"""
from __future__ import print_function

from time import time
import numpy as np
from astroML.datasets import fetch_LINEAR_sample
from astroML.time_series import lomb_scargle, multiterm_periodogram, \
    search_frequencies

import sqlite3

Ngrid = 50000
DATABASE = 'periods.db'

data = fetch_LINEAR_sample()

# set up a database to hold periods
con = sqlite3.connect(DATABASE)

with con:
    cur = con.cursor()

    try:
        cur.execute("CREATE TABLE Periods(id INT, omega FLOAT)")
    except:
        pass

    for count, id in enumerate(data.ids):
        # only compute period if it hasn't been computed before
        cur.execute("SELECT * from Periods WHERE id = %i" % id)
        res = cur.fetchall()

        if len(res) > 0:
            print(res[0])

        else:
            print("computing period for id = {0} ({1} / {2})"
                  "".format(id, count + 1, len(data.ids))))

            lc = data[id]

            t0 = time()
            omega, power = search_frequencies(lc[:, 0], lc[:, 1], lc[:, 2],
                                              LS_func=multiterm_periodogram,
                                              n_save=5, n_retry=5,
                                              n_eval=10000,
                                              LS_kwargs=dict(n_terms=5))
            omega_best = omega[np.argmax(power)]
            t1 = time()
            print(" - execution time: %.2g sec" % (t1 - t0))

            # insert value and commit to disk
            cur.execute("INSERT INTO Periods VALUES(%i, %f)"
                        % (id, omega_best))
            con.commit()

    con.close()

    #cur.execute("SELECT * from Periods")
    #print(cur.fetchall())
