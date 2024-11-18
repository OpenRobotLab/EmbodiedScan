'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''


def get_eta(start, end, extra, num_left):
    exe_s = end - start
    eta_s = (exe_s + extra) * num_left
    eta = {'h': 0, 'm': 0, 's': 0}
    if eta_s < 60:
        eta['s'] = int(eta_s)
    elif eta_s >= 60 and eta_s < 3600:
        eta['m'] = int(eta_s / 60)
        eta['s'] = int(eta_s % 60)
    else:
        eta['h'] = int(eta_s / (60 * 60))
        eta['m'] = int(eta_s % (60 * 60) / 60)
        eta['s'] = int(eta_s % (60 * 60) % 60)

    return eta


def decode_eta(eta_sec):
    eta = {'h': 0, 'm': 0, 's': 0}
    if eta_sec < 60:
        eta['s'] = int(eta_sec)
    elif eta_sec >= 60 and eta_sec < 3600:
        eta['m'] = int(eta_sec / 60)
        eta['s'] = int(eta_sec % 60)
    else:
        eta['h'] = int(eta_sec / (60 * 60))
        eta['m'] = int(eta_sec % (60 * 60) / 60)
        eta['s'] = int(eta_sec % (60 * 60) % 60)

    return eta
