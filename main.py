# import time
# import numpy as np
# import matplotlib.pyplot as plt
#
# fig, axs = plt.subplots(1,1)
#
#
# t0 = time.time()
#
# n = 121
# m = 52
#
# np.save('./tools/weights1_nm.npy', np.ones((n, m)))
#
# for i in range(100):
#     arr = np.load('./tools/weights1_nm.npy')
#     arr += 0.01
#     np.save('./tools/weights1_nm.npy', arr)
#
# t1 = time.time()
# print(t1-t0)
#
# eg_frame = np.load('./tools/example_frame.npy')
# axs.imshow(eg_frame)
# plt.show()

from initialisation import find_spot_ampls_200
import numpy as np
import time


y_center_indxs_200 = np.load('./tools/y_center_indxs_200.npy')
x_edge_indxs_200 = np.load('./tools/x_edge_indxs_200.npy')

arr = np.random.rand(1024, 120)
#
meas = find_spot_ampls_200(arr)

print(meas)

# Print iterations progress
def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='#', printend=""):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + ' ' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printend)
    # Print New Line on Complete
    if iteration == total:
        print()


# A List of Items
items = list(range(0, 57))
l = len(items)

# Initial call to print 0% progress
progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
for i, item in enumerate(items):
    # Do stuff...
    time.sleep(0.1)
    # Update Progress Bar
    progress_bar(i + 1, l, prefix='Progress:', suffix='Complete', length=50)