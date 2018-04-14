#!/usr/bin/env python
import os
import re
#from smatch import compute_score as score

PATH = '/data/soumye/cnn/stories/'
cnt = 0
for filename in os.listdir(PATH):
    if filename.endswith('.story'):
        #print(filename)
        f = open(PATH+filename, 'r')
        data = f.read()
        data = data.split('@highlight')
        f.close()

        source_text = data[0]
        data[0] = data[0].split('\n')
        source_text = ''.join(x + '\n' for x in data[0] if x is not '')
        with open('/data/`/UGP-Code/summ_data/text/'+filename, 'w+') as f:
            f.write(source_text)

            f.seek(0)
            num_lines = 0
            for line in f:
                num_lines += 1

        for idx, y in enumerate(data[1:]):
            f = open('/data/soumye/UGP-Code/summ_data/summ/'+filename+'_'+str(idx), 'w+')
            y = y.split('\n')
            summ_text = ''.join(x + '\n' for x in y if x is not '')
            for i in range(int(num_lines)):
                f.write(summ_text)
            f.close()

        cnt += 1
        if cnt == 100:
            break