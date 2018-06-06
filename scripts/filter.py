import sys

if __name__ == '__main__':
    log = sys.argv[1]

    for line in log.split('\n'):
        elems = line.split(',')
        if len(elems) >= 4:
            out1 = elems[2] + ' '*(15-len(elems[2]))
            out2 = elems[3] + ' '*(15-len(elems[3]))
            out3 = elems[-1]
            print("%s%s%s" % (out1, out2, out3))

