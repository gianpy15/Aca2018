import sys


start_sign = "==="
end_sign = "---"
fpsign = "fp"
quantsign = "int"

if __name__ == '__main__':
    opt = ''
    if len(sys.argv) > 1:
        opt = sys.argv[1]

    parsing = False
    tp = None
    for line in sys.stdin:
        line = str.strip(line)
        if not parsing:
            if line == start_sign+fpsign:
                parsing = True
                tp = "fp32"
            elif line == start_sign+quantsign:
                parsing = True
                tp = "int8"
            continue
        else:
            if line == end_sign:
                parsing = False
                continue
            elems = line.split(',')
            if len(elems) >= 4:
                if opt == '-h':
                    out0 = tp + ' '*(15 - len(tp))
                    out1 = elems[2] + ' '*(15-len(elems[2]))
                    out2 = elems[3] + ' '*(15-len(elems[3]))
                    out3 = elems[-1]
                    print("%s%s%s%s" % (out0, out1, out2, out3))
                else:
                    print("%s,%s,%s,%s" % (tp, elems[2], elems[3], elems[-1]))


