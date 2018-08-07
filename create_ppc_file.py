import sys
from time import gmtime, strftime


def create_ppc_file(casename, ppc):

    file = 'from numpy import array\n' \
            'def '+casename+'():\n' \
            '\tppc = {"version": \'2\'}\n' \
            '\tppc["baseMVA"] = ' + str(ppc["baseMVA"]) + '\n'

    file += '\tppc["bus"] = array([\n'
    r_n=0
    for r in ppc["bus"]:
        r_n += 1
        row = '\t\t['
        i_n=0
        for i in r:
            i_n += 1
            row += str(i)
            if i_n==ppc["bus"].shape[1]:
                row += ']'
                if r_n != ppc["bus"].shape[0]:
                    row += ','
            else:
                row += ', '
        if r_n==ppc["bus"].shape[0]:
            file += row + '\n\t])\n\n'
            break
        else:
            file += row + '\n'

    file += '\tppc["gen"] = array([\n'
    r_n=0
    for r in ppc["gen"]:
        r_n += 1
        row = '\t\t['
        i_n=0
        for i in r:
            i_n += 1
            row += str(i)
            if i_n==ppc["gen"].shape[1]:
                row += ']'
                if r_n != ppc["gen"].shape[0]:
                    row += ','
            else:
                row += ', '
        if r_n==ppc["gen"].shape[0]:
            file += row + '\n\t])\n\n'
            break
        else:
            file += row + '\n'

    file += '\tppc["branch"] = array([\n'
    r_n=0
    for r in ppc["branch"]:
        r_n += 1
        row = '\t\t['
        i_n=0
        for i in r:
            i_n += 1
            row += str(i)
            if i_n==ppc["branch"].shape[1]:
                row += ']'
                if r_n != ppc["branch"].shape[0]:
                    row += ','
            else:
                row += ', '
        if r_n==ppc["branch"].shape[0]:
            file += row + '\n\t])\n\n'
            break
        else:
            file += row + '\n'

    file += '\tppc["gencost"] = array([\n'
    r_n=0
    for r in ppc["gencost"]:
        r_n += 1
        row = '\t\t['
        i_n=0
        for i in r:
            i_n += 1
            row += str(i)
            if i_n==ppc["gencost"].shape[1]:
                row += ']'
                if r_n != ppc["gencost"].shape[0]:
                    row += ','
            else:
                row += ', '
        if r_n==ppc["gencost"].shape[0]:
            file += row + '\n\t])\n\n'
            break
        else:
            file += row + '\n'

    file += '\n\treturn ppc\n'

    fd = open('pcc_check/' + str(casename) + '.py', 'w')
    orig_stdout = sys.stdout
    sys.stdout = fd
    print('"""File created: '+strftime("%Y-%m-%d %H:%M:%S", gmtime())+'"""\n')
    print(file)
    sys.stdout = orig_stdout
    fd.close()

    return True
