import random, math, numpy, copy
from scipy.stats import t, f

N = 15
m = 3
p = 0.95
q = 1-p
d = 11
f1 = m-1
f2 = N
f3 = f1*f2
f4 = N-d


x1min = -7
x1max = 8
x10 = (x1max + x1min)/2
dx1 = x1max - x10
x2min = -1
x2max = 5
x20 = (x2max + x2min)/2
dx2 = x2max - x20
x3min = -7
x3max = 2
x30 = (x3max + x3min)/2
dx3 = x3max - x30
l =  1.215
x1l = l * dx1 + x10
x2l = l * dx2 + x20
x3l = l * dx3 + x30
xAvmin = (x1min + x2min + x3min)/3
xAvmax = (x1max + x2max + x3max)/3
ymin = 200 + xAvmin
ymax = 200 + xAvmax
random.seed()

table_NormExperiment = [["N", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3", "x1^2", "x2^2", "x3^2"],
                        [1,    -1,   -1,   -1,     1,      1,      1,      -1,        1,      1,      1],
                        [2,    -1,   -1,    1,     1,     -1,     -1,       1,        1,      1,      1],
                        [3,    -1,    1,   -1,    -1,      1,     -1,       1,        1,      1,      1],
                        [4,    -1,    1,    1,    -1,     -1,      1,      -1,        1,      1,      1],
                        [5,     1,   -1,   -1,    -1,     -1,      1,       1,        1,      1,      1],
                        [6,     1,   -1,    1,    -1,      1,     -1,      -1,        1,      1,      1],
                        [7,     1,    1,   -1,     1,     -1,     -1,      -1,        1,      1,      1],
                        [8,     1,    1,    1,     1,      1,      1,       1,        1,      1,      1],
                        [9,    -l,    0,    0,     0,      0,      0,       0,      l*l,      0,      0],
                        [10,    l,    0,    0,     0,      0,      0,       0,      l*l,      0,      0],
                        [11,    0,   -l,    0,     0,      0,      0,       0,        0,    l*l,      0],
                        [12,    0,    l,    0,     0,      0,      0,       0,        0,    l*l,      0],
                        [13,    0,    0,   -l,     0,      0,      0,       0,        0,      0,    l*l],
                        [14,    0,    0,    l,     0,      0,      0,       0,        0,      0,    l*l],
                        [15,    0,    0,    0,     0,      0,      0,       0,        0,      0,      0]]

table_NaturExperiment = [["N", "x1",  "x2",  "x3",  "x1x2",       "x1x3",       "x2x3",        "x1x2x3",           "x1^2",        "x2^2",        "x3^2"],
                         [1,    x1min, x2min, x3min, x1min*x2min,  x1min*x3min,  x2min*x3min,  x1min*x2min*x3min,   x1min*x1min,   x2min*x2min,   x3min*x3min],
                         [2,    x1min, x2min, x3max, x1min*x2min,  x1min*x3max,  x2min*x3max,  x1min*x2min*x3max,   x1min*x1min,   x2min*x2min,   x3min*x3min],
                         [3,    x1min, x2max, x3min, x1min*x2max,  x1min*x3min,  x2max*x3min,  x1min*x2max*x3min,   x1min*x1min,   x2max*x2min,   x3min*x3min],
                         [4,    x1min, x2max, x3max, x1min*x2max,  x1min*x3max,  x2max*x3max,  x1min*x2max*x3max,   x1min*x1min,   x2max*x2min,   x3min*x3min],
                         [5,    x1max, x2min, x3min, x1max*x2min,  x1max*x3min,  x2min*x3min,  x1max*x2min*x3min,   x1max*x1max,   x2min*x2min,   x3min*x3min],
                         [6,    x1max, x2min, x3max, x1max*x2min,  x1max*x3max,  x2min*x3max,  x1max*x2min*x3max,   x1max*x1max,   x2min*x2min,   x3min*x3min],
                         [7,    x1max, x2max, x3min, x1max*x2max,  x1max*x3min,  x2max*x3min,  x1max*x2max*x3min,   x1max*x1max,   x2max*x2min,   x3min*x3min],
                         [8,    x1max, x2max, x3max, x1max*x2max,  x1max*x3max,  x2max*x3max,  x1max*x2max*x3max,   x1max*x1max,   x2max*x2min,   x3min*x3min],
                         [9,    -x1l,  x20,   x30,   -x1l*x20,     -x1l*x30,     x20*x30,      -x1l*x20*x30,        x1l * x1l,     x20*x20,       x30*x30],
                         [10,   x1l,   x20,   x30,    x1l*x20,     x1l*x30,      x20*x30,      x1l*x20*x30,         x1l * x1l,     x20*x20,       x30*x30],
                         [11,   x10,   -x2l,  x30,    x10*(-x2l),  x10*x30,      -x2l*x30,     x10*(-x2l)*x30,      x10*x10,       x2l * x2l,     x30*x30],
                         [12,   x10,   x2l,   x30,    x10*x2l,     x10*x30,      x2l*x30,      x10*x2l*x30,         x10*x10,       x2l * x2l,     x30*x30],
                         [13,   x10,   x20,   -x3l,   x10*x20,     x10*(-x3l),   x20*(-x3l),   x10*x20*(-x3l),      x10*x10,       x20*x20,       x3l*x3l],
                         [14,   x10,   x20,   x3l,    x10*x20,     x10*x3l,      x20*x3l,      x10*x20*x3l,         x10*x10,       x20*x20,       x3l*x3l],
                         [15,   x10,   x20,   x30,    x10*x20,     x10*x30,      x20*x30,      x10*x20*x30,         x10*x10,       x20*x20,       x30*x30]]

coef_eqQuadB = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

coef_eqNLinB = [[1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]]

coef_eqLinB = [[1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

coef_eqQuad = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

coef_eqNLin = [[1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]]

coef_eqLin = [[1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]


free_el = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
free_elB = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(11):
    for j in range(max(i, 1), 11):
        for n in range(1, 15):
            if i == 0:
                coef_eqQuad[i][j] += table_NaturExperiment[n][j]
                coef_eqQuadB[i][j] += table_NormExperiment[n][j]
            else:
                coef_eqQuad[i][j] += table_NaturExperiment[n][i] * table_NaturExperiment[n][j]
                coef_eqQuadB[i][j] += table_NormExperiment[n][i] * table_NormExperiment[n][j]
        coef_eqQuad[i][j] /= 15
        coef_eqQuad[j][i] = coef_eqQuad[i][j]
        coef_eqQuadB[i][j] /= 15
        coef_eqQuadB[j][i] = coef_eqQuadB[i][j]
        if i < 8 and j < 8:
            coef_eqNLin[i][j] = coef_eqQuad[i][j]
            coef_eqNLin[j][i] = coef_eqNLin[i][j]
            coef_eqNLinB[i][j] = coef_eqQuadB[i][j]
            coef_eqNLinB[j][i] = coef_eqNLinB[i][j]
            if i < 4 and j < 4:
                coef_eqLin[i][j] = coef_eqQuad[i][j]
                coef_eqLin[j][i] = coef_eqLin[i][j]
                coef_eqLinB[i][j] = coef_eqQuadB[i][j]
                coef_eqLinB[j][i] = coef_eqLinB[i][j]



b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

AvYs = []

DisYs = []

t_val = []
t_cr = 0.0
insign = []

F_val = 0.0
F_cr = 0.0

def print_table(table):
    if not "Yi_av" in table_NormExperiment[0]:
        table_NormExperiment[0].append("Yi_av")
        table_NormExperiment[0].append("S_Yi")
    print("\n", "-" * len(table_NormExperiment[0]) * 11, "\n")
    for i in range(16):
        print("|", end="")
        for j in range(len(table_NormExperiment[i])):
            if i > 0 and j > 10:
                print("{:.2f}".format(float(table_NormExperiment[i][j])), end="    |")
            elif i == 0:
                print(table_NormExperiment[i][j], " "*(9-len(str(table_NormExperiment[i][j]))), end="|")
            elif i < 9 or j == 0:
                print(table[i][j], " "*(9-len(str(table[i][j]))), end="|")
            else:
                print("{:+.4f}".format(float(table[i][j])), end="   |")
        if i > 0:
            print("{:.2f}".format(float(AvYs[i-1])), end="    |")
            print("{:.2f}".format(float(DisYs[i-1])), end="    |")
        print("\n", "-" * len(table_NormExperiment[0])*11, "\n")


def randomize(s, e):
    global AvYs
    AvYs = []
    global DisYs
    DisYs = []
    for i in range(16):
        for j in range(s, e):
            if i == 0:
                table_NormExperiment[i].append("Yi{}".format(j - 2))
            else:
                table_NormExperiment[i].append(random.uniform(ymin, ymax))
    for i in range(1, 16):
        sum_y = 0
        for j in range(11, m + 11):
            sum_y += table_NormExperiment[i][j]
        AvYs.append(sum_y/m)
    for i in range(1, 16):
        sum_y = 0
        for j in range(11, m+11):
            sum_y += pow(table_NormExperiment[i][j]- AvYs[i-1], 2)
        DisYs.append(sum_y/(m-1))


def cochran():
    global DisYs
    max_dispersion = max(DisYs)
    Gp = max_dispersion/sum(DisYs)
    fisher = table_fisher(p, 1, f3)
    Gt = fisher/(fisher+f2-1)
    return Gp < Gt


def table_fisher(prob, d, f3):
    x_vec = [i*0.001 for i in range(int(10/0.001))]
    for i in x_vec:
        if abs(f.cdf(i, N-d, f3)-prob) < 0.0001:
            return i


def coef(lin):
    global a
    global b
    b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    free_el[0] = sum(AvYs)/len(AvYs)
    free_elB[0] = sum(AvYs) / len(AvYs)
    for i in range(1, 11):
        for n in range(1, 16):
            free_el[i] += table_NaturExperiment[n][i]*AvYs[n-1]
            free_elB[i] += table_NormExperiment[n][i] * AvYs[n - 1]
        free_el[i] /= 15
        free_elB[i] /= 15
    if lin == "l":
        denominator = numpy.linalg.det(numpy.array(coef_eqLin))
        denominatorB = numpy.linalg.det(numpy.array(coef_eqLinB))
        for i in range(4):
            numerM = copy.deepcopy(coef_eqLin)
            numerMB = copy.deepcopy(coef_eqLinB)
            for j in range(4):
                numerM[j][i] = free_el[j]
                numerMB[j][i] = free_elB[j]
            numerator = numpy.linalg.det(numpy.array(numerM))
            numeratorB = numpy.linalg.det(numpy.array(numerMB))
            a[i] = numerator/denominator
            b[i] = numeratorB / denominatorB

    elif lin == "n":
        denominator = numpy.linalg.det(numpy.array(coef_eqNLin))
        denominatorB = numpy.linalg.det(numpy.array(coef_eqNLinB))
        for i in range(8):
            numerM = copy.deepcopy(coef_eqNLin)
            numerMB = copy.deepcopy(coef_eqNLinB)
            for j in range(8):
                numerM[i][j] = free_el[j]
                numerMB[i][j] = free_elB[j]
            numerator = numpy.linalg.det(numpy.array(numerM))
            numeratorB = numpy.linalg.det(numpy.array(numerMB))
            a[i] = numerator / denominator
            b[i] = numeratorB / denominatorB
    else:
        denominator = numpy.linalg.det(numpy.array(coef_eqQuad))
        denominatorB = numpy.linalg.det(numpy.array(coef_eqQuadB))
        for i in range(11):
            numerM = copy.deepcopy(coef_eqQuad)
            numerMB = copy.deepcopy(coef_eqQuadB)
            for j in range(8):
                numerM[i][j] = free_el[j]
                numerMB[i][j] = free_elB[j]
            numerator = numpy.linalg.det(numpy.array(numerM))
            numeratorB = numpy.linalg.det(numpy.array(numerMB))
            a[i] = numerator / denominator
            b[i] = numeratorB / denominatorB



def student(lin):
    global DisYs
    global AvYs
    global d
    global t_val
    global t_cr
    global insign
    global b
    AvDisYs = sum(DisYs)/len(DisYs)
    Sb = math.sqrt(AvDisYs/(N*m))
    t_val = []
    if lin  == "l":
        r = 4
    elif lin == "n":
        r = 8
    else:
        r = 11
    for x in range(r):
        t_val.append(math.fabs(b[x])/Sb)
    t_cr = 0
    x_vec = [i * 0.0001 for i in range(int(5 / 0.0001))]
    par = 0.5 + p / 0.1 * 0.05
    for i in x_vec:
        if abs(t.cdf(i, f3) - par) < 0.000005:
            t_cr = i
            break
    insign = []
    for i in range(len(t_val)):
        if t_val[i] <= t_cr:
            insign.append(i)
            d -= 1


def fisher():
    global DisYs
    global F_val
    global F_cr
    AvDisYs = sum(DisYs) / len(DisYs)
    Sad = 0
    for av in range(len(AvYs)):
        y_val = 0
        for i in range(11):
            if not i in insign and not a[i]==0:
                if i == 0:
                    y_val += b[0]
                else:
                    y_val += b[i]*table_NormExperiment[av+1][i]
        Sad += math.pow(y_val-AvYs[av], 2)
    Sad = Sad*m/(N-d)
    F_val = Sad/AvDisYs
    x_vec = [i * 0.001 for i in range(int(10 / 0.001))]
    F_cr = None
    for i in x_vec:
        if abs(f.cdf(i, N - d, f3) - p) < 0.0001:
            F_cr = i
            break
    if not F_cr:
        print("\nSomething went wrong.\nUnable to calculate critical value for Fisher's test")
    elif F_cr >= F_val:
        print("\nF = {}\t\t\tF_cr = {}\t\t\tF =< F_cr\nAccording to Fisher's F-test model is adequate to the original.".format(F_val, F_cr))
        return True
    else:
        print("\nF = {}\t\t\tF_cr = {}\t\t\tF > F_cr\nAccording to Fisher's F-test model is not adequate to the original.".format(F_val, F_cr))
        return False


def printRes(lin):
    print("\n\nNormalized Experiment:")
    print_table(table_NormExperiment)
    print("\nNaturalized Experiment:")
    print_table(table_NaturExperiment)
    print("\n\nNormalized equation:\ny = ", end="")
    for i in range(4):
        if i == 0:
            print("{:.2f} ".format(b[i]), end="")
        else:
            print("{:+.2f}*x{}".format(b[i], i), end="")
    if lin == "n":
        print("{:+.2f}*x1x2 {:+.2f}*x1x3 {:+.2f}*x2x3 {:+.2f}*x1x2x3".format(b[4], b[5], b[6], b[7]))
        print("\nCheck:")
        for i in range(1, 15):
            print("{:.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} = {:.2f}\ny = {:.2f}\n".format(b[0],
                                                                                 b[1] * table_NormExperiment[i][1],
                                                                                 b[2] * table_NormExperiment[i][2],
                                                                                 b[3] * table_NormExperiment[i][3],
                                                                                 b[4] * table_NormExperiment[i][4],
                                                                                 b[5] * table_NormExperiment[i][5],
                                                                                 b[6] * table_NormExperiment[i][6],
                                                                                 b[7] * table_NormExperiment[i][7],
                                                                                 b[0] +
                                                                                 b[1] * table_NormExperiment[i][1] +
                                                                                 b[2] * table_NormExperiment[i][2] +
                                                                                 b[3] * table_NormExperiment[i][3] +
                                                                                 b[4] * table_NormExperiment[i][4] +
                                                                                 b[5] * table_NormExperiment[i][5] +
                                                                                 b[6] * table_NormExperiment[i][6] +
                                                                                 b[7] * table_NormExperiment[i][7],
                                                                                 AvYs[i - 1]))
    elif lin == "l":
        print("\nCheck:")
        for i in range(1, 15):
            print("{:.2f} {:+.2f} {:+.2f} {:+.2f} = {:.2f}\ny = {:.2f}\n".format(b[0],
                                                                                 b[1] * table_NormExperiment[i][1],
                                                                                 b[2] * table_NormExperiment[i][2],
                                                                                 b[3] * table_NormExperiment[i][3],
                                                                                 b[0] +
                                                                                 b[1] * table_NormExperiment[i][1] +
                                                                                 b[2] * table_NormExperiment[i][2] +
                                                                                 b[3] * table_NormExperiment[i][3],
                                                                                 AvYs[i - 1]))
    else:
        print("{:+.2f}*x1x2 {:+.2f}*x1x3 {:+.2f}*x2x3 {:+.2f}*x1x2x3 {:+.2f}*x1^2 {:+.2f}*x2^2 {:+.2f}*x3^2".format(b[4], b[5], b[6], b[7], b[8], b[9], b[10]))
        print("\nCheck:")
        for i in range(1, 15):
            print("{:.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} = {:.2f}\ny = {:.2f}\n".format(b[0],
                                                                                                                 b[1] * table_NormExperiment[i][1],
                                                                                                                 b[2] * table_NormExperiment[i][2],
                                                                                                                 b[3] * table_NormExperiment[i][3],
                                                                                                                 b[4] * table_NormExperiment[i][4],
                                                                                                                 b[5] * table_NormExperiment[i][5],
                                                                                                                 b[6] * table_NormExperiment[i][6],
                                                                                                                 b[7] * table_NormExperiment[i][7],
                                                                                                                 b[8] * table_NormExperiment[i][8],
                                                                                                                 b[9] * table_NormExperiment[i][9],
                                                                                                                 b[10] * table_NormExperiment[i][10],
                                                                                                                 b[0] +
                                                                                                                 b[1] * table_NormExperiment[i][1] +
                                                                                                                 b[2] * table_NormExperiment[i][2] +
                                                                                                                 b[3] * table_NormExperiment[i][3] +
                                                                                                                 b[4] * table_NormExperiment[i][4] +
                                                                                                                 b[5] * table_NormExperiment[i][5] +
                                                                                                                 b[6] * table_NormExperiment[i][6] +
                                                                                                                 b[7] * table_NormExperiment[i][7] +
                                                                                                                 b[8] * table_NormExperiment[i][8] +
                                                                                                                 b[9] * table_NormExperiment[i][9] +
                                                                                                                 b[10] * table_NormExperiment[i][10],
                                                                                                                 AvYs[i - 1]))
    print("\n\nNaturalized equasion:\ny = ", end="")
    for i in range(4):
        if i == 0:
            print("{:.2f} ".format(a[i]), end="")
        else:
            print("{:+.2f}*x{}".format(a[i], i), end="")
    if lin == "n":
        print("{:+.2f}*x1x2 {:+.2f}*x1x3 {:+.2f}*x2x3 {:+.2f}*x1x2x3".format(a[4], a[5], a[6], a[7]))
        print("\nCheck:")
        for i in range(1, 15):
            print("{:.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} = {:.2f}\ny = {:.2f}\n".format(a[0],
                                                                                 a[1] * table_NaturExperiment[i][1],
                                                                                 a[2] * table_NaturExperiment[i][2],
                                                                                 a[3] * table_NaturExperiment[i][3],
                                                                                 a[4] * table_NaturExperiment[i][4],
                                                                                 a[5] * table_NaturExperiment[i][5],
                                                                                 a[6] * table_NaturExperiment[i][6],
                                                                                 a[7] * table_NaturExperiment[i][7],
                                                                                 a[0] +
                                                                                 a[1] * table_NaturExperiment[i][1] +
                                                                                 a[2] * table_NaturExperiment[i][2] +
                                                                                 a[3] * table_NaturExperiment[i][3] +
                                                                                 a[4] * table_NaturExperiment[i][4] +
                                                                                 a[5] * table_NaturExperiment[i][5] +
                                                                                 a[6] * table_NaturExperiment[i][6] +
                                                                                 a[7] * table_NaturExperiment[i][7],
                                                                                 AvYs[i - 1]))
    elif lin == "l":
        print("\nCheck:")
        for i in range(1, 15):
            print("{:.2f} {:+.2f} {:+.2f} {:+.2f} = {:.2f}\ny = {:.2f}\n".format(a[0],
                                                                                 a[1] * table_NaturExperiment[i][1],
                                                                                 a[2] * table_NaturExperiment[i][2],
                                                                                 a[3] * table_NaturExperiment[i][3],
                                                                                 a[0] +
                                                                                 a[1] * table_NaturExperiment[i][1] +
                                                                                 a[2] * table_NaturExperiment[i][2] +
                                                                                 a[3] * table_NaturExperiment[i][3],
                                                                                 AvYs[i - 1]))
    else:
        print(
            "{:+.2f}*x1x2 {:+.2f}*x1x3 {:+.2f}*x2x3 {:+.2f}*x1x2x3 {:+.2f}*x1^2 {:+.2f}*x2^2 {:+.2f}*x3^2".format(a[4], a[5], a[6], a[7], a[8], a[9], a[10]))
        print("\nCheck:")
        for i in range(1, 15):
            print("{:.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} {:+.2f} = {:.2f}\ny = {:.2f}\n".format(a[0],
                                                                                                                 a[1] * table_NaturExperiment[i][1],
                                                                                                                 a[2] * table_NaturExperiment[i][2],
                                                                                                                 a[3] * table_NaturExperiment[i][3],
                                                                                                                 a[4] * table_NaturExperiment[i][4],
                                                                                                                 a[5] * table_NaturExperiment[i][5],
                                                                                                                 a[6] * table_NaturExperiment[i][6],
                                                                                                                 a[7] * table_NaturExperiment[i][7],
                                                                                                                 a[8] * table_NaturExperiment[i][8],
                                                                                                                 a[9] * table_NaturExperiment[i][9],
                                                                                                                 a[10] * table_NaturExperiment[i][10],
                                                                                                                 a[0] +
                                                                                                                 a[1] * table_NaturExperiment[i][1] +
                                                                                                                 a[2] * table_NaturExperiment[i][2] +
                                                                                                                 a[3] * table_NaturExperiment[i][3] +
                                                                                                                 a[4] * table_NaturExperiment[i][4] +
                                                                                                                 a[5] * table_NaturExperiment[i][5] +
                                                                                                                 a[6] * table_NaturExperiment[i][6] +
                                                                                                                 a[7] * table_NaturExperiment[i][7] +
                                                                                                                 a[8] * table_NaturExperiment[i][8] +
                                                                                                                 a[9] * table_NaturExperiment[i][9] +
                                                                                                                 a[10] * table_NaturExperiment[i][10],
                                                                                                                 AvYs[i - 1]))
    print("According to Student's t-test these coefficients are insignificant:")
    for ind in insign:
        print("t = {}\t\t\tt_cr = {}\t\t\tt < t_cr\nb{} = {:.2f} and a{} = {:.2f}".format(t_val[ind], t_cr, ind, b[ind], ind,a[ind]))
    print("\nThen the equations change:\nNormalized:\ny = ", end="")
    for i in range(4):
        if not i in insign:
            if i == 0:
                print("{:.2f} ".format(b[i]), end="")
            else:
                print("{:+.2f}*x{}".format(b[i], i), end="")
    if not lin == "l":
        for i in range(4, 11):
            if not i in insign:
                print("{:+.2f}*{}".format(b[i], table_NormExperiment[0][i+1]), end="")
    print("\n\nNaturalized:\ny = ", end="")
    for i in range(4):
        if not i in insign:
            if i == 0:
                print("{:.2f} ".format(a[i]), end="")
            else:
                print("{:+.2f}*x{}".format(a[i], i), end="")
    if not lin == "l":
        for i in range(4, 11):
            if not i in insign:
                print("{:+.2f}*{}".format(a[i], table_NormExperiment[0][i+1]), end="")
    if not F_cr:
        print("\n\nSomething went wrong.\nUnable to calculate critical value for Fisher's test")
    elif F_cr >= F_val:
        print("\n\nF = {}\t\t\tF_cr = {}\t\t\tF =< F_cr\nAccording to Fisher's F-test model is adequate to the original.".format(F_val, F_cr))
    else:
        print("\n\nF = {}\t\t\tF_cr = {}\t\t\tF > F_cr\nAccording to Fisher's F-test model is not adequate to the original.".format(F_val, F_cr))


fish = False
m = 3
N = 15
d = 4
f1 = m - 1
f4 = N - d
startY = 11
endY = m + 11
print("\t\t\tLinear equation\nGenerating values...")
randomize(startY, endY)
eq = "l"
cochran_cond = cochran()
while not cochran_cond:
    m += 1
    startY = endY
    endY = m + 11
    f1 = m - 1
    randomize(startY, endY)
    cochran_cond = cochran()
print("Calculating coefficients...")
coef(eq)
student(eq)
f4 = N - d
fish = fisher()
if not fish:
    print("\t\t\tNon-linear equation\nGenerating values...")
    table_NormExperiment = [["N", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3", "x1^2", "x2^2", "x3^2"],
                            [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1],
                            [2, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1],
                            [3, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1],
                            [4, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1],
                            [5, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                            [6, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1],
                            [7, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
                            [8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [9, -l, 0, 0, 0, 0, 0, 0, l * l, 0, 0],
                            [10, l, 0, 0, 0, 0, 0, 0, l * l, 0, 0],
                            [11, 0, -l, 0, 0, 0, 0, 0, 0, l * l, 0],
                            [12, 0, l, 0, 0, 0, 0, 0, 0, l * l, 0],
                            [13, 0, 0, -l, 0, 0, 0, 0, 0, 0, l * l],
                            [14, 0, 0, l, 0, 0, 0, 0, 0, 0, l * l],
                            [15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    table_NaturExperiment = [["N", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3", "x1^2", "x2^2", "x3^2"],
                             [1, x1min, x2min, x3min, x1min * x2min, x1min * x3min, x2min * x3min,
                              x1min * x2min * x3min, x1min * x1min, x2min * x2min, x3min * x3min],
                             [2, x1min, x2min, x3max, x1min * x2min, x1min * x3max, x2min * x3max,
                              x1min * x2min * x3max, x1min * x1min, x2min * x2min, x3min * x3min],
                             [3, x1min, x2max, x3min, x1min * x2max, x1min * x3min, x2max * x3min,
                              x1min * x2max * x3min, x1min * x1min, x2max * x2min, x3min * x3min],
                             [4, x1min, x2max, x3max, x1min * x2max, x1min * x3max, x2max * x3max,
                              x1min * x2max * x3max, x1min * x1min, x2max * x2min, x3min * x3min],
                             [5, x1max, x2min, x3min, x1max * x2min, x1max * x3min, x2min * x3min,
                              x1max * x2min * x3min, x1max * x1max, x2min * x2min, x3min * x3min],
                             [6, x1max, x2min, x3max, x1max * x2min, x1max * x3max, x2min * x3max,
                              x1max * x2min * x3max, x1max * x1max, x2min * x2min, x3min * x3min],
                             [7, x1max, x2max, x3min, x1max * x2max, x1max * x3min, x2max * x3min,
                              x1max * x2max * x3min, x1max * x1max, x2max * x2min, x3min * x3min],
                             [8, x1max, x2max, x3max, x1max * x2max, x1max * x3max, x2max * x3max,
                              x1max * x2max * x3max, x1max * x1max, x2max * x2min, x3min * x3min],
                             [9, -x1l, x20, x30, -x1l * x20, -x1l * x30, x20 * x30, -x1l * x20 * x30, x1l * x1l,
                              x20 * x20, x30 * x30],
                             [10, x1l, x20, x30, x1l * x20, x1l * x30, x20 * x30, x1l * x20 * x30, x1l * x1l, x20 * x20,
                              x30 * x30],
                             [11, x10, -x2l, x30, x10 * (-x2l), x10 * x30, -x2l * x30, x10 * (-x2l) * x30, x10 * x10,
                              x2l * x2l, x30 * x30],
                             [12, x10, x2l, x30, x10 * x2l, x10 * x30, x2l * x30, x10 * x2l * x30, x10 * x10, x2l * x2l,
                              x30 * x30],
                             [13, x10, x20, -x3l, x10 * x20, x10 * (-x3l), x20 * (-x3l), x10 * x20 * (-x3l), x10 * x10,
                              x20 * x20, x3l * x3l],
                             [14, x10, x20, x3l, x10 * x20, x10 * x3l, x20 * x3l, x10 * x20 * x3l, x10 * x10, x20 * x20,
                              x3l * x3l],
                             [15, x10, x20, x30, x10 * x20, x10 * x30, x20 * x30, x10 * x20 * x30, x10 * x10, x20 * x20,
                              x30 * x30]]

    m = 3
    N = 15
    d = 8
    f1 = m - 1
    f4 = N - d
    startY = 3
    endY = m + 3
    eq = "n"
    randomize(startY, endY)
    cochran_cond = cochran()
    while not cochran_cond:
        m += 1
        startY = endY
        endY = m + 11
        f1 = m - 1
        randomize(startY, endY)
        cochran_cond = cochran()
    print("Calculating coefficients...")
    coef(eq)
    student(eq)
    f4 = N - d
    fish = fisher()
    if not fish:
        table_NormExperiment = [["N", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3", "x1^2", "x2^2", "x3^2"],
                                [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1],
                                [2, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1],
                                [3, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1],
                                [4, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1],
                                [5, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                                [6, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1],
                                [7, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
                                [8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [9, -l, 0, 0, 0, 0, 0, 0, l * l, 0, 0],
                                [10, l, 0, 0, 0, 0, 0, 0, l * l, 0, 0],
                                [11, 0, -l, 0, 0, 0, 0, 0, 0, l * l, 0],
                                [12, 0, l, 0, 0, 0, 0, 0, 0, l * l, 0],
                                [13, 0, 0, -l, 0, 0, 0, 0, 0, 0, l * l],
                                [14, 0, 0, l, 0, 0, 0, 0, 0, 0, l * l],
                                [15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        table_NaturExperiment = [["N", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3", "x1^2", "x2^2", "x3^2"],
                                 [1, x1min, x2min, x3min, x1min * x2min, x1min * x3min, x2min * x3min,
                                  x1min * x2min * x3min, x1min * x1min, x2min * x2min, x3min * x3min],
                                 [2, x1min, x2min, x3max, x1min * x2min, x1min * x3max, x2min * x3max,
                                  x1min * x2min * x3max, x1min * x1min, x2min * x2min, x3min * x3min],
                                 [3, x1min, x2max, x3min, x1min * x2max, x1min * x3min, x2max * x3min,
                                  x1min * x2max * x3min, x1min * x1min, x2max * x2min, x3min * x3min],
                                 [4, x1min, x2max, x3max, x1min * x2max, x1min * x3max, x2max * x3max,
                                  x1min * x2max * x3max, x1min * x1min, x2max * x2min, x3min * x3min],
                                 [5, x1max, x2min, x3min, x1max * x2min, x1max * x3min, x2min * x3min,
                                  x1max * x2min * x3min, x1max * x1max, x2min * x2min, x3min * x3min],
                                 [6, x1max, x2min, x3max, x1max * x2min, x1max * x3max, x2min * x3max,
                                  x1max * x2min * x3max, x1max * x1max, x2min * x2min, x3min * x3min],
                                 [7, x1max, x2max, x3min, x1max * x2max, x1max * x3min, x2max * x3min,
                                  x1max * x2max * x3min, x1max * x1max, x2max * x2min, x3min * x3min],
                                 [8, x1max, x2max, x3max, x1max * x2max, x1max * x3max, x2max * x3max,
                                  x1max * x2max * x3max, x1max * x1max, x2max * x2min, x3min * x3min],
                                 [9, -x1l, x20, x30, -x1l * x20, -x1l * x30, x20 * x30, -x1l * x20 * x30, x1l * x1l,
                                  x20 * x20, x30 * x30],
                                 [10, x1l, x20, x30, x1l * x20, x1l * x30, x20 * x30, x1l * x20 * x30, x1l * x1l,
                                  x20 * x20, x30 * x30],
                                 [11, x10, -x2l, x30, x10 * (-x2l), x10 * x30, -x2l * x30, x10 * (-x2l) * x30,
                                  x10 * x10, x2l * x2l, x30 * x30],
                                 [12, x10, x2l, x30, x10 * x2l, x10 * x30, x2l * x30, x10 * x2l * x30, x10 * x10,
                                  x2l * x2l, x30 * x30],
                                 [13, x10, x20, -x3l, x10 * x20, x10 * (-x3l), x20 * (-x3l), x10 * x20 * (-x3l),
                                  x10 * x10, x20 * x20, x3l * x3l],
                                 [14, x10, x20, x3l, x10 * x20, x10 * x3l, x20 * x3l, x10 * x20 * x3l, x10 * x10,
                                  x20 * x20, x3l * x3l],
                                 [15, x10, x20, x30, x10 * x20, x10 * x30, x20 * x30, x10 * x20 * x30, x10 * x10,
                                  x20 * x20, x30 * x30]]

        print("\t\t\tQuadratic equation\nGenerating values...")
        m = 3
        N = 15
        d = 11
        f1 = m - 1
        f4 = N - d
        startY = 3
        endY = m + 3
        eq = "q"
        randomize(startY, endY)
        cochran_cond = cochran()
        while not cochran_cond:
            m += 1
            startY = endY
            endY = m + 11
            f1 = m - 1
            randomize(startY, endY)
            cochran_cond = cochran()
        print("Calculating coefficients...")
        coef(eq)
        student(eq)
        f4 = N - d
        fish = fisher()
    printRes(eq)
else:
    printRes(eq)
