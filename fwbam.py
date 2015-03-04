__author__ = 'artiom'

import time
import pysam
import math

samfile = pysam.AlignmentFile("ex40.7.sorted.bam", "rb")

lst = []
for read in samfile.fetch('chr1'):
    lst.append(read)
reads_in_chr = len(lst)

lengths_chr_list = samfile.lengths
k = 100000

bin = [0] * int(math.ceil(lengths_chr_list[0] / k) + 1)
amount_of_bins = len(bin)

def count_reads_naive_loop(N):
    for i in range(N):
        for read in samfile.fetch('chr1', i, i + 1):
            bin[i / k] += 1

    read_count = 0
    for i in range(amount_of_bins):
        read_count += bin[i]
    print('amount of reads: {}'.format(read_count))

def count_reads_quick_loop():
    d = {}
    for i in range(amount_of_bins):
        d[i] = (i * k, (i + 1) * k - 1)  # look below in loop
    #print(d)

    am1, am2 = 0, 0
    for i in range(amount_of_bins):  # 0,1,2...(n-1)
        for read in samfile.fetch('chr1', i * k, (i + 1) * k - 1):  # amount of reads in each bin
            am1 += 1

    for read in samfile.fetch('chr1', 0, lengths_chr_list[0]):
        am2 += 1

    print('amount of reads: ({}, {})\n'.format(am1, am2))

def explanation_test():
    # Problem: Each read should star and end inside bin! if this condition doesn't perform the read will omit!

    am1, am2, am = 0, 0, 0
    for read in samfile.fetch('chr1', 10500000, 10600000):
        am1 += 1

    for read in samfile.fetch('chr1', 1060100, 10700000):
        am2 += 1

    for read in samfile.fetch('chr1', 10500000, 10699999):
        am += 1

    print('( {} , {} )\n'.format(am, am1+am2))

    # my thoughts consist of additional counting of the reads that lie on the edge
    # starting in the i-bin, and ending in (i+1)-bin
    # but what length to take?

def coverage_vector():
    count = 0
    for read in samfile.fetch('chr1', 0, lengths_chr_list[0]):
        bin[int(math.ceil(read.reference_start / k))] += 1

    for i in range(len(bin)):
        count += bin[i]
    print('\n            reads: {}\nsum reads in bins: {}\n'.format(reads_in_chr, count))
    print("coverage vector:\n{}\n".format(bin))


if __name__ == "__main__":

    start = time.time()

    print('statistics:\n    length: {}\n     reads: {}\n      bins: {}\n'
          .format(lengths_chr_list[0], reads_in_chr, amount_of_bins))
    # naive_loop(100000)
    #count_reads_quick_loop()
    #explanation_test()
    coverage_vector()


    end = time.time()
    print('time: {} sec'.format(round((end - start), 2)))

samfile.close()

# for pcolumn in samfile.pileup('chr1',10156,10157):
#     print(pcolumn)
# print(samfile.count('chr1',1000
#  print(samfile.lengths)0,20000))

# print([samfile.header['SQ'][i]['SN'] for i in range(len(samfile.header['SQ']))])

# def reg2bin(beg, end):
#     if (beg>>14 == end>>14) : return ((1<<15)-1)/7 + (beg>>14)
#     if (beg>>17 == end>>17) : return ((1<<12)-1)/7 + (beg>>17)
#     if (beg>>20 == end>>20) : return ((1<<9)-1)/7 + (beg>>20)
#     if (beg>>23 == end>>23) : return ((1<<6)-1)/7 + (beg>>23)
#     if (beg>>26 == end>>26) : return ((1<<3)-1)/7 + (beg>>26)
#     return 0
#

# for refseqlength in [samfile.header['SQ'][i]['LN'] for i in range(len(samfile.header['SQ']))]:
#     print(refseqlength)

#     binnumbers = reg2bin(0, refseqlength - 1)
#     print("reference sequence length: %d has %d bins" %
#           (refseqlength, binnumbers))

# for pileupcolumn in samfile.pileup('chr1',10155,10156):
#     print ("at position %s starts %s reads" %
#             (pileupcolumn.pos, pileupcolumn.n))
#     print(pileupcolumn)


#     for pileupread in pileupcolumn.pileups:
#         print(pileupread.query_position)
#         print ('\tbase in read %s = %s' %
#                 (pileupread.alignment.query_name,
#                  pileupread.alignment.query_sequence[pileupread.query_position]))
