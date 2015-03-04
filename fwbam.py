__author__ = 'artiom'

import time
import pysam
import math
import numpy as np

samfile = pysam.AlignmentFile("ex40.7.sorted.bam", "rb")
lengths_chr_list = samfile.lengths
k = 100000

def coverage_vector(name, order):
    bin = np.zeros((int(math.ceil(lengths_chr_list[order] / k) + 1)), dtype=int)
    count = 0
    for read in samfile.fetch(name):
        bin[int(math.ceil(read.reference_start / k))] += 1
    for i in range(len(bin)):
        count += bin[i]
    print("reference name: %s\n" % name)
    print('was reads: {}\ngot reads: {}\n'.format(samfile.count(samfile.references[order]), count))
    print("coverage vector:\n{}\n".format(bin))

if __name__ == "__main__":
    start = time.time()
    # print('statistics:\n    length: {}\n     reads: {}\n      bins: {}\n'
    #       .format(lengths_chr_list[0], reads_in_chr, amount_of_bins))
    order = 0
    for name_reference in samfile.references:
        coverage_vector(name_reference, order)
        order += 1
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
