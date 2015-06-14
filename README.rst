faireanalysis
========

``faireanalisys`` is a tool for faire-seq data **analyzing** of **some** biological samples.

The algorithm is based on learning of Hidden Markov Model and is used to **comparing** of samples.

Getting the latest code
=======================

To get the latest code using git, simply type::

        $ git clone git://github.com/baton2809/EM.git

Installing
==========

To install ``faireanalysis`` run::

        $ python setup.py install

in the source code directory.

Preparing
=========

Make sure you have installed the ``samtools``, see `samtools
<https://github.com/samtools/samtools>`_.

Move you ``bam`` files into source code directory and do sorting::

                $ samtools sort XXX.bam XXX.sorted

and then make index::

                $ samtools index XXX.sorted.bam

for each file.

Running
=======
To run the tool without arguments do::

                $ python -m fairy
        
Use ``--help`` to learn more::
        
                $ python -m fairy --help

Format to run for example::

                $ python -m fairy --gr1=ENCFF000TJP,ENCFF000TJR --gr2=ENCFF000TJJ,ENCFF000TJK --chr=chr2,chr3

Uninstall
=========
To uninstall call::

                $ chmod u+x cleanup
                $ ./cleanup

