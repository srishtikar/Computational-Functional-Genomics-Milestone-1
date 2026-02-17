For CFG First Milestone.py

# Requirements: 
  Python 3.11
  Libraries: numpy, pandas, matplotlib, pyfaidx

# File structure:
  CFG First Milestone.py - the primary script to run. 
  Input m (order of markov model), k (number of folds for the cross validation), the
  transcription factor (REST, EP300 or CTCF). 

  Data files: Download hg38.fa as a zipped file from https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/
  The chromosome data for all chromosomes (excluding 3,10, 17 and the sex chromosomes) are present
  as a zipped file titled chr_200bins.zip. These are to be unzipped and then loaded along with the 
  .fasta file into the Python workspace

# How to Run:
  Input the m (order of the markov model), k (number of folds for the cross validation), the
  transcription factor desired (REST, EP300 or CTCF) and also the chromosome number in the
  format chrX (say, chr4 or chr11). Ensure the previously mentioned data files are already
  loaded into the environment so the program can access it.


For simplerVersion.py

# Requirements
  Python 3.11
  Libraries: numpy, pyfaidx

# File structure
  simplerVersion.py- is the primary script to run. Along with this, load your desired fasta 
  file into the environment.

# How to Run:
  The first few lines of the actual code asks you to input the name of the file into the 
  variable FASTA_FILE. (Note: The fasta file must be loaded into the environment for access
  to the file. Moreover, the inputted name in FASTA_FILE must be properly spelled). Other
  than this, you will be prompted to input an m once you run the program. Thereafter, the
  output should be the log-odds probabilities of each region.
  
