# VirRep: A hybrid language representation learning framework for identifying viruses from human gut metagenomes
VirRep is a fully automated tool for identifying viruses from human gut metagenomes, 
which comnbines a context-aware encoder and an evolution-aware encoder to leverage the 
strength of both DNA language rules and sequence homology to improve classification performance.

## Requirements and Installation
Make sure you have the dependencies below installed and accessible in your $PATH.

### Prepare dependencies

- python >= 3.8
- biopython
- numpy
- pandas
- pytorch >= 2.1
- tqdm
- scipy
- scikit-learn
- packaging

1. Create a virtual environment
```
conda create -n vr python
conda activate vr
```

2. Install python modules: biopython, numpy, pandas, tqdm, scipy, scikit-learn, packaging
```
conda install -c bioconda biopython numpy, pandas, tqdm, scipy, scikit-learn, packaging
```
or

```
pip install biopython numpy, pandas, tqdm, scipy, scikit-learn, packaging
```

3. Install pytorch following the instructions in [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). 
For example, if your machine has NVIDIA GPU and supprots CUDA 11.8, you can run the following command:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
### Install VirRep via git

```
git clone https://github.com/Dongyq815/VirRep.git
```
___

## Usage
The input of VirRep is a `fasta` file containing the sequences to predict, and the output consists of a `.tsv` file recording 
the predicted score and a `fasta` file containing the predicted viral sequences. The higher score indicate higher 
likelihood of being a viral sequence.  

### Quick start
***
1. Run VirRep on a test dataset with GPU acceleration:
```
python VirRep.py -i test/test.fasta -o vr_out -w 2 --use-amp --min-score 1000-5000:0.9,5001-10000:0.8,10001-Inf:0.7
ls vr_out
```

In the output directory (vr_out), there are two files:

- `test_viruses.fna`: identified viral sequences
- `test_score.tsv`: table with score of each viral sequence and a few more features

**Note**

Note that suffix `||full` or `||partial` is appended to the original sequence identifier, 
indicating whether the viral sequence is extracted from a larger scaffold.

2. Run VirRep skipping over the iterative segment extension mechanism:
```
python VirRep.py -i test/test.fasta -o vr_out --use-amp --provirus-off -w 2
```

This is useful when comparing VirRep with other methods on a benchmark dataset, 
as all input sequences will report a score in the output `.tsv` file.

3. Run VirRep on bulk metagenomes:
```
python VirRep.py -i test/toy.fasta -o vr_out --use-amp --conservative -w 2
```
In this mode, VirRep will use conservative settings to reduce false positives and only output high-confidence 
viral sequences.

### Complete options
You can run `python VirRep.py -h` to see all options.

```
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input-file INPUT_FILE
                        Input file in fasta format. (default: None)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory for saving the prediction. (default: None)
  --label LABEL         Prefix to append to the output filename. (default: )
  -m MODEL, --model MODEL
                        Model path. (default: model/VirRep.pth)
  --use-amp             Use automatic mixed precision to accelerate computational effeciency. (default: False)
  --conservative        Apply conservative settings. This is equivalent to the following criteria: 
                        baseline 0.8; min_score 1000-8000:0.9,8001-10000:0.85,10001-Inf:0.8. (default: False)
  -l MINLEN, --minlen MINLEN
                        Minimun length (should be >= 1000) of a sequence to predict. 
                        Sequences shorter than this length will be ignored. (default:1000)
  --provirus-off        Skip the iterative segment extension procedure (default: False)
  -b BASELINE, --baseline BASELINE
                        The baseline score for starting segment extension (default: 0.5)
  --max-gap MAX_GAP     Maximum distance for two adjacent extended regions to be merged. 
                        Two regions are merged if the distance between them meet this 
                        condition or the 'max-frac' condition. A value < 1000 means to 
                        skip the extension procedure. (default: 5000)
  --max-frac MAX_FRAC   Maximum ratio of the distance between the two regions to be merged to the sum of 
                        the lengths of the two regions. Two regions are merged if the distance between 
                        them meet this condition or the 'max-gap' condition. A value <= 0 means to 
                        disable this option. (default: 0.1)
  --provirus-minlen PROVIRUS_MINLEN
                        Minimun length of a region within the prokaryotic genome to be considered as a provirus. 
                        A value <= 1000 means all significant regions will be considered as proviruses. Smaller 
                        values will result in more viral sequences in the expense of higher false positives. 
                        (default: 5000)
  --provirus-minfrac PROVIRUS_MINFRAC
                        Minimum ratio of the length of the region to be considered as a provirus to the length 
                        of the whole sequence. A value <= 0 means all significant regions will be considered as 
                        proviruses. Smaller values will result in more viral sequences in the expense of higher 
                        false positives. (default: 1)
  -c MIN_SCORE, --min-score MIN_SCORE
                        Minimum score of a sequence to be finally retained as a viral candidate. 
                        Users can set different cutoffs for different sequence lengths 
                        (e.g., 1000-5000:0.95,5001-10000:0.9,10001-Inf:0.8). Length range must 
                        start from 1000 and end with Inf without any gap. Different cutoffs are 
                        seperated by comma in desending order with correspoding length intervals 
                        in asending order. (default: 1000-Inf:0.5)
  -k MINLEN_KEEP, --minlen-keep MINLEN_KEEP
                        Minimum length of a viral hit to be finally kept. (default: 1500)
  -n BATCH_SIZE, --batch-size BATCH_SIZE
                        How many 1 kb-long segments to score at one time. Larger batch size may reducce the 
                        running time, but will require more GPU memory. (default: 256)
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        How many subprocesses to use for data loading. 
                        0 means that the data will be loaded in the main process. (default: 0)
  --cpu                 Run VirRep on CPU. By default, VirRep will be run on GPU. (default: False)
  --cpu-threads CPU_THREADS
                        Number of threads used to run VirRep on CPU. 
                        0 means to use all available threads. (default: 0)
  --gpu-device GPU_DEVICE
                        Device number of the GPU to run VirRep. (default: 0)              
```
