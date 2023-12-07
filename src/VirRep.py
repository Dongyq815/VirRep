import torch
from seqcls import SeqCls
from predict import predict
from extract_hit import extract_seq, extract_seq_prov
import argparse
import sys
import os


def get_opts(args):
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='A hybrid language representation learning framework for identifying viruses from human gut metagenomes.')
    
    parser.add_argument('-i', '--input-file',
                        required=True,
                        type=str,
                        help='Input file in fasta format.')
    
    parser.add_argument('-o', '--output-dir',
                        required=True,
                        type=str,
                        help='Output directory for saving the prediction.')
    
    parser.add_argument('-m', '--model',
                        required=False,
                        type=str,
                        default='model/VirRep.pth',
                        help='Model path.')
    
    parser.add_argument('--conservative',
                        required=False,
                        default=False,
                        action='store_true',
                        help=('Apply conservative settings. This is equivalent to the following criteria: '
                              'baseline 0.8; min_score 1000-8000:0.9,8001-10000:0.85,10001-Inf:0.8.'))
    
    parser.add_argument('-l', '--minlen',
                        required=False,
                        type=int,
                        default=1000,
                        help=('Minimun length (should be >= 1000) of a sequence to predict. '
                              'Sequences shorter than this length will be ignored.'))
    
    parser.add_argument('--provirus-off',
                        required=False,
                        action='store_true',
                        default=False,
                        help='Skip the iterative segment extension procedure')
    
    parser.add_argument('-b', '--baseline',
                        required=False,
                        type=float,
                        default=0.5,
                        help='The baseline score for starting segment extension')
    
    parser.add_argument('--max-gap',
                        required=False,
                        type=int,
                        default=5000,
                        help=('Maximum distance for two adjacent extended regions to be merged. Two regions are '
                              "merged if the distance between them meet this condition or the 'max-frac' condition. "
                              "A value < 1000 means to skip the extension procedure."))
    
    parser.add_argument('--max-frac',
                        required=False,
                        type=float,
                        default=0.1,
                        help=('Maximum ratio of the distance between the two regions to be merged '
                              'to the sum of the lengths of the two regions. Two regions are merged if '
                              "the distance between them meet this condition or the 'max-gap' condition. "
                              "A value <= 0 means to disable this option."))
    
    parser.add_argument('--provirus-minlen',
                        required=False,
                        type=int,
                        default=5000,
                        help=('Minimun length of a region within the prokaryotic genome to be considered as a provirus. '
                              'A value <= 1000 means all significant regions will be considered as proviruses. '
                              'Smaller values will result in more viral sequences in the expense of higher false positives.'))
    
    parser.add_argument('--provirus-minfrac',
                        required=False,
                        type=float,
                        default=1,
                        help=('Minimum ratio of the length of the region to be considered as a provirus to the length of '
                              'the whole sequence. A value <= 0 means all significant regions will be considered as proviruses. '
                              'Smaller values will result in more viral sequences in the expense of higher false positives.'))
    
    parser.add_argument('-c', '--min-score',
                        required=False,
                        type=str,
                        default='1000-Inf:0.5',
                        help=('Minimum score of a sequence to be finally retained as a viral candidate. '
                              'Users can set different cutoffs for different sequence lengths '
                              '(e.g., 1000-5000:0.95,5001-10000:0.9,10001-Inf:0.8). '
                              'Length range must start from 1000 and end with Inf without any gap. '
                              'Different cutoffs are seperated by comma in desending order '
                              'with correspoding length intervals in asending order.'))
    
    parser.add_argument('-k', '--minlen-keep',
                        required=False,
                        type=int,
                        default=1000,
                        help='Minimum length of a viral hit to be finally kept.')
    
    parser.add_argument('-n', '--batch-size',
                        required=False,
                        type=int,
                        default=256,
                        help=('How many 1 kb-long segments to score at one time. '
                              'Larger batch size may reducce the running time, but will require more GPU memory.'))
    
    parser.add_argument('-w', '--num-workers',
                        required=False,
                        type=int,
                        default=0,
                        help='How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
    
    parser.add_argument('--cpu',
                        required=False,
                        default=False,
                        action='store_true',
                        help='Run VirRep on CPU. By default, VirRep will be run on GPU.')
    
    parser.add_argument('--cpu-threads',
                        required=False,
                        type=int,
                        default=0,
                        help=('Number of threads used to run VirRep on CPU. '
                              '0 means to use all available threads.'))
    
    parser.add_argument('--gpu-device',
                        required=False,
                        type=int,
                        default=0,
                        help='Device number of the GPU to run VirRep.')
    
    if not args:
        parser.print_help(sys.stderr)
        sys.exit()
    
    return parser.parse_args(args)


def parse_minscore(score_text):
    
    score_text_split = score_text.split(',')
    score_dict = {}
    
    for cutoff in score_text_split:
        
        cutoff_split = cutoff.split(':')
        score_dict[cutoff_split[0]] = float(cutoff_split[1])
        
    return score_dict


def check_cutoff_dict(cutoff_dict):
    
    lower_list, upper_list, score_list = [], [], []
    
    for len_interval, score in cutoff_dict.items():
        lower_list.append(int(len_interval.split('-')[0]))
        upper = len_interval.split('-')[1]
        
        if upper == 'Inf':
            upper_list.append(upper)
        else:
            upper_list.append(int(upper))
            
        score_list.append(score)
    
    if (lower_list[0] != 1000) or (upper_list[-1] != 'Inf'):
        raise ValueError('Error in min-score: length range must start from 1000 and end with Inf!')
    
    for i in range(len(lower_list)-1):
        if lower_list[i+1] <= lower_list[i]:
            raise ValueError('Error in min-score: length intervals should be in asending order!')
        
        if score_list[i+1] >= score_list[i]:
            raise ValueError('Error in min-score: cutoffs should be in desending order!')
            
        if upper_list[i] < lower_list[i]:
            raise ValueError('Error in min-score: Invalid length interval for %s-%s'%(lower_list[i], upper_list[i]))
            
        if upper_list[i]+1 != lower_list[i+1]:
            raise ValueError('Error in min-score: Two consecutive length intervals should have no gaps or overlaps!')
            

def check_opts(options):
    
    if options.conservative:
        if options.baseline == 0.5:
            options.baseline = 0.8
            
        if options.min_score == '1000-Inf:0.5':
            options.min_score = '1000-8000:0.9,8001-10000:0.85,10001-Inf:0.8'
            
        if (options.baseline != 0.8) or (options.min_score != '1000-8000:0.9,8001-10000:0.85,10001-Inf:0.8'):
            raise ValueError("Conflict settings for option '--conservative' with option '--baseline' or '--min-score'")
    
    if options.minlen < 1000:
        raise ValueError('Error in minlen: minimum sequence length must be >= 1000 bp!')
    
    if options.baseline < 0 or options.baseline > 1:
        raise ValueError('Error in baseline: the value should be in [0, 1].')
        
    if options.cpu_threads == 0 or options.cpu_threads > os.cpu_count():
        options.cpu_threads = os.cpu_count()
        
    cutoff_dict = parse_minscore(options.min_score)
    check_cutoff_dict(cutoff_dict)
    options.min_score = cutoff_dict
    
    return options


if __name__ == '__main__':
    
    args = sys.argv[1:]
    options = get_opts(args)
    options = check_opts(options)
    model = torch.load(options.model, map_location=torch.device('cpu'))
    
    pred_df = predict(model,
                      options.input_file,
                      options.minlen,
                      options.baseline,
                      options.batch_size,
                      options.num_workers,
                      options.cpu,
                      options.cpu_threads,
                      options.gpu_device,
                      options.provirus_off,
                      options.max_gap,
                      options.max_frac,
                      options.provirus_minlen,
                      options.provirus_minfrac,
                      options.min_score)
   
    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)
        
    prefix = '_'.join(options.input_file.split('/')[-1].split('.')[0:-1])
    outpath_df = os.path.join(options.output_dir, prefix+'_score.tsv')
    pred_df.to_csv(outpath_df, index=False, sep='\t')
    
    outpath_fa = os.path.join(options.output_dir, prefix+'_viruses.fna')
    
    if options.provirus_off:
        extract_seq(pred_df, options.input_file, options.min_score,
                    options.minlen_keep, outpath_fa)
    else:
        extract_seq_prov(pred_df, options.input_file,
                         options.minlen_keep, outpath_fa)