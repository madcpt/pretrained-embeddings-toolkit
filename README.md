# Pretrained-Embedding-ToolKit 

Toolkit for processing pretrained-embeddings.

Currently support:
 - 'FastText'
 - 'Glove'
 - 'KazumaChar'
 - 'SL999'

## FastTextEmbedding

## GloveEmbedding

## KazumaCharEmbedding

## sl999-Embedding

These are 300 dimensional Paragram embeddings tuned on the SimLex999 dataet. They achieve
human-level performance on both SimLex999 and WS353 datasets.

New! Paragram-SL999 300 dimensional Paragram embeddings tuned on SimLex999 dataset. 1.7 GB download.

(https://drive.google.com/file/d/0B9w48e1rj-MOLVdZRzFfTlNsem8/view?usp=sharing)[paragram\_300\_sl999.txt] : embeddings, each line is an embedding for the token at the
                            beginning of the line

If you use our embeddingsfor your work please cite:

@article{wieting2015ppdb,
title={From Paraphrase Database to Compositional Paraphrase Model and Back},
author={John Wieting and Mohit Bansal and Kevin Gimpel and Karen Livescu and Dan Roth},
journal={Transactions of the ACL (TACL)},
year={2015}
}

More details on the construction of the embeddings can be found in the arxiv version:
http://arxiv.org/abs/1506.03487
