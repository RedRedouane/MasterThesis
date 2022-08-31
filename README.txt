This is the GitHub repository of my Master Thesis project.



To reproduce the experiments described in the thesis, a couple of setup steps have to be taken:

1) All experiments are done within the Fairseq package, so first setup that framework (https://github.com/facebookresearch/fairseq).
    - Replace some files in the Fairseq package for the files present in /Code (keep original name).
      - fairseq_cli_train_original/recycle/summ_only_dec.py replaces fairseq/fairseq_cli/train.py
      - enc_dec_new.py replaces fairseq/fairseq/models/roberta/enc_dec.py
      - utils_new.py replaces fairseq/fairseq/dataclass/utils.py
2) Download the original models and save them in the Models directory.
    - RoBERTa (https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md)
    - RobBERT (https://github.com/iPieter/RobBERT)
    - BART (https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md)
3) Download the original datasets and save them in the Datasets directory.
    - OSCAR (https://oscar-corpus.com/)
    - Europarl (https://www.statmt.org/europarl/)
    - Wikilingua (https://github.com/esdurmus/Wikilingua)
4) You're now ready to run the job in the Job_files directory!
    - During this project the LISA cluster was used, for which job files were written. If you're running this manually, look inside the job file for the correct fairseq command.
