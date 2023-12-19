# COMP550Project

This project is done by Sienna Hsu (260948832), Joshua Jiang (260928880), and William Huang (260972252).

There are several files:
- NMT_text_sim.ipynb: The code containing the RNN model and experiments in Jupyter Notebook.
- nmt_text_sim.py: The code containing the RNN model and experiments. 
- lexical_simplification.py: The code containing the baseline lexical simplification model.
- eval_func.py: The code containing all the evaluation metrics.
- normal_test.txt: A text file containing the original sentences of the 8384 test instances.
- simple_test.txt: A text file containing the ground truth simple sentences to the test instances.
- simple_lex.txt: A text file containing the simplified text by the baseline model on the test instances.
- simple_s2s.txt: A text file containing the simplified text by the RNN model on the test instances.
- dataset: A folder containing the dataset we used.

# Packages
- Spacy's pre-trained model: en_core_web_sm
- textstat
- nltk
- sklearn
- pyswd.lesk
- einops
- matplotlib
- tensorflow
