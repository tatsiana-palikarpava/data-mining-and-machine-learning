In this folder you see the file "code.py", where is located the code used to train my model.
Some of the blocks of this program are disabled, because they require additional libraries.
However, if you find it interesting to run these parts of code, here you are provided commands to install the libraries.
I am not hundred percent sure that it works the same in Linux, but in Windows I ran Anaconda Prompt (command line) and typed the following:
prince-factor-analysis (for PCA and FAMD):
	conda install -c conda-forge prince-factor-analysis
pydot (for tree visualization):
	conda install -c conda-forge pydot
rfpimp (for permutation importances): 
	conda install -c conda-forge rfpimp

Keep in mind also that some of the optional parts are rather time consuming, that is why they are disabled
(especially hyperparameters tuning, performance analysis and permutation importances).
