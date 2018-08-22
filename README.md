# odormatic

Python scripts to generate the findings in Gutierrez, Durandhar, Meyer, and Cecchi (2018), along with some of the accompanying figures.

Predictions are generated in four different scripts.  Two of the scripts are for the Direct models (DirSem, DirRat, DirMix in the text) and two of the scripts are for the Imputed models (ImpSem, ImpRat, ImpMix in the text).

DirectMolecules.py and ImputedMolecules.py show the effect of increasing the number of training molecules on the Direct and Imputed models, respectively; this is what is shown in Figure 3 of the paper.

DirectDescriptors.py and ImputedDescriptors.py show the effect of increasing the number of descriptors used during training on the Direct and Imputed models, respectively; this is what is shown in Figure 4a of the paper.


List of files:
*mol_utils.py:* script with supporting utilities used by other scripts.

*Direct Molecules.py:* DirectMolecules.py shows the effect of increasing the number of training molecules on the Direct models.  Generates findings of Figure 3a of paper.

*Imputed Molecules.py:* ImputedMolecules.py shows the effect of increasing the number of training molecules on the Imputed models.  Generates findings of Figure 3b of paper.

*DirectDescriptors.py:* shows the effect of increasing the number of descriptors used during training on the Direct Semantic (DirSem) model.   Generates findings of Figure 4a of paper.

*ImputedDescriptors.py:* shows the effect of increasing the number of descriptors used during training on the Imputed Semantic (ImpSem) model.   Generates findings of Figure 4a of paper.
