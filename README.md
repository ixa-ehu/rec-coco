# REC-COCO DATASET FOR INFERRING SPATIAL RELATIONS FROM TEXTUAL DESCRIPTIONS
# Requirements

Python 3
Keras 2.0.9 (tested with tensorflow backend)
sklearn 0.19.1 (for evaluation)
h5py (if we want to store model weights)


Notice that we do not require actual images for our setting but only coordinates and bounding boxes.

Download the glove embeddings, and saved them on the ./embeddings folder:

https://drive.google.com/file/d/1CLNCtIh8CXi7abctH7UYwb6WqNdZ6pVE/view?usp=sharing

The REF-COCO dataset is saved in the ./training_data folder.


# Train and evaluate the model

Go to the folder of the model you want to train, run on the terminal:

python learn_and_eval.py

You can change the type of triplets, concept or textual. See --help for details.

Results are automatically stored in the ./results folder.
