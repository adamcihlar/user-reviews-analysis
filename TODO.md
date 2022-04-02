1. **Clean the code and structure**
 	- [x] create main.py
	- [x] incorporate functionality of my Dataset class to the torch Dataset - eventually renamed my Dataset to RawDataset and kept them apart as the two classes will be used in different stages of pipeline
 	- [x] turn the spaghetti to functions/classes (especially in model.py)
2. **Improve the model** - 50-60 % on validation set is not the best
 	- [ ] try out a model pretrained for sentiment - number of classes?
 	- [ ] larger model (tiny-bert might be too small?)
	- [ ] small dataset, smaller ML algos - tree based, SVM, KNN, TF-IDF instead of semantic embeddings
	- [ ] class TF-IDF - vectorize the sentences separately based on their class - (dim reduction) - train MLP (or other ML model, but MLP could handle the different inputs "meaning" for every class) - predictions as follows: encode the sample to every TF-IDF class defined space, run the model for n-class times, combine the predictions - sum probs (or logits)/take max/...
	- [ ] Try this approach https://developers.google.com/machine-learning/guides/text-classification
	- [ ] convert reviews to positive x negative, then pretrained sentiment model
3. **Seperate WordClouds for classes** (/positive and negative reviews)
4. **Embedding analysis** - get embeddings from the classifier, project to lower dim space (PCA?), color based on Ratings, clustering, main topics for the created clusters
