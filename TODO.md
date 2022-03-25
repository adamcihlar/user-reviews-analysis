1. **Clean the code and structure**
 	- [ ] create main.py
 	- [ ] turn the spaghetti to functions/classes (especially in model.py)
2. **Improve the model** - 50-60 % on validation set is not the best
 	- [ ] try out a model pretrained for sentiment - number of classes?
 	- [ ] larger model (tiny-bert might be too small?)
	- [ ] small dataset, smaller ML algos - tree based, SVM, KNN, TF-IDF instead of semantic embeddings
	- [ ] convert reviews to positive x negative, then pretrained sentiment model
3. **Seperate WordClouds for classes** (/positive and negative reviews)
4. **Embedding analysis** - get embeddings from the classifier, project to lower dim space (PCA?), color based on Ratings, clustering, main topics for the created clusters
