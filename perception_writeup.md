## Project: Perception Pick & Place

---

[//]: # (Image References)

[image1]: ./pr2_robot/scripts/raw_confusion.png
[image2]: ./pr2_robot/scripts/normalized_confusion.png
[image3]: ./pr2_robot/scripts/labels_1.png
[image4]: ./pr2_robot/scripts/labels_2.png
[image5]: ./pr2_robot/scripts/labels_3.png

### Exercise 1, 2 and 3 pipeline implemented
---

#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

First off, I tuned the parameters of the statistical outlier filter for more aggressive filtering of noisy vision data. I did this before beginning work on the perception pipeline in order to begin the process with the cleanest data possible. I arrived at a threshold value of 0.1 after experimentation with different values and inspection of the results in RViz.

Next, to improve overall performance, I used a smaller leaf size for my Voxel downsampling than that specified by the project outline (.0035 vs the specified .01), resulting in less downsampling and therefore denser clusters. Also, I added a second passthrough filter along the X-axis of the input cloud in order to remove the edges of the bins from the incoming point cloud and avoid including unclassifiable clusters in subsequent steps.

RANSAC segmentation was performed as specified and worked well.

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

Clustering proceeded as specified and worked well. Due to my smaller Voxel leaf size, I had to increase the maximum cluster size significantly in order to accommodate larger objects. After some tuning, though, I was able to obtain consistently good clusters for all objects. The rest of the sample code worked as expected.

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

For convenience, I wound up integrating much of the Python code form the `sensor_stick` project into the `pr2_robot` project. To accomplish this, I needed to add the message and service definitions from the `sensor_stick` and include them in `pr2_robot/CMakeLists.txt`. After some googling, this was very straightforward.

The key observations for computing the feature histograms are that the normal vectors should be in the range `[-1, 1]` and that a higher bin resolution makes more detailed results possible. After incorporating these into `sensor_stick/features.py`, I was able to get good and accurate results in training the SVM exactly as specified in the project outline.

At this point, I proceeded to incorporate the trained classifier into my perception pipeline. This process was straightforward and described well in the project outline. The only difficulty came in a naming error either in sample code or my transcription of same wherein I was attempting to use my trained classifier on white point cloud data from the clustering portion. This can be though of as an orthographic projection of the point cloud data into a lower dimensional space than that upon which the classifier was trained and so produced poor results. After chasing down and correcting the error, the classifier worked more closely to what was expected, but results were still not perfect. In test world 3, the `glue` object would always be classified as `soap`.

To improve the accuracy of my classifier, I first tried using the `rbf` kernel with various values of `C` and `gamma`. This could produce significant gains in cross validation accuracy, but those gains appeared to be due to overfitting as the model performed extremely poorly once integrated into my perception pipeline. For example, it would classify six of eight available objects as books while correctly classifying the other two.

After concluding that no further gains in accuracy could be achieved by simply tuning the SVM parameters, I decided to try applying other classifiers from the `sklearn` package. First I tried `AdaBoost`, which produced no better than 25% accuracy in cross validation. Next I tried a `DecisionTree`, which performed better than the SVM in cross validation but didn't correctly classify the glue in test world 3. Finally, I tried a `RandomForest`, an ensemble method based on the `Decision Tree`, and bingo! The `RandomForest` achieved 93% accuracy in cross validation and also correctly classified all objects in all test worlds. Although this choice of model technically violates the project specification, it should be plain to see from the associated confusion matrices that it was the best choice in this particular case, producing 90% accuracy or better for all objects in cross validation.


![alt text][image1]
![alt text][image2]


What follows are images of all objects correctly classified in the three test worlds.

![alt text][image3]
![alt text][image4]
![alt text][image5]

### Pick and Place Setup
---

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

This final step of calculating centroids and generating `PickPlace` requests proceeded without any difficulty. 
You will find the `output_*.yaml` files with associated `PickPlace` requests in the `pr2_robot/scrips` directory of this repository.

For the final integration of cluster data and object recognition into the `pr2_mover` routine, I used a dictionary to hold the calculated centroids of each cluster, indexed by label. This allowed me to look up the appropriate centroid while iterating over the list of objects from the incoming pick list.



