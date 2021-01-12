# Exoplanet Classification

Shaunak Pal Abhinav Suresh Ennazhiyil Rajat Prakash

shaunak18098@iiitd.ac.in
abhinav18003@iiitd.ac.in
rajat18078@iiitd.ac.in

![image](https://github.com/blackhat-7/Exoplanet-Classification/blob/main/Plots/PlantHabitability_Vs_PlanetESI.png?raw=true)


## Summary
&nbsp;&nbsp;&nbsp;&nbsp; Existing work on characterizing exoplanets are based on assigning habitability scores to each planet which allows for a quantitative comparison with Earth. Over the past two decades, discoveries of exoplanets have poured in by the hundreds and the rate at which exoplanets are being discovered is increasing the actual number of planets exceeding the number of stars in our galaxy by orders of magnitude. 
&nbsp The  research  is  based  on  classifying  exoplanets  as  Habitable  and  Non-Habitable. The  research  uses  the  datasetprovided by NASA’s Exoplanet Archives which is from the TESS satellite.  We tried various preprocessing techniques and classifiers along with it such as KNN (K-Nearest Neigh-bors),  Hard-Boundary, SVM (Support  Vector  Machines) and,  Tree  based  classifiers  like  Random  Forests  and  en-semble classifiers like XGBoost to thoroughly analyze whatworks best for this domain.

&nbsp;&nbsp;&nbsp;&nbsp; Exoplanet Classification is a problem which will always remain a highly imbalanced classification problem with majority of the planets being discovered being uninhabitable. Hence dealing with Imbalanced classes like using Synthetic Minority Oversampling, Stratified splits etc is a must to get better predictions, and scaling is necessary since certain features can have astronomic values while some can have microscopic values. 
&nbsp;&nbsp;&nbsp;&nbsp; Almost  all  the  models  which  were  tested  have  90%+  accuracy with Earth Similarity Index as the most important feature. After doing Grid search over a lot of hyper param-eters for a lot of models we concluded that support vectormachines perform the best with f1 scores close to 98.9%, whereas tree based algorithms did not fare that well with f1 scores just below 90% which is not that bad and we made sure that at least all habitable planets get classified rightly. This shows that this problem statement can easily be solved by machine learning methods and can give astronomers ahuge insight into how to classify the habitable exoplanets comfortably.

&nbsp;&nbsp;&nbsp;&nbsp; The models and plots are stored in their folders respectively and the jupyter file is attached as well. Our project report is also attached that consists of all the work done.
