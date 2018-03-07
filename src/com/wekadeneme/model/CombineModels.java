package com.wekadeneme.model;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class CombineModels {

	public static void main(String[] args) throws Exception {
				// Load Data
				DataSource dataSource = new DataSource("./sources/datasets/weather.nominal.arff");
				Instances data = dataSource.getDataSet();
				
				//set class index
				if (data.classIndex() == -1) {
					data.setClassIndex(data.numAttributes() - 1);
				}
				
				/* Boosting a weak classifier using the Adaboost M1 method
				 * for boosting a nominal class classifier
				 * Tackles only nominal class problems
				 * Improves performance
				 * Sometimes overfits.
				 */
				//AdaBoost .. 
				AdaBoostM1 m1 = new AdaBoostM1();
				m1.setClassifier(new DecisionStump());//needs one base-classifier
				m1.setNumIterations(20);
				m1.buildClassifier(data);
				
				/* Bagging a classifier to reduce variance.
				 * Can do classification and regression (depending on the base model)
				 */
				//Bagging .. 
				Bagging bagger = new Bagging();
				bagger.setClassifier(new RandomTree());//needs one base-model
				bagger.setNumIterations(25);
				bagger.buildClassifier(data);		
				
				/*
				 * The Stacking method combines several models
				 * Can do classification or regression. 
				 */
				//Stacking ..
				Stacking stacker = new Stacking();
				stacker.setMetaClassifier(new J48());//needs one meta-model
				Classifier[] classifiers = {				
						new J48(),
						new NaiveBayes(),
						new RandomForest()
				};
				stacker.setClassifiers(classifiers);//needs one or more models
				stacker.buildClassifier(data);
				
				/*
				 * Class for combining classifiers.
				 * Different combinations of probability estimates for classification are available. 
				 */
				//Vote .. 
				Vote voter = new Vote();
				voter.setClassifiers(classifiers);//needs one or more classifiers
				voter.buildClassifier(data);
	}

}
