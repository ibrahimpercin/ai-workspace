package com.wekadeneme;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class StartWeka {

	public static void main(String[] args) throws Exception {
		DataSource dataSource = new DataSource("./sources/datasets/segment-challenge.arff");
		DataSource dataTestSource = new DataSource("./sources/datasets/segment-test.arff");

		Instances trainData = dataSource.getDataSet();
		Instances testData = dataTestSource.getDataSet();

		// Setting class attribute
		/**
		 * classIndex() --> Returns the class attribute's index. Returns negative number
		 * if it's undefined.
		 */
		if (trainData.classIndex() == -1)
			trainData.setClassIndex(trainData.numAttributes() - 1);
		if (testData.classIndex() == -1)
			testData.setClassIndex(testData.numAttributes() - 1);

		// Naive Bayes
		NaiveBayes naiveBayes = new NaiveBayes();
		naiveBayes.buildClassifier(trainData);

		// Logistic Regression
		Logistic logistic = new Logistic();
		logistic.buildClassifier(trainData);

		// OneR
		OneR oneR = new OneR();
		oneR.buildClassifier(trainData);

		// Random Forest
		RandomForest forest = new RandomForest();
		forest.buildClassifier(trainData);

		// Linear Regression
		// LinearRegression linearRegression = new LinearRegression();
		// linearRegression.buildClassifier(train);

		J48 j48 = new J48();
		j48.buildClassifier(trainData);

		Evaluation eval = new Evaluation(trainData);
		eval.crossValidateModel(forest, testData, 10, new Random(1));

		// Evaluation evaluation = new EM();

		System.out.println(eval.toSummaryString("\nResults\n======\n", true));

		System.out.println("Correct % = " + eval.pctCorrect());
		System.out.println("Incorrect % = " + eval.pctIncorrect());
		System.out.println("AUC = " + eval.areaUnderROC(1));
		System.out.println("kappa = " + eval.kappa());
		System.out.println("MAE = " + eval.meanAbsoluteError());
		System.out.println("RMSE = " + eval.rootMeanSquaredError());
		System.out.println("RAE = " + eval.relativeAbsoluteError());
		System.out.println("RRSE = " + eval.rootRelativeSquaredError());
		System.out.println("Precision = " + eval.precision(1));
		System.out.println("Recall = " + eval.recall(1));
		System.out.println("fMeasure = " + eval.fMeasure(1));
		System.out.println("Error Rate = " + eval.errorRate());

		// the confusion matrix
		System.out.println(eval.toMatrixString("\n=== Overall Confusion Matrix ===\n"));

	}

}
