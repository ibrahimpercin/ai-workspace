package com.wekadeneme.first;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CSV2Arfff {

	public static void main(String[] args) throws Exception {
		
		// Load CSV File
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("C:\\Users\\user\\Desktop\\yapayzeka\\SacramentocrimeJanuary2006.csv"));
		Instances data = loader.getDataSet();
		
		
		// Save data to Arff 
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);// set the dataset we want convert
		//save as Arff
		saver.setFile(new File("C:\\Users\\user\\Desktop\\yapayzeka\\results\\csv3.arff"));
		saver.writeBatch();
		
	}

}
