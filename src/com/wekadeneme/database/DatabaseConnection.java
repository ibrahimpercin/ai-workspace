package com.wekadeneme.database;

import weka.core.Instances;
import weka.experiment.InstanceQuery;

public class DatabaseConnection {

	public static void main(String[] args) throws Exception {
		/**
		 * DatabaseUtils.props Editing for Oracle
		 * http://tahasozgen.blogspot.com.tr/2016/10/connection-to-oracle-database-in-weka.html
		 * http://weka.wikispaces.com/Properties+File
		 */

		InstanceQuery query = new InstanceQuery();
		query.setUsername("USERNAME");
		query.setPassword("PASSWORD");
		query.setDatabaseURL("jdbc:oracle:thin:@YOURDATABASE");
		query.setQuery("select * from YOURTABLE t");

		Instances database = query.retrieveInstances();

		System.out.println(database.toString());
	}

}
