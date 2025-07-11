/**
 * MongoDB connection module for serverless environments
 * Implements connection caching to work efficiently in Vercel
 */
const { MongoClient } = require('mongodb');
const path = require('path');

// Load environment variables if not in production
if (process.env.NODE_ENV !== 'production') {
  require('dotenv').config({ path: path.resolve(__dirname, '../../.env') });
}

// Connection caching - critical for serverless
let cachedClient = null;
let cachedDb = null;

// Connection options optimized for serverless
const options = {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  maxPoolSize: 10, // Keep connection pool small for serverless
  serverSelectionTimeoutMS: 15000, // Increased timeout for cold starts
  socketTimeoutMS: 30000,
  connectTimeoutMS: 30000
};

/**
 * Connect to MongoDB with connection caching for serverless environments
 * @returns {Promise<{db: object, client: object}>} MongoDB connection objects
 */
async function connectToDatabase() {
  // If we already have a connection, return it
  if (cachedClient && cachedDb) {
    console.log("Using cached MongoDB connection");
    return { client: cachedClient, db: cachedDb };
  }

  if (!process.env.MONGO_URI) {
    throw new Error("MONGO_URI environment variable not defined");
  }

  // If no connection exists, create a new one
  console.log("Creating new MongoDB connection");
  const client = new MongoClient(process.env.MONGO_URI, options);
  
  try {
    await client.connect();
    const db = client.db(process.env.DATABASE_NAME || "wheat_disease_detection");
    
    // Cache the connection
    cachedClient = client;
    cachedDb = db;
    
    console.log("Successfully connected to MongoDB");
    return { client, db };
  } catch (error) {
    console.error("MongoDB connection error:", error);
    throw error;
  }
}

/**
 * Get a MongoDB collection with connection handling
 * @param {string} collectionName - Name of the collection to access
 * @returns {Promise<object>} MongoDB collection
 */
async function getCollection(collectionName) {
  try {
    const { db } = await connectToDatabase();
    return db.collection(collectionName);
  } catch (error) {
    console.error(`Error getting collection ${collectionName}:`, error);
    throw error;
  }
}

module.exports = {
  connectToDatabase,
  getCollection
};