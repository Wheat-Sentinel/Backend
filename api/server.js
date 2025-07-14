/**
 * Wheat Disease Detection - Node.js API Server
 * Main entry point for the Express server
 */
const express = require('express');
const cors = require('cors');
const path = require('path');
const { connectToDatabase } = require('./config/database');

// For local development, load from .env file
// In Vercel, we'll use Environment Variables in the Vercel dashboard
try {
  require('dotenv').config({ path: path.resolve(__dirname, '../.env') });
  console.log("Environment variables loaded");
} catch (error) {
  console.warn("Warning: Could not load .env file, using environment variables");
}

// Import routes
const detectionRoutes = require('./routes/detections');
const authRoutes = require('./routes/auth');
const notificationRoutes = require('./routes/notifications');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(express.json({limit: '50mb'}));

// Register authentication routes BEFORE applying API key middleware
// This ensures auth endpoints remain public
app.use('/auth', authRoutes);

// API Key middleware for protected routes
const apiKeyAuth = (req, res, next) => {
  // Check for API key in various header formats and query parameter
  const apiKey = req.headers['x-api-key'];
  
  const validApiKey = process.env.API_KEY;
  
  console.log("Authentication attempt:");
  console.log("Headers present:", Object.keys(req.headers));
  console.log("API Key present in request:", apiKey ? "Yes" : "No");
  console.log("Valid API key configured in environment:", validApiKey ? "Yes" : "No");
  
  if (apiKey && validApiKey && apiKey === validApiKey) {
    console.log("API Key authentication successful");
    return next();
  }
  
  console.log("API Key Authentication failed");
  console.log("API key match result:", apiKey === validApiKey ? "Match" : "No match");
  
  // If no API key or invalid API key, return 401 Unauthorized
  return res.status(401).json({ 
    status: 'error',
    message: 'Unauthorized: Valid API key required'
  });
};

// Apply API key authentication to protected routes
console.log("Enabling API key authentication for protected routes");
app.use('/detections', apiKeyAuth);
// Only protect the GET tokens route for notifications
app.use('/notifications/tokens', apiKeyAuth);

// Register protected routes AFTER applying API key middleware
app.use('/detections', detectionRoutes);
// Register notifications routes
app.use('/notifications', notificationRoutes);

// Root route - public for health checks
app.get('/', (req, res) => {
  res.json({ 
    message: "Wheat Disease Detection API is running",
    endpoints: {
      detections: "/detections (requires authentication)",
      auth: {
        register: "/auth/register",
        login: "/auth/login",
        me: "/auth/me (requires authentication)"
      },
      notifications: {
        registerToken: "/notifications/register-token",
        removeToken: "/notifications/remove-token",
        listTokens: "/notifications/tokens"
      }
    }
  });
});

// Start server
const startServer = async () => {
  try {
    // Test database connection during server start
    try {
      await connectToDatabase();
      console.log("Database connection successful at startup");
    } catch (dbError) {
      console.error("Warning: Could not connect to database at startup:", dbError.message);
      // Log more detailed error information
      if (dbError.stack) {
        console.error("Database connection error stack:", dbError.stack);
      }
      if (dbError.code) {
        console.error("MongoDB error code:", dbError.code);
      }
      console.log("Server will start anyway and attempt to connect when needed");
    }
    
    // Start server
    app.listen(PORT, () => {
      console.log(`Starting Wheat Disease Detection API server on port ${PORT}...`);
      console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
      console.log(`Database: ${process.env.DATABASE_NAME || 'wheat_disease_detection'}`);
      console.log(`MongoDB URI configured: ${process.env.MONGO_URI ? 'Yes' : 'No'}`);
    });
  } catch (error) {
    console.error(`Error starting server: ${error}`);
    console.error(error.stack);
  }
};

// Call function to start server
startServer();

module.exports = app; // Export for testing purposes