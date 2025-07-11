/**
 * Detections routes for the wheat disease detection API
 * Handles receiving and retrieving detection data
 * Using environment variables instead of config file
 */
const express = require('express');
const router = express.Router();
const path = require('path');
const Detection = require('../models/Detection');

// Load environment variables for local development
if (process.env.NODE_ENV !== 'production') {
  require('dotenv').config({ path: path.resolve(__dirname, '../../.env') });
}

/**
 * @route POST /detections
 * @desc Receive disease detection data and store it in MongoDB
 */
router.post('/', async (req, res) => {
  try {
    const detectionData = req.body;
    
    try {
      // Create detection using the model
      const detection = await Detection.create(detectionData);
      
      return res.status(200).json({ 
        status: 'success', 
        message: `Detection recorded: ${detection.disease}`, 
        image_url: detection.image_url 
      });
    } catch (error) {
      if (error.message.includes('Missing required field')) {
        return res.status(400).json({ 
          status: 'error', 
          message: error.message 
        });
      }
      
      if (error.message.includes('Failed to connect to MongoDB')) {
        return res.status(503).json({ 
          status: 'error',
          message: 'Database connection unavailable',
          detail: 'Could not establish connection to MongoDB. Check if your MongoDB Atlas IP whitelist includes 0.0.0.0/0.',
          image_url: detectionData.image_url 
        });
      }
      
      throw error;
    }
  } catch (error) {
    console.error(`Error processing detection: ${error.message}`);
    return res.status(500).json({ 
      status: 'error', 
      message: `Error processing detection: ${error.message}` 
    });
  }
});

/**
 * @route GET /detections
 * @desc List all disease detections stored in the database
 */
router.get('/', async (req, res) => {
  try {
    try {
      // Get all detections using the model
      const detections = await Detection.getAll();
      
      return res.status(200).json({
        count: detections.length,
        detections: detections
      });
    } catch (error) {
      if (error.message.includes('Failed to connect to MongoDB')) {
        return res.status(503).json({ 
          status: 'error',
          message: 'Database connection unavailable',
          detail: 'Could not establish connection to MongoDB. Check if your MongoDB Atlas IP whitelist includes 0.0.0.0/0.'
        });
      }
      
      throw error;
    }
  } catch (error) {
    console.error(`Error retrieving detections: ${error.message}`);
    return res.status(500).json({ 
      status: 'error', 
      message: `Error retrieving detections: ${error.message}` 
    });
  }
});

/**
 * @route GET /detections/:id
 * @desc Get a specific detection by ID
 */
router.get('/:id', async (req, res) => {
  try {
    const id = req.params.id;
    
    try {
      const detection = await Detection.getById(id);
      
      if (!detection) {
        return res.status(404).json({
          status: 'error',
          message: `Detection with ID ${id} not found`
        });
      }
      
      return res.status(200).json({
        status: 'success',
        detection: detection
      });
    } catch (error) {
      if (error.message.includes('Failed to connect to MongoDB')) {
        return res.status(503).json({ 
          status: 'error',
          message: 'Database connection unavailable'
        });
      }
      
      throw error;
    }
  } catch (error) {
    console.error(`Error retrieving detection: ${error.message}`);
    return res.status(500).json({ 
      status: 'error', 
      message: `Error retrieving detection: ${error.message}` 
    });
  }
});

/**
 * @route GET /detections/filter
 * @desc Get detections by filter criteria
 */
router.get('/filter', async (req, res) => {
  try {
    // Extract filter criteria from query params
    const { disease, device_id, zone_id } = req.query;
    const filter = {};
    
    if (disease) filter.disease = disease;
    if (device_id) filter.device_id = device_id;
    if (zone_id) filter.zone_id = zone_id;
    
    try {
      const detections = await Detection.getByFilter(filter);
      
      return res.status(200).json({
        status: 'success',
        count: detections.length,
        detections: detections
      });
    } catch (error) {
      if (error.message.includes('Failed to connect to MongoDB')) {
        return res.status(503).json({ 
          status: 'error',
          message: 'Database connection unavailable'
        });
      }
      
      throw error;
    }
  } catch (error) {
    console.error(`Error retrieving filtered detections: ${error.message}`);
    return res.status(500).json({ 
      status: 'error', 
      message: `Error retrieving filtered detections: ${error.message}` 
    });
  }
});

module.exports = router;