/**
 * Detection model for wheat disease detection
 * Handles data validation, storage and retrieval operations
 */
const { getCollection } = require('../config/database');

class Detection {
  /**
   * Create a new disease detection record
   * @param {Object} detectionData - Detection data including disease, confidence, timestamp, etc.
   * @returns {Promise<Object>} Created detection object with database ID
   */
  static async create(detectionData) {
    try {
      // Check for required fields
      const requiredFields = ['disease', 'confidence', 'timestamp', 'zone_id', 'device_id', 'image_url',"latitude", "longitude"];
      for (const field of requiredFields) {
        if (!detectionData[field]) {
          throw new Error(`Missing required field: ${field}`);
        }
      }
      
      const detectionsCollection = await getCollection("detections");
      
      // Prepare detection document for storage
      const detectionDocument = {
        disease: detectionData.disease,
        confidence: detectionData.confidence,
        timestamp: detectionData.timestamp,
        zone_id: detectionData.zone_id,
        device_id: detectionData.device_id,
        image_url: detectionData.image_url,
        created_at: new Date(),
        latitude: detectionData.latitude,
        longitude: detectionData.longitude,
      };
      
      // Insert into database
      const result = await detectionsCollection.insertOne(detectionDocument);
      
      console.log(`   - Saved to MongoDB with ID: ${result.insertedId}`);
      console.log(`   - Image URL from Supabase: ${detectionData.image_url}`);
      
      // Return the created object with its ID
      return { 
        id: result.insertedId,
        ...detectionDocument
      };
    } catch (error) {
      console.error(`Error creating detection: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get all detections from the database
   * @returns {Promise<Array>} Array of detection objects
   */
  static async getAll() {
    try {
      const detectionsCollection = await getCollection("detections");
      
      // Get all detections from the database
      const mongoDetections = await detectionsCollection.find({}).toArray();
      
      // Format the detections for API response
      return mongoDetections.map(detection => {
        const formatted = { ...detection };
        if (formatted._id) {
          formatted.id = formatted._id.toString();
          delete formatted._id;
        }
        if (formatted.created_at && formatted.created_at instanceof Date) {
          formatted.created_at = formatted.created_at.toISOString();
        }
        return formatted;
      });
    } catch (error) {
      console.error(`Error getting detections: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get a detection by ID
   * @param {string} id - Detection ID
   * @returns {Promise<Object|null>} Detection object or null if not found
   */
  static async getById(id) {
    try {
      const detectionsCollection = await getCollection("detections");
      
      // Find the detection by ID
      const detection = await detectionsCollection.findOne({ _id: id });
      
      if (!detection) return null;
      
      // Format the detection
      const formatted = { ...detection };
      formatted.id = formatted._id.toString();
      delete formatted._id;
      
      if (formatted.created_at && formatted.created_at instanceof Date) {
        formatted.created_at = formatted.created_at.toISOString();
      }
      
      return formatted;
    } catch (error) {
      console.error(`Error getting detection by ID: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get detections by filter criteria
   * @param {Object} filter - Filter criteria
   * @returns {Promise<Array>} Array of matching detection objects
   */
  static async getByFilter(filter = {}) {
    try {
      const detectionsCollection = await getCollection("detections");
      
      // Find detections matching the filter
      const detections = await detectionsCollection.find(filter).toArray();
      
      // Format the detections
      return detections.map(detection => {
        const formatted = { ...detection };
        if (formatted._id) {
          formatted.id = formatted._id.toString();
          delete formatted._id;
        }
        if (formatted.created_at && formatted.created_at instanceof Date) {
          formatted.created_at = formatted.created_at.toISOString();
        }
        return formatted;
      });
    } catch (error) {
      console.error(`Error getting detections by filter: ${error.message}`);
      throw error;
    }
  }
}

module.exports = Detection;