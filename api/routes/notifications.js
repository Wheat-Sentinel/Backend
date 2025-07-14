/**
 * Push Notifications API Routes
 * Handles registration and management of Expo push notification tokens
 */

const express = require('express');
const router = express.Router();
const { authMiddleware } = require('../middleware/auth');
const { getCollection } = require('../config/database');

/**
 * @route POST /api/notifications/register-token
 * @desc Register a new push notification token
 * @access Public - Consider using auth for production
 */
router.post('/-token', async (req, res) => {
  try {
    const { token } = req.body;
    
    // Validate token format (basic validation)
    if (!token || typeof token !== 'string' || !token.startsWith('ExponentPushToken[')) {
      return res.status(400).json({ 
        success: false, 
        message: 'Invalid Expo push token format' 
      });
    }
    
    // Get tokens collection
    const tokensCollection = await getCollection('push_tokens');
    
    // Check if token already exists
    const existingToken = await tokensCollection.findOne({ token });
    
    if (existingToken) {
      // Update last seen timestamp if token exists
      await tokensCollection.updateOne(
        { token }, 
        { $set: { last_seen: new Date() }}
      );
      
      return res.status(200).json({ 
        success: true, 
        message: 'Token already registered',
        alreadyExists: true
      });
    }
    
    // Add the new token with metadata
    const result = await tokensCollection.insertOne({
      token,
      created_at: new Date(),
      last_seen: new Date(),
      device_info: req.body.deviceInfo || {}
    });
    
    if (!result.acknowledged) {
      return res.status(500).json({ 
        success: false, 
        message: 'Error saving token' 
      });
    }
    
    return res.status(201).json({ 
      success: true, 
      message: 'Push token registered successfully' 
    });
    
  } catch (error) {
    console.error('Error registering push token:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Server error',
      error: error.message 
    });
  }
});

/**
 * @route DELETE /api/notifications/remove-token
 * @desc Remove a push notification token
 * @access Public - Consider using auth for production
 */
router.delete('/remove-token', async (req, res) => {
  try {
    const { token } = req.body;
    
    if (!token) {
      return res.status(400).json({ 
        success: false, 
        message: 'Token is required' 
      });
    }
    
    // Get tokens collection
    const tokensCollection = await getCollection('push_tokens');
    
    // Remove the token
    const result = await tokensCollection.deleteOne({ token });
    
    if (result.deletedCount === 0) {
      return res.status(404).json({ 
        success: false, 
        message: 'Token not found' 
      });
    }
    
    return res.status(200).json({ 
      success: true, 
      message: 'Push token removed successfully' 
    });
    
  } catch (error) {
    console.error('Error removing push token:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Server error',
      error: error.message 
    });
  }
});

/**
 * @route GET /api/notifications/tokens
 * @desc Get all registered push tokens (protected route)
 * @access Private
 */
router.get('/tokens', authMiddleware, async (req, res) => {
  try {
    // Get tokens collection
    const tokensCollection = await getCollection('push_tokens');
    
    // Get all tokens
    const tokens = await tokensCollection.find({}).toArray();
    
    return res.status(200).json({ 
      success: true, 
      count: tokens.length,
      tokens 
    });
    
  } catch (error) {
    console.error('Error fetching push tokens:', error);
    return res.status(500).json({ 
      success: false, 
      message: 'Server error',
      error: error.message 
    });
  }
});

module.exports = router;