/**
 * Authentication routes for user registration and login
 */
const express = require('express');
const jwt = require('jsonwebtoken');
const User = require('../models/User');

const router = express.Router();

/**
 * @route POST /auth/register
 * @desc Register a new user
 * @access Public
 */
router.post('/register', async (req, res) => {
  try {
    const { fullName, email, password } = req.body;
    
    if (!fullName || !email || !password) {
      return res.status(400).json({ 
        status: 'error', 
        message: 'Please provide all required fields: fullName, email, and password' 
      });
    }
    
    // Create user
    const user = await User.create({ fullName, email, password });
    console.log('User created successfully:', { fullName: user.fullName, email: user.email });
    
    // Generate JWT token with better error handling
    let token;
    try {
      const jwtSecret = process.env.JWT_SECRET || 'wheat-disease-detection-jwt-secret';
      console.log('Generating JWT token with secret available:', jwtSecret ? 'Yes' : 'No');
      
      token = jwt.sign(
        { id: user._id, email: user.email },
        jwtSecret,
        { expiresIn: '7d' }
      );
      console.log('JWT token generated successfully');
    } catch (tokenError) {
      console.error('JWT token generation failed:', tokenError);
      return res.status(500).json({ 
        status: 'error', 
        message: 'User created but token generation failed' 
      });
    }
    
    res.status(201).json({
      status: 'success',
      message: 'User registered successfully',
      data: {
        user,
        token
      }
    });
  } catch (error) {
    console.error('Registration error:', error);
    
    // Handle specific errors
    if (error.message.includes('already exists')) {
      return res.status(409).json({ 
        status: 'error', 
        message: error.message 
      });
    }
    
    res.status(500).json({ 
      status: 'error', 
      message: 'Failed to register user' 
    });
  }
});

/**
 * @route POST /auth/login
 * @desc Authenticate user and get token
 * @access Public
 */
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    if (!email || !password) {
      return res.status(400).json({ 
        status: 'error', 
        message: 'Please provide email and password' 
      });
    }
    
    // Authenticate user
    const user = await User.authenticate(email, password);
    
    if (!user) {
      return res.status(401).json({ 
        status: 'error', 
        message: 'Invalid email or password' 
      });
    }
    
    // Generate JWT token
    const token = jwt.sign(
      { id: user._id, email: user.email },
      process.env.JWT_SECRET || 'wheat-disease-detection-jwt-secret',
      { expiresIn: '7d' }
    );
    
    res.status(200).json({
      status: 'success',
      message: 'Login successful',
      data: {
        user,
        token
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ 
      status: 'error', 
      message: 'Authentication failed' 
    });
  }
});

/**
 * @route GET /auth/me
 * @desc Get current user info
 * @access Private
 */
router.get('/me', async (req, res) => {
  try {
    // This route will be protected by auth middleware
    // which will attach the user to the request object
    const user = req.user;
    
    if (!user) {
      return res.status(401).json({
        status: 'error',
        message: 'Not authenticated'
      });
    }
    
    res.status(200).json({
      status: 'success',
      data: {
        user
      }
    });
  } catch (error) {
    console.error('Error getting user profile:', error);
    res.status(500).json({
      status: 'error',
      message: 'Failed to retrieve user information'
    });
  }
});

module.exports = router;