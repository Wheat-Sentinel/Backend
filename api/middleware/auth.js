/**
 * Authentication middleware
 * Verifies JWT tokens and protects routes
 */
const jwt = require('jsonwebtoken');
const User = require('../models/User');

/**
 * Middleware to protect routes with JWT authentication
 * Verifies the token and attaches the user to the request object
 */
const authMiddleware = async (req, res, next) => {
  try {
    // Get token from header
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ 
        status: 'error', 
        message: 'Authentication required. No valid token provided.' 
      });
    }
    
    // Extract token from header
    const token = authHeader.split(' ')[1];
    
    if (!token) {
      return res.status(401).json({ 
        status: 'error', 
        message: 'Authentication required. No token provided.' 
      });
    }
    
    try {
      // Verify token
      const decoded = jwt.verify(
        token, 
        process.env.JWT_SECRET
      );
      
      // Find user by email
      const user = await User.findByEmail(decoded.email);
      
      if (!user) {
        return res.status(401).json({ 
          status: 'error', 
          message: 'User not found' 
        });
      }
      
      // Remove password from user object
      const { password, ...userWithoutPassword } = user;
      
      // Attach user to request
      req.user = userWithoutPassword;
      next();
    } catch (error) {
      console.error('Token verification error:', error);
      return res.status(401).json({ 
        status: 'error', 
        message: 'Invalid or expired token' 
      });
    }
  } catch (error) {
    console.error('Authentication middleware error:', error);
    return res.status(500).json({ 
      status: 'error', 
      message: 'Authentication failed' 
    });
  }
};

module.exports = authMiddleware;