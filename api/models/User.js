/**
 * User model for authentication
 * Defines the schema and methods for user accounts
 */
const { getCollection } = require('../config/database');
const bcrypt = require('bcryptjs');

class User {
  /**
   * Create a new user
   * @param {Object} userData - User data including fullName, email, and password
   * @returns {Promise<Object>} Created user object (without password)
   */
  static async create(userData) {
    try {
      // Validate required fields
      if (!userData.email || !userData.password || !userData.fullName) {
        throw new Error('Missing required fields: fullName, email, and password are required');
      }

      // Get users collection
      const usersCollection = await getCollection('users');

      // Check if user with this email already exists
      const existingUser = await usersCollection.findOne({ email: userData.email });
      if (existingUser) {
        throw new Error('User with this email already exists');
      }

      // Hash the password
      const salt = await bcrypt.genSalt(10);
      const hashedPassword = await bcrypt.hash(userData.password, salt);

      // Create user object
      const newUser = {
        fullName: userData.fullName,
        email: userData.email.toLowerCase(),
        password: hashedPassword,
        createdAt: new Date(),
        updatedAt: new Date()
      };

      // Insert user into database
      const result = await usersCollection.insertOne(newUser);
      
      // Return user without password
      const { password, ...userWithoutPassword } = newUser;
      return userWithoutPassword;
    } catch (error) {
      console.error('Error creating user:', error);
      throw error;
    }
  }

  /**
   * Find a user by email
   * @param {string} email - User's email address
   * @returns {Promise<Object|null>} User object or null if not found
   */
  static async findByEmail(email) {
    try {
      if (!email) return null;
      
      const usersCollection = await getCollection('users');
      return await usersCollection.findOne({ email: email.toLowerCase() });
    } catch (error) {
      console.error('Error finding user by email:', error);
      throw error;
    }
  }

  /**
   * Validate user credentials
   * @param {string} email - User's email address
   * @param {string} password - User's password
   * @returns {Promise<Object|null>} User object without password or null if invalid
   */
  static async authenticate(email, password) {
    try {
      // Find user by email
      const user = await this.findByEmail(email);
      if (!user) return null;

      // Validate password
      const isValidPassword = await bcrypt.compare(password, user.password);
      if (!isValidPassword) return null;

      // Return user without password
      const { password: pwd, ...userWithoutPassword } = user;
      return userWithoutPassword;
    } catch (error) {
      console.error('Error authenticating user:', error);
      throw error;
    }
  }
}

module.exports = User;