const path = require('path');

module.exports = {
  mode: 'development',
  entry: './src/index.js', // Your main JavaScript file
  output: {
    path: path.resolve(__dirname, 'public'),
    filename: 'bundle.js',
  }
};