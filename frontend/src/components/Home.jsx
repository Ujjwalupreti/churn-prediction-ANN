import { motion } from 'framer-motion';

const Home = ({ onStart }) => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] text-center">
      <motion.div 
        initial={{ opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-24 h-24 bg-indigo-100 rounded-full flex items-center justify-center text-5xl mb-8"
      >
        🚀
      </motion.div>
      <h1 className="text-5xl font-bold text-indigo-900 mb-4">Telco Churn Prediction System</h1>
      <p className="text-xl text-gray-600 max-w-2xl mb-8">
        An Enterprise-grade Artificial Neural Network designed to identify at-risk customers before they leave.
      </p>
      <button 
        onClick={onStart}
        className="px-8 py-4 bg-indigo-600 text-white rounded-full font-bold text-lg shadow-lg hover:bg-indigo-700 transition-transform transform hover:scale-105"
      >
        Launch Demo System
      </button>
    </div>
  );
};

export default Home;