import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Home from './components/Home';
import ChurnPredictor from './components/ChurnPredictor';
import ModelExplainer from './components/ModelExplainer';

function App() {
  const [currentPage, setCurrentPage] = useState('home');

  const renderPage = () => {
    switch (currentPage) {
      case 'home': return <Home onStart={() => setCurrentPage('demo')} />;
      case 'demo': return <ChurnPredictor />;
      case 'model': return <ModelExplainer />;
      default: return <Home />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-800">
      <Navbar currentPage={currentPage} setPage={setCurrentPage} />
      <div className="container mx-auto p-4">
        {renderPage()}
      </div>
    </div>
  );
}

export default App;