import React from 'react';

const Navbar = ({ currentPage, setPage }) => {
  const navItems = [
    { id: 'home', label: 'Home' },
    { id: 'demo', label: 'Live Demo' },
    { id: 'model', label: 'How ANN Works' },
  ];

  return (
    <nav className="bg-white shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => setPage('home')}>
            <span className="text-2xl">🔮</span>
            <span className="font-bold text-xl text-indigo-900">Retention AI</span>
          </div>
          <div className="flex items-center space-x-8">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setPage(item.id)}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors
                  ${currentPage === item.id 
                    ? 'text-indigo-600 bg-indigo-50' 
                    : 'text-gray-500 hover:text-indigo-600'}`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;