import React, { useState } from 'react';
import { motion } from 'framer-motion';

const ModelExplainer = () => {
  const [simulationInput, setSimulationInput] = useState(0.5); // 0 to 1

  // Simulate activation: Higher input = brighter nodes in next layer
  const hiddenActivation = Math.min(1, simulationInput * 1.5); 
  const outputProb = 1 / (1 + Math.exp(-((hiddenActivation * 4) - 2))); // Sigmoid simulation

  const layers = [
    { 
      name: "Input Layer", 
      nodes: 3, 
      color: "bg-blue-500", 
      activeColor: "bg-blue-400",
      desc: "Raw Data (Tenure, Cost)",
      value: simulationInput 
    },
    { 
      name: "Hidden Layer (ReLU)", 
      nodes: 4, 
      color: "bg-indigo-600", 
      activeColor: "bg-indigo-400",
      desc: "Pattern Extraction",
      value: hiddenActivation 
    },
    { 
      name: "Output (Sigmoid)", 
      nodes: 1, 
      color: outputProb > 0.5 ? "bg-red-500" : "bg-emerald-500", 
      activeColor: outputProb > 0.5 ? "bg-red-400" : "bg-emerald-400",
      desc: "Final Probability",
      value: outputProb
    }
  ];

  return (
    <div className="max-w-6xl mx-auto py-12 px-4">
      <div className="text-center mb-12">
        <h2 className="text-3xl font-bold text-indigo-900">Interactive Neural Network</h2>
        <p className="text-gray-500 mt-2">Drag the slider to see how data flows through the layers .</p>
      </div>

      {/* CONTROL PANEL */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 mb-12 max-w-lg mx-auto text-center">
        <label className="text-sm font-bold text-gray-500 uppercase tracking-wide">
          Simulate Customer Risk Level
        </label>
        <div className="flex items-center gap-4 mt-2">
          <span className="text-xs font-bold text-gray-400">Low Risk</span>
          <input 
            type="range" 
            min="0" 
            max="1" 
            step="0.01" 
            value={simulationInput}
            onChange={(e) => setSimulationInput(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
          />
          <span className="text-xs font-bold text-gray-400">High Risk</span>
        </div>
      </div>

      {/* VISUALIZATION */}
      <div className="flex flex-col md:flex-row items-center justify-center gap-12 md:gap-24 relative">
        
        {layers.map((layer, layerIdx) => (
          <div key={layerIdx} className="relative z-10 flex flex-col items-center gap-4">
            <div className="bg-white p-4 rounded-xl shadow-lg border border-gray-100 min-w-[160px] text-center">
              <h3 className="font-bold text-gray-800 text-sm">{layer.name}</h3>
              <p className="text-xs text-gray-400">{layer.desc}</p>
            </div>

            {/* NEURONS */}
            <div className="flex flex-col gap-3">
              {[...Array(layer.nodes)].map((_, nodeIdx) => (
                <motion.div
                  key={nodeIdx}
                  animate={{ 
                    scale: 1 + (layer.value * 0.2), // Pulse size based on activity
                    opacity: 0.3 + (layer.value * 0.7) // Brightness based on activity
                  }}
                  className={`w-12 h-12 rounded-full ${layer.color} shadow-inner border-2 border-white flex items-center justify-center text-white text-xs font-bold`}
                >
                  {(layer.value * 100).toFixed(0)}%
                </motion.div>
              ))}
            </div>
          </div>
        ))}

        {/* CONNECTING LINES (SVG) */}
        <svg className="absolute top-1/2 left-0 w-full h-full -translate-y-1/2 -z-10 pointer-events-none opacity-20">
          <line x1="20%" y1="50%" x2="50%" y2="50%" stroke="black" strokeWidth="2" strokeDasharray="5,5" />
          <line x1="50%" y1="50%" x2="80%" y2="50%" stroke="black" strokeWidth="2" strokeDasharray="5,5" />
        </svg>

      </div>

      {/* EXPLANATION CARDS */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16">
        <div className="bg-blue-50 p-6 rounded-xl border border-blue-100">
          <h4 className="font-bold text-blue-900 mb-2">1. The Input</h4>
          <p className="text-sm text-blue-800">
            We feed raw numbers (Tenure, Bill Amount) into the input nodes. We normalize them so large numbers (like $1500) don't overwhelm the math.
          </p>
        </div>
        <div className="bg-indigo-50 p-6 rounded-xl border border-indigo-100">
          <h4 className="font-bold text-indigo-900 mb-2">2. Hidden Pattern Matching</h4>
          <p className="text-sm text-indigo-800">
            The "Hidden Layer" uses <b>ReLU</b> activation. Think of it as a filter: it blocks irrelevant signals (negatives) and amplifies important patterns (positives).
          </p>
        </div>
        <div className="bg-emerald-50 p-6 rounded-xl border border-emerald-100">
          <h4 className="font-bold text-emerald-900 mb-2">3. The Decision</h4>
          <p className="text-sm text-emerald-800">
            The final node uses a <b>Sigmoid</b> function. It squashes the total sum into a clean probability between 0% and 100%.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ModelExplainer;