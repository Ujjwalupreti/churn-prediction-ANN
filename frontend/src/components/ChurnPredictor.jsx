import React, { useState } from 'react';
import { motion } from 'framer-motion';

const ChurnPredictor = () => {
  // 1. Initial State with ALL fields from schema.py
  const [formData, setFormData] = useState({
    // Numerical
    tenure: 12,
    MonthlyCharges: 70.0,
    TotalCharges: 1500.0,
    // Categorical
    Contract: 'Month-to-month',
    InternetService: 'Fiber optic',
    PaymentMethod: 'Electronic check',
    // Demographics
    gender: 'Female',
    SeniorCitizen: 0,
    Partner: 'No',
    Dependents: 'No',
    // Services (Yes/No/No internet service)
    PhoneService: 'Yes',
    MultipleLines: 'No',
    OnlineSecurity: 'No',
    OnlineBackup: 'No',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'No',
    StreamingMovies: 'No',
    PaperlessBilling: 'Yes'
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error('API Request Failed');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      alert("Error: Backend not reachable. Ensure it's running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  // Reusable Select Component to keep code clean
  const SelectField = ({ label, name, options }) => (
    <div>
      <label className="text-xs font-bold text-gray-500 uppercase tracking-wide block mb-1">{label}</label>
      <select 
        name={name} 
        value={formData[name]} 
        onChange={handleChange} 
        className="w-full p-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none"
      >
        {options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
      </select>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto mt-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
      
      {/* --- LEFT COLUMN: THE FORM --- */}
      <motion.div 
        initial={{ opacity: 0, x: -20 }} 
        animate={{ opacity: 1, x: 0 }} 
        className="lg:col-span-2 bg-white p-8 rounded-2xl shadow-xl"
      >
        <h2 className="text-2xl font-bold text-indigo-900 mb-6 flex items-center gap-2">
           Customer Profile
        </h2>

        <form onSubmit={handleSubmit} className="space-y-8">
          
          {/* Section 1: Subscription Info */}
          <div>
            <h3 className="text-sm font-bold text-indigo-600 border-b border-indigo-100 pb-2 mb-4">SUBSCRIPTION DETAILS</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="text-xs font-bold text-gray-500 uppercase block mb-1">Tenure (Months)</label>
                <input type="number" name="tenure" value={formData.tenure} onChange={handleChange} className="w-full p-3 bg-gray-50 border rounded-lg" />
              </div>
              <div>
                <label className="text-xs font-bold text-gray-500 uppercase block mb-1">Monthly Bill ($)</label>
                <input type="number" name="MonthlyCharges" value={formData.MonthlyCharges} onChange={handleChange} className="w-full p-3 bg-gray-50 border rounded-lg" />
              </div>
              <div>
                <label className="text-xs font-bold text-gray-500 uppercase block mb-1">Total Charges ($)</label>
                <input type="number" name="TotalCharges" value={formData.TotalCharges} onChange={handleChange} className="w-full p-3 bg-gray-50 border rounded-lg" />
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <SelectField label="Contract Type" name="Contract" options={['Month-to-month', 'One year', 'Two year']} />
              <SelectField label="Payment Method" name="PaymentMethod" options={['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']} />
            </div>
          </div>

          {/* Section 2: Services */}
          <div>
            <h3 className="text-sm font-bold text-indigo-600 border-b border-indigo-100 pb-2 mb-4">ACTIVE SERVICES</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <SelectField label="Internet" name="InternetService" options={['Fiber optic', 'DSL', 'No']} />
              <SelectField label="Tech Support" name="TechSupport" options={['No', 'Yes', 'No internet service']} />
              <SelectField label="Online Security" name="OnlineSecurity" options={['No', 'Yes', 'No internet service']} />
              <SelectField label="Device Prot." name="DeviceProtection" options={['No', 'Yes', 'No internet service']} />
            </div>
          </div>

          {/* Section 3: Demographics */}
          <div>
            <h3 className="text-sm font-bold text-indigo-600 border-b border-indigo-100 pb-2 mb-4">DEMOGRAPHICS</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <SelectField label="Gender" name="gender" options={['Female', 'Male']} />
              <SelectField label="Senior Citizen" name="SeniorCitizen" options={[0, 1]} />
              <SelectField label="Partner" name="Partner" options={['No', 'Yes']} />
              <SelectField label="Dependents" name="Dependents" options={['No', 'Yes']} />
            </div>
          </div>

          <button 
            type="submit" 
            disabled={loading} 
            className={`w-full py-4 rounded-xl font-bold text-lg text-white shadow-lg transition-all 
              ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-indigo-500/30'}`}
          >
            {loading ? 'Analyzing Neural Network...' : 'Analyze Churn Risk'}
          </button>
        </form>
      </motion.div>

      {/* --- RIGHT COLUMN: RESULTS --- */}
      <div className="space-y-6">
        {result ? (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }} 
            animate={{ opacity: 1, scale: 1 }} 
            className={`p-8 rounded-2xl shadow-xl text-white h-full flex flex-col justify-center items-center text-center
              ${result.prediction === 'Yes' ? 'bg-gradient-to-br from-red-500 to-rose-600' : 'bg-gradient-to-br from-emerald-500 to-teal-600'}`}
          >
            <div className="text-6xl mb-4">{result.prediction === 'Yes' ? '⚠️' : '✅'}</div>
            <h2 className="text-3xl font-bold mb-2">{result.prediction === 'Yes' ? 'High Churn Risk' : 'Safe Customer'}</h2>
            <div className="text-5xl font-mono font-bold opacity-90 mb-6">{(result.probability * 100).toFixed(1)}%</div>
            
            <div className="w-full bg-black/20 h-4 rounded-full overflow-hidden mb-8">
              <motion.div 
                initial={{ width: 0 }} 
                animate={{ width: `${result.probability * 100}%` }} 
                transition={{ duration: 1.5, type: "spring" }} 
                className="h-full bg-white" 
              />
            </div>

            <div className="bg-white/10 p-4 rounded-lg backdrop-blur-sm border border-white/20 w-full">
              <p className="text-sm font-medium uppercase tracking-wider opacity-75 mb-1">AI Recommendation</p>
              <p className="font-medium">
                {result.prediction === 'Yes' 
                  ? "Offer 20% discount on 1-year contract renewal immediately."
                  : "Customer is healthy. Suggest adding 'Online Security' package."}
              </p>
            </div>
          </motion.div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center bg-white/50 rounded-2xl border-2 border-dashed border-gray-200 p-8 text-center text-gray-400">
            <div className="text-6xl mb-4 grayscale opacity-50">📊</div>
            <p className="text-lg font-medium">Ready to Analyze</p>
            <p className="text-sm">Fill out the customer profile on the left to generate a real-time risk assessment.</p>
          </div>
        )}
      </div>

    </div>
  );
};

export default ChurnPredictor;