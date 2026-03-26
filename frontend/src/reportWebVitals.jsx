const reportWebVitals = onPerfEntry => {
  if (onPerfEntry && onPerfEntry instanceof Function) {
    import('web-vitals').then(({ onCLS, onFID, onFCP, onLCP, onTTFB }) => {
      onCLS(onPerfEntry);  // Cumulative Layout Shift
      onFID(onPerfEntry);  // First Input Delay
      onFCP(onPerfEntry);  // First Contentful Paint
      onLCP(onPerfEntry);  // Largest Contentful Paint
      onTTFB(onPerfEntry); // Time to First Byte
    });
  }
};

export default reportWebVitals;