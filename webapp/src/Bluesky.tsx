import React, { useEffect } from "react";

const Bluesky: React.FC = () => {
  useEffect(() => {
    // Dynamically load the Skyview script
    const script = document.createElement("script");
    script.src = "build/index.js";
    script.async = true;
    document.body.appendChild(script);

    return () => {
      // Cleanup script if the component unmounts
      document.body.removeChild(script);
    };
  }, []);

  return (
    <div className="bg-black text-white w-full h-full">
      <div id="skyview-container">
        {/* Skyview custom component */}
        <skyview-app embed="true"></skyview-app>
      </div>
    </div>
  );
};

export default Bluesky; 
