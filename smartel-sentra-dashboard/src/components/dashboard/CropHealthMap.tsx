
import React from "react";
import { MapPin } from "lucide-react";

const CropHealthMap = () => {
  return (
    <div className="glass-card p-6 h-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium flex items-center gap-2">
          <MapPin className="h-5 w-5 text-smartel-green-500" />
          Crop Health Map
        </h3>
        <div className="flex gap-2">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="text-xs">Healthy</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <span className="text-xs">Warning</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <span className="text-xs">Critical</span>
          </div>
        </div>
      </div>

      <div className="relative bg-white/30 rounded-lg border border-white/30 h-[300px] flex items-center justify-center">
        <div className="absolute inset-0 p-4">
          <div className="relative h-full w-full">
            {/* This is a placeholder for the actual map */}
            <div className="absolute top-[20%] left-[30%]">
              <MapPin className="h-5 w-5 text-green-500" />
            </div>
            <div className="absolute top-[40%] left-[50%]">
              <MapPin className="h-5 w-5 text-yellow-500" />
            </div>
            <div className="absolute top-[60%] left-[70%]">
              <MapPin className="h-5 w-5 text-red-500" />
            </div>
            <div className="absolute top-[30%] left-[60%]">
              <MapPin className="h-5 w-5 text-green-500" />
            </div>
            <div className="absolute top-[70%] left-[40%]">
              <MapPin className="h-5 w-5 text-yellow-500" />
            </div>
          </div>
        </div>
        <p className="text-smartel-gray-500 text-sm italic">Interactive map showing crop health status across monitored farms</p>
      </div>
    </div>
  );
};

export default CropHealthMap;
