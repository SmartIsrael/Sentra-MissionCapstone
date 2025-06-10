
import React from "react";
import { CircleOff, Wifi } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface DeviceData {
  id: string;
  name: string;
  location: string;
  status: "online" | "offline" | "maintenance";
  battery: number;
  signal: number;
  lastSync: string;
}

const mockDevices: DeviceData[] = [
  {
    id: "d1",
    name: "Field Station 01",
    location: "North Sector",
    status: "online",
    battery: 85,
    signal: 92,
    lastSync: "2 mins ago",
  },
  {
    id: "d2",
    name: "Field Station 02",
    location: "East Sector",
    status: "online",
    battery: 67,
    signal: 78,
    lastSync: "5 mins ago",
  },
  {
    id: "d3",
    name: "Field Station 03",
    location: "South Sector",
    status: "offline",
    battery: 23,
    signal: 0,
    lastSync: "5 hours ago",
  },
  {
    id: "d4",
    name: "Field Station 04",
    location: "West Sector",
    status: "maintenance",
    battery: 50,
    signal: 45,
    lastSync: "2 days ago",
  },
];

const DeviceStatus = () => {
  return (
    <div className="glass-card p-6">
      <div className="flex items-center gap-2 mb-6">
        <Wifi className="h-5 w-5 text-smartel-green-500" />
        <h3 className="text-lg font-medium">Device Status</h3>
      </div>

      <div className="space-y-4">
        {mockDevices.map((device) => (
          <div 
            key={device.id} 
            className="p-3 rounded-lg bg-white/50 border border-white/30"
          >
            <div className="flex justify-between items-center mb-2">
              <h4 className="font-medium">{device.name}</h4>
              <div 
                className={`px-2.5 py-0.5 rounded-full text-xs font-medium flex items-center gap-1
                  ${
                    device.status === "online"
                      ? "bg-green-100 text-green-800"
                      : device.status === "offline"
                      ? "bg-red-100 text-red-800"
                      : "bg-amber-100 text-amber-800"
                  }
                `}
              >
                {device.status === "online" ? (
                  <div className="w-2 h-2 rounded-full bg-green-600"></div>
                ) : device.status === "offline" ? (
                  <CircleOff className="w-3 h-3" />
                ) : (
                  <div className="w-2 h-2 rounded-full bg-amber-600"></div>
                )}
                <span>{device.status}</span>
              </div>
            </div>
            <p className="text-sm text-smartel-gray-500 mb-3">{device.location}</p>
            
            <div className="grid grid-cols-2 gap-4 mb-2">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-xs text-smartel-gray-500">Battery</span>
                  <span className="text-xs font-medium">{device.battery}%</span>
                </div>
                <Progress value={device.battery} className="h-1.5" />
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-xs text-smartel-gray-500">Signal</span>
                  <span className="text-xs font-medium">{device.signal}%</span>
                </div>
                <Progress value={device.signal} className="h-1.5" />
              </div>
            </div>
            
            <div className="text-right text-xs text-smartel-gray-400">
              Last sync: {device.lastSync}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DeviceStatus;
