
import React from "react";
import { Button } from "@/components/ui/button";
import { MapPin, Plus, Search, Settings, Battery, Signal, RefreshCw } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface Device {
  id: string;
  name: string;
  location: string;
  coordinates: string;
  farmer: string;
  status: "online" | "offline" | "maintenance";
  lastSync: string;
  battery: number;
  signal: number;
  uptime: string;
}

const mockDevices: Device[] = [
  {
    id: "d1",
    name: "Field Station 01",
    location: "North Sector, Farm 3",
    coordinates: "1.2345° S, 36.5678° E",
    farmer: "James Mwangi",
    status: "online",
    lastSync: "2 minutes ago",
    battery: 85,
    signal: 92,
    uptime: "14 days",
  },
  {
    id: "d2",
    name: "Field Station 02",
    location: "East Sector, Farm 5",
    coordinates: "1.3456° S, 36.6789° E",
    farmer: "Amina Kimani",
    status: "online",
    lastSync: "5 minutes ago",
    battery: 67,
    signal: 78,
    uptime: "7 days",
  },
  {
    id: "d3",
    name: "Field Station 03",
    location: "South Sector, Farm 8",
    coordinates: "1.4567° S, 36.7890° E",
    farmer: "David Ochieng",
    status: "offline",
    lastSync: "5 hours ago",
    battery: 23,
    signal: 0,
    uptime: "2 days",
  },
  {
    id: "d4",
    name: "Field Station 04",
    location: "West Sector, Farm 12",
    coordinates: "1.5678° S, 36.8901° E",
    farmer: "Sarah Wanjiku",
    status: "maintenance",
    lastSync: "2 days ago",
    battery: 50,
    signal: 45,
    uptime: "5 hours",
  },
];

const getStatusColor = (status: Device["status"]) => {
  switch (status) {
    case "online":
      return "bg-green-100 text-green-800";
    case "offline":
      return "bg-red-100 text-red-800";
    case "maintenance":
      return "bg-amber-100 text-amber-800";
    default:
      return "bg-gray-100 text-gray-800";
  }
};

const getBatteryColor = (level: number) => {
  if (level >= 70) return "text-green-600";
  if (level >= 30) return "text-amber-600";
  return "text-red-600";
};

const Devices = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient">Devices</h1>
          <p className="mt-1 text-smartel-gray-500">
            Monitor and manage all deployed Save-Bot devices in the field.
          </p>
        </div>
        <Button className="bg-smartel-green-500 hover:bg-smartel-green-600">
          <Plus className="h-4 w-4 mr-2" />
          Add Device
        </Button>
      </div>

      <div className="glass-card p-6">
        <div className="flex flex-col sm:flex-row justify-between gap-4 mb-6">
          <div className="relative w-full sm:w-64 md:w-96">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-smartel-gray-500" />
            <input
              type="text"
              placeholder="Search devices..."
              className="pl-9 pr-4 py-2 w-full rounded-lg text-sm bg-white/60 border border-white/30 focus:outline-none focus:ring-2 focus:ring-smartel-green-400"
            />
          </div>
          
          <div className="flex gap-2">
            <Button variant="outline" className="text-smartel-gray-600 bg-white/60 border-white/30">
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {mockDevices.map((device) => (
            <div
              key={device.id}
              className="p-5 rounded-lg bg-white/60 border border-white/30 hover:bg-white/80 transition-colors"
            >
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 flex-shrink-0 flex items-center justify-center rounded-full bg-smartel-teal-100 text-smartel-teal-600">
                    <MapPin className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-medium">{device.name}</h3>
                    <p className="text-sm text-smartel-gray-500">
                      {device.location}
                    </p>
                  </div>
                </div>
                
                <span
                  className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                    device.status
                  )}`}
                >
                  {device.status}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <p className="text-xs text-smartel-gray-500 mb-1">Farmer</p>
                  <p className="text-sm font-medium">{device.farmer}</p>
                </div>
                <div>
                  <p className="text-xs text-smartel-gray-500 mb-1">Coordinates</p>
                  <p className="text-sm">{device.coordinates}</p>
                </div>
                <div>
                  <p className="text-xs text-smartel-gray-500 mb-1">Last Sync</p>
                  <p className="text-sm">{device.lastSync}</p>
                </div>
                <div>
                  <p className="text-xs text-smartel-gray-500 mb-1">Uptime</p>
                  <p className="text-sm">{device.uptime}</p>
                </div>
              </div>

              <div className="space-y-3 mb-4">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs flex items-center gap-1 text-smartel-gray-500">
                      <Battery className="h-3 w-3" /> Battery
                    </span>
                    <span className={`text-xs font-medium ${getBatteryColor(device.battery)}`}>
                      {device.battery}%
                    </span>
                  </div>
                  <Progress value={device.battery} className="h-1.5" />
                </div>
                
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs flex items-center gap-1 text-smartel-gray-500">
                      <Signal className="h-3 w-3" /> Signal Strength
                    </span>
                    <span className="text-xs font-medium">
                      {device.signal}%
                    </span>
                  </div>
                  <Progress value={device.signal} className="h-1.5" />
                </div>
              </div>

              <div className="flex justify-end gap-2">
                <Button size="sm" variant="outline" className="bg-white/60 border-white/30">
                  <Settings className="h-3.5 w-3.5 mr-1.5" />
                  Configure
                </Button>
                <Button size="sm" className="bg-smartel-teal-500 hover:bg-smartel-teal-600">
                  View Details
                </Button>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 flex items-center justify-between">
          <p className="text-sm text-smartel-gray-500">
            Showing 4 of 12 devices
          </p>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="bg-white/60 border-white/30">
              Previous
            </Button>
            <Button variant="outline" size="sm" className="bg-white/60 border-white/30">
              Next
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Devices;
