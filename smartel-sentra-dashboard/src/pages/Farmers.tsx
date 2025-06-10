
import React from "react";
import { Button } from "@/components/ui/button";
import { ChevronDown, Plus, Search, User } from "lucide-react";

interface Farmer {
  id: string;
  name: string;
  location: string;
  farms: number;
  devices: number;
  lastActive: string;
  healthScore: number;
}

const mockFarmers: Farmer[] = [
  {
    id: "f1",
    name: "James Mwangi",
    location: "Kigali Province",
    farms: 3,
    devices: 2,
    lastActive: "Today",
    healthScore: 92,
  },
  {
    id: "f2",
    name: "Amina Uwase",
    location: "Eastern Province",
    farms: 1,
    devices: 1,
    lastActive: "Yesterday",
    healthScore: 78,
  },
  {
    id: "f3",
    name: "David Niyonzima",
    location: "Northern Province",
    farms: 2,
    devices: 2,
    lastActive: "3 days ago",
    healthScore: 85,
  },
  {
    id: "f4",
    name: "Sarah Mukamana",
    location: "Western Province",
    farms: 4,
    devices: 3,
    lastActive: "1 week ago",
    healthScore: 67,
  },
  {
    id: "f5",
    name: "John Mugabo",
    location: "Southern Province",
    farms: 2,
    devices: 1,
    lastActive: "Today",
    healthScore: 95,
  },
];

const getHealthScoreColor = (score: number) => {
  if (score >= 90) return "bg-green-100 text-green-800";
  if (score >= 70) return "bg-blue-100 text-blue-800";
  if (score >= 50) return "bg-amber-100 text-amber-800";
  return "bg-red-100 text-red-800";
};

const Farmers = () => {
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient">Rwanda Farmers</h1>
          <p className="mt-1 text-smartel-gray-500">
            Manage and monitor all registered farmers across Rwanda's provinces and districts.
          </p>
        </div>
        <Button className="bg-smartel-green-500 hover:bg-smartel-green-600">
          <Plus className="h-4 w-4 mr-2" />
          Add Farmer
        </Button>
      </div>

      <div className="glass-card p-6">
        <div className="flex flex-col sm:flex-row justify-between gap-4 mb-6">
          <div className="relative w-full sm:w-64 md:w-96">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-smartel-gray-500" />
            <input
              type="text"
              placeholder="Search farmers..."
              className="pl-9 pr-4 py-2 w-full rounded-lg text-sm bg-white/60 border border-white/30 focus:outline-none focus:ring-2 focus:ring-smartel-green-400"
            />
          </div>
          
          <div className="flex gap-2">
            <Button variant="outline" className="text-smartel-gray-600 bg-white/60 border-white/30">
              Province
              <ChevronDown className="ml-2 h-4 w-4" />
            </Button>
            <Button variant="outline" className="text-smartel-gray-600 bg-white/60 border-white/30">
              Health Score
              <ChevronDown className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="text-sm text-smartel-gray-600 border-b border-white/30">
              <tr>
                <th className="px-4 py-3 font-medium">Farmer</th>
                <th className="px-4 py-3 font-medium">Province</th>
                <th className="px-4 py-3 font-medium text-center">Farms</th>
                <th className="px-4 py-3 font-medium text-center">Devices</th>
                <th className="px-4 py-3 font-medium">Last Active</th>
                <th className="px-4 py-3 font-medium">Health Score</th>
                <th className="px-4 py-3 font-medium sr-only">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/30">
              {mockFarmers.map((farmer) => (
                <tr 
                  key={farmer.id} 
                  className="hover:bg-white/30 transition-colors"
                >
                  <td className="px-4 py-4">
                    <div className="flex items-center gap-3">
                      <div className="h-9 w-9 rounded-full flex items-center justify-center bg-smartel-green-100 text-smartel-green-500">
                        <User className="h-5 w-5" />
                      </div>
                      <span className="font-medium">{farmer.name}</span>
                    </div>
                  </td>
                  <td className="px-4 py-4 text-smartel-gray-600">
                    {farmer.location}
                  </td>
                  <td className="px-4 py-4 text-center">{farmer.farms}</td>
                  <td className="px-4 py-4 text-center">{farmer.devices}</td>
                  <td className="px-4 py-4 text-smartel-gray-600">
                    {farmer.lastActive}
                  </td>
                  <td className="px-4 py-4">
                    <span
                      className={`px-2.5 py-1 rounded-full text-xs font-medium ${getHealthScoreColor(
                        farmer.healthScore
                      )}`}
                    >
                      {farmer.healthScore}%
                    </span>
                  </td>
                  <td className="px-4 py-4 text-right">
                    <Button variant="ghost" size="sm">
                      View
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="mt-4 flex items-center justify-between">
          <p className="text-sm text-smartel-gray-500">
            Showing 5 of 243 registered Rwandan farmers
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

export default Farmers;
