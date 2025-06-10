
import React from "react";
import { Users, Bug, CloudSun, AlertTriangle } from "lucide-react";
import StatCard from "@/components/dashboard/StatCard";
import AlertsOverview from "@/components/dashboard/AlertsOverview";
import CropHealthMap from "@/components/dashboard/CropHealthMap";
import DeviceStatus from "@/components/dashboard/DeviceStatus";

const Dashboard = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient">Dashboard</h1>
          <p className="mt-1 text-smartel-gray-500">
            Welcome back! Here's an overview of your farming community.
          </p>
        </div>
        <div className="glass-panel px-4 py-2 rounded-lg text-smartel-gray-500">
          <span className="font-medium">Today:</span> April 29, 2025
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
        <StatCard 
          title="Total Farmers" 
          value="243" 
          icon={<Users size={20} />} 
          trend={{ value: 12, isPositive: true }}
        />
        <StatCard 
          title="Active Alerts" 
          value="17" 
          icon={<AlertTriangle size={20} />} 
          trend={{ value: 5, isPositive: false }}
        />
        <StatCard 
          title="Pest Detections" 
          value="8" 
          icon={<Bug size={20} />} 
          trend={{ value: 3, isPositive: false }}
        />
        <StatCard 
          title="Weather Alerts" 
          value="4" 
          icon={<CloudSun size={20} />} 
          trend={{ value: 1, isPositive: true }}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <CropHealthMap />
        </div>
        <div>
          <AlertsOverview />
        </div>
      </div>

      <div>
        <DeviceStatus />
      </div>
    </div>
  );
};

export default Dashboard;
