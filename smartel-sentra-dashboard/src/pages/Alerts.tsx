
import React from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Bell, Clock, ChevronDown, Search, X } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface Alert {
  id: string;
  type: "pest" | "disease" | "weather" | "nutrition";
  severity: "high" | "medium" | "low";
  status: "new" | "acknowledged" | "resolved";
  message: string;
  location: string;
  farmer: string;
  detectedAt: string;
  resolvedAt?: string;
}

const mockAlerts: Alert[] = [
  {
    id: "a1",
    type: "pest",
    severity: "high",
    status: "new",
    message: "Fall armyworm infestation detected",
    location: "Farm 12, Sector 3",
    farmer: "James Mwangi",
    detectedAt: "April 29, 2025 - 08:23 AM",
  },
  {
    id: "a2",
    type: "disease",
    severity: "medium",
    status: "acknowledged",
    message: "Early signs of leaf blight detected",
    location: "Farm 5, Sector 2",
    farmer: "Amina Kimani",
    detectedAt: "April 28, 2025 - 03:15 PM",
  },
  {
    id: "a3",
    type: "nutrition",
    severity: "low",
    status: "resolved",
    message: "Nitrogen deficiency detected",
    location: "Farm 8, Sector 1",
    farmer: "David Ochieng",
    detectedAt: "April 27, 2025 - 11:42 AM",
    resolvedAt: "April 28, 2025 - 02:30 PM",
  },
  {
    id: "a4",
    type: "weather",
    severity: "high",
    status: "new",
    message: "Heavy rainfall predicted in next 24 hours",
    location: "All farms in Sector 4",
    farmer: "Multiple farmers",
    detectedAt: "April 29, 2025 - 07:30 AM",
  },
  {
    id: "a5",
    type: "pest",
    severity: "medium",
    status: "acknowledged",
    message: "Aphid population increasing",
    location: "Farm 3, Sector 1",
    farmer: "Sarah Wanjiku",
    detectedAt: "April 28, 2025 - 09:10 AM",
  },
];

const getSeverityColor = (severity: Alert["severity"]) => {
  switch (severity) {
    case "high":
      return "bg-red-100 text-red-700 border-red-300";
    case "medium":
      return "bg-amber-100 text-amber-700 border-amber-300";
    case "low":
      return "bg-blue-100 text-blue-700 border-blue-300";
    default:
      return "bg-gray-100 text-gray-700 border-gray-300";
  }
};

const getStatusColor = (status: Alert["status"]) => {
  switch (status) {
    case "new":
      return "bg-smartel-teal-100 text-smartel-teal-700 border-smartel-teal-300";
    case "acknowledged":
      return "bg-purple-100 text-purple-700 border-purple-300";
    case "resolved":
      return "bg-green-100 text-green-700 border-green-300";
    default:
      return "bg-gray-100 text-gray-700 border-gray-300";
  }
};

const getTypeIcon = (type: Alert["type"]) => {
  switch (type) {
    case "pest":
      return "🐛";
    case "disease":
      return "🦠";
    case "weather":
      return "🌧️";
    case "nutrition":
      return "🌱";
    default:
      return "❓";
  }
};

const Alerts = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient">Alerts</h1>
          <p className="mt-1 text-smartel-gray-500">
            Manage and respond to detected issues across all monitored farms.
          </p>
        </div>
        
        <div className="flex gap-2">
          <Button variant="outline" className="bg-white/60 border-white/30">
            <Bell className="h-4 w-4 mr-2" />
            Subscribe
          </Button>
          <Button variant="outline" className="bg-white/60 border-white/30">
            <Clock className="h-4 w-4 mr-2" />
            History
          </Button>
        </div>
      </div>
      
      <div className="glass-card p-6">
        <div className="flex flex-col md:flex-row justify-between gap-4 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-smartel-gray-500" />
            <input
              type="text"
              placeholder="Search alerts..."
              className="pl-9 pr-4 py-2 w-full rounded-lg text-sm bg-white/60 border border-white/30 focus:outline-none focus:ring-2 focus:ring-smartel-green-400"
            />
          </div>
          
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className="bg-white/60 border-white/30 text-smartel-gray-700 px-3 py-1 flex items-center gap-2">
              Type: All
              <X className="h-3 w-3" />
            </Badge>
            <Badge variant="outline" className="bg-white/60 border-white/30 text-smartel-gray-700 px-3 py-1 flex items-center gap-2">
              Status: New
              <X className="h-3 w-3" />
            </Badge>
            <Button variant="outline" className="bg-white/60 border-white/30 h-7 px-3">
              <ChevronDown className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        <div className="space-y-4">
          {mockAlerts.map((alert) => (
            <div
              key={alert.id}
              className="p-4 rounded-lg bg-white/60 border border-white/30 hover:bg-white/80 transition-colors"
            >
              <div className="grid grid-cols-12 gap-4">
                <div className="col-span-12 sm:col-span-7">
                  <div className="flex items-start gap-3">
                    <div className="h-10 w-10 flex-shrink-0 flex items-center justify-center rounded-full bg-smartel-green-100 text-lg">
                      {getTypeIcon(alert.type)}
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">{alert.message}</h4>
                      <div className="flex flex-wrap gap-2 mb-2">
                        <span
                          className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(
                            alert.severity
                          )}`}
                        >
                          {alert.severity} severity
                        </span>
                        <span
                          className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                            alert.status
                          )}`}
                        >
                          {alert.status}
                        </span>
                      </div>
                      <div className="text-sm text-smartel-gray-500">
                        <div>Location: {alert.location}</div>
                        <div>Farmer: {alert.farmer}</div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="col-span-12 sm:col-span-5">
                  <div className="h-full flex flex-col justify-between">
                    <div className="space-y-2">
                      <div className="text-xs text-smartel-gray-500">
                        Detected: {alert.detectedAt}
                      </div>
                      {alert.resolvedAt && (
                        <div className="text-xs text-smartel-gray-500">
                          Resolved: {alert.resolvedAt}
                        </div>
                      )}
                      
                      {!alert.resolvedAt && (
                        <div className="pt-1">
                          <div className="flex justify-between items-center text-xs text-smartel-gray-500 mb-1">
                            <span>Resolution progress</span>
                            <span>
                              {alert.status === "new"
                                ? "0%"
                                : alert.status === "acknowledged"
                                ? "50%"
                                : "100%"}
                            </span>
                          </div>
                          <Progress
                            value={
                              alert.status === "new"
                                ? 0
                                : alert.status === "acknowledged"
                                ? 50
                                : 100
                            }
                            className="h-1.5"
                          />
                        </div>
                      )}
                    </div>
                    
                    <div className="flex justify-end gap-2 mt-3">
                      {alert.status === "new" && (
                        <Button size="sm" variant="outline" className="bg-white/60 border-white/30">
                          Acknowledge
                        </Button>
                      )}
                      {alert.status !== "resolved" && (
                        <Button size="sm" variant="default" className="bg-smartel-green-500 hover:bg-smartel-green-600">
                          {alert.status === "acknowledged" ? "Resolve" : "View Details"}
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-6 flex items-center justify-between">
          <p className="text-sm text-smartel-gray-500">
            Showing 5 of 17 alerts
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

export default Alerts;
