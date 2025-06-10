
import React from "react";
import { Bell } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Alert {
  id: string;
  type: "pest" | "disease" | "weather" | "nutrition";
  severity: "high" | "medium" | "low";
  message: string;
  location: string;
  time: string;
}

const mockAlerts: Alert[] = [
  {
    id: "a1",
    type: "pest",
    severity: "high",
    message: "Fall armyworm detected in Maize field",
    location: "Sector 3, Farm 12",
    time: "2h ago",
  },
  {
    id: "a2",
    type: "disease",
    severity: "medium",
    message: "Early signs of leaf blight in Rice paddies",
    location: "Sector 2, Farm 5",
    time: "5h ago",
  },
  {
    id: "a3",
    type: "nutrition",
    severity: "low",
    message: "Nitrogen deficiency detected",
    location: "Sector 1, Farm 8",
    time: "1d ago",
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

const AlertsOverview = () => {
  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Bell className="h-5 w-5 text-smartel-green-500" />
          <h3 className="text-lg font-medium">Recent Alerts</h3>
        </div>
        <Button variant="outline" size="sm" className="text-xs bg-white/50 border-white/30">
          View All Alerts
        </Button>
      </div>

      <div className="space-y-4">
        {mockAlerts.map((alert) => (
          <div
            key={alert.id}
            className="flex items-start gap-4 p-3 rounded-lg bg-white/50 hover:bg-white/70 transition-colors border border-white/30"
          >
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-smartel-green-100">
              <span className="text-lg">{getTypeIcon(alert.type)}</span>
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between mb-1">
                <h4 className="font-medium text-smartel-gray-800 truncate">{alert.message}</h4>
                <span
                  className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(
                    alert.severity
                  )}`}
                >
                  {alert.severity}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <p className="text-sm text-smartel-gray-500">{alert.location}</p>
                <p className="text-xs text-smartel-gray-400">{alert.time}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AlertsOverview;
