
import React from "react";
import { Button } from "@/components/ui/button";
import { FileText, Download, Filter } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

// Rwanda-specific report data
const reportData = [
  {
    id: "RW-001",
    farmerName: "Jean Mugabo",
    location: "Musanze, Northern Province",
    cropType: "Irish Potatoes",
    reportType: "Pest Detection",
    date: "2023-04-15",
    status: "Critical",
  },
  {
    id: "RW-002",
    farmerName: "Marie Uwimana",
    location: "Nyagatare, Eastern Province",
    cropType: "Maize",
    reportType: "Disease Analysis",
    date: "2023-04-17",
    status: "Resolved",
  },
  {
    id: "RW-003",
    farmerName: "Emmanuel Niyonzima",
    location: "Huye, Southern Province",
    cropType: "Coffee",
    reportType: "Nutrient Deficiency",
    date: "2023-04-18",
    status: "Pending",
  },
  {
    id: "RW-004",
    farmerName: "Jeanne Mukamana",
    location: "Karongi, Western Province",
    cropType: "Tea",
    reportType: "Water Stress",
    date: "2023-04-19",
    status: "In Progress",
  },
  {
    id: "RW-005",
    farmerName: "Pascal Habimana",
    location: "Gasabo, Kigali",
    cropType: "Vegetables",
    reportType: "Pest Detection",
    date: "2023-04-20",
    status: "Critical",
  },
];

export const FarmerReportsTable = () => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" className="gap-1">
            <Filter className="h-4 w-4" />
            Filter
          </Button>
          <Button size="sm" variant="outline" className="gap-1">
            <Download className="h-4 w-4" />
            Export
          </Button>
        </div>
        <Button variant="default" size="sm" className="gap-1 bg-smartel-green-500 hover:bg-smartel-green-600">
          <FileText className="h-4 w-4" />
          New Report
        </Button>
      </div>
      
      <div className="rounded-md border border-white/30 overflow-hidden">
        <Table>
          <TableHeader className="bg-white/10">
            <TableRow>
              <TableHead>Report ID</TableHead>
              <TableHead>Farmer</TableHead>
              <TableHead>Location</TableHead>
              <TableHead>Crop Type</TableHead>
              <TableHead>Report Type</TableHead>
              <TableHead>Date</TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {reportData.map((report) => (
              <TableRow key={report.id} className="hover:bg-white/10">
                <TableCell className="font-medium">{report.id}</TableCell>
                <TableCell>{report.farmerName}</TableCell>
                <TableCell>{report.location}</TableCell>
                <TableCell>{report.cropType}</TableCell>
                <TableCell>{report.reportType}</TableCell>
                <TableCell>{report.date}</TableCell>
                <TableCell>
                  <span className={`px-2 py-1 text-xs rounded-full inline-block ${
                    report.status === "Critical" ? "bg-red-500/70 text-white" :
                    report.status === "Resolved" ? "bg-green-500/70 text-white" :
                    report.status === "Pending" ? "bg-yellow-500/70" :
                    "bg-blue-500/70 text-white"
                  }`}>
                    {report.status}
                  </span>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
};
