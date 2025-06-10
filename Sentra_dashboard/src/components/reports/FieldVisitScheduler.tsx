
import React from "react";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Card, CardContent } from "@/components/ui/card";
import { MapPin, Calendar as CalendarIcon, User, Plus } from "lucide-react";

// Rwanda-specific field visit data
const upcomingVisits = [
  {
    id: "V001",
    farmerName: "Jean Mugabo",
    location: "Musanze, Northern Province",
    date: "2023-05-01",
    time: "09:00 AM",
    purpose: "Irish potato blight prevention training",
  },
  {
    id: "V002",
    farmerName: "Marie Uwimana",
    location: "Nyagatare, Eastern Province",
    date: "2023-05-02",
    time: "10:30 AM",
    purpose: "Follow-up on maize disease control measures",
  },
  {
    id: "V003",
    farmerName: "Emmanuel Niyonzima",
    location: "Huye, Southern Province",
    date: "2023-05-03",
    time: "02:00 PM",
    purpose: "Coffee rust inspection and soil sampling",
  },
];

export const FieldVisitScheduler = () => {
  const [date, setDate] = React.useState<Date | undefined>(new Date());

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <Card className="glass-card border-white/30 md:col-span-1">
        <CardContent className="p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-medium">Select Date</h3>
            <Button size="sm" variant="ghost">Today</Button>
          </div>
          <Calendar
            mode="single"
            selected={date}
            onSelect={setDate}
            className="border border-white/20 rounded-md bg-white/10"
          />
          <div className="mt-4">
            <Button className="w-full gap-2 bg-smartel-green-500 hover:bg-smartel-green-600">
              <Plus className="h-4 w-4" />
              Schedule New Visit
            </Button>
          </div>
        </CardContent>
      </Card>
      
      <div className="md:col-span-2 space-y-4">
        <h3 className="text-lg font-medium">Upcoming Field Visits in Rwanda</h3>
        {upcomingVisits.map((visit) => (
          <Card key={visit.id} className="glass-card border-white/30 hover:bg-white/10 transition-colors">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium flex items-center gap-2">
                    <User className="h-4 w-4" />
                    {visit.farmerName}
                  </h4>
                  <p className="text-sm text-muted-foreground flex items-center gap-2 mt-1">
                    <MapPin className="h-4 w-4" />
                    {visit.location}
                  </p>
                </div>
                <div className="text-right">
                  <p className="flex items-center gap-1 text-smartel-teal-500 font-medium">
                    <CalendarIcon className="h-4 w-4" />
                    {visit.date} at {visit.time}
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">{visit.purpose}</p>
                </div>
              </div>
              <div className="flex justify-end mt-4 gap-2">
                <Button variant="outline" size="sm">Reschedule</Button>
                <Button variant="default" size="sm" className="bg-smartel-teal-500 hover:bg-smartel-teal-600">
                  View Details
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};
