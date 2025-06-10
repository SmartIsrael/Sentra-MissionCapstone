import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions, RefreshControl } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { GlassCard } from '@/components/GlassCard';
import { SimpleChart } from '@/components/SimpleChart';
import { TrendingUp, TrendingDown, Calendar, Activity, Droplets, Thermometer, Sprout, ChartBar as BarChart3 } from 'lucide-react-native';

const { width } = Dimensions.get('window');

export default function AnalyticsScreen() {
  const [selectedPeriod, setSelectedPeriod] = useState('week');
  const [refreshing, setRefreshing] = useState(false);

  const periods = [
    { id: 'week', label: 'Week' },
    { id: 'month', label: 'Month' },
    { id: 'quarter', label: 'Quarter' },
    { id: 'year', label: 'Year' },
  ];

  const metrics = [
    {
      title: 'Crop Health Score',
      value: '94%',
      change: '+6.2%',
      trend: 'up' as const,
      icon: Sprout,
      color: '#10B981',
    },
    {
      title: 'Pest Detection Rate',
      value: '1.8%',
      change: '-1.2%',
      trend: 'down' as const,
      icon: Activity,
      color: '#EF4444',
    },
    {
      title: 'Moisture Efficiency',
      value: '89%',
      change: '+4.1%',
      trend: 'up' as const,
      icon: Droplets,
      color: '#06B6D4',
    },
    {
      title: 'Temperature Stability',
      value: '96%',
      change: '+2.5%',
      trend: 'up' as const,
      icon: Thermometer,
      color: '#F59E0B',
    },
  ];

  const chartData = [
    { label: 'Mon', value: 90 },
    { label: 'Tue', value: 92 },
    { label: 'Wed', value: 89 },
    { label: 'Thu', value: 95 },
    { label: 'Fri', value: 93 },
    { label: 'Sat', value: 96 },
    { label: 'Sun', value: 94 },
  ];

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    setTimeout(() => {
      setRefreshing(false);
    }, 2000);
  }, []);

  return (
    <LinearGradient
      colors={['#059669', '#10B981']}
      style={styles.container}
    >
      <SafeAreaView style={styles.safeArea}>
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.title}>Analytics & Insights</Text>
            <Text style={styles.subtitle}>Data-driven farm management</Text>
          </View>
          <TouchableOpacity style={styles.calendarButton}>
            <Calendar size={24} color="#FFFFFF" />
          </TouchableOpacity>
        </View>

        {/* Period Selector */}
        <ScrollView 
          horizontal 
          showsHorizontalScrollIndicator={false}
          style={styles.periodContainer}
          contentContainerStyle={styles.periodContent}
        >
          {periods.map((period) => (
            <TouchableOpacity
              key={period.id}
              style={[
                styles.periodTab,
                selectedPeriod === period.id && styles.activePeriodTab,
              ]}
              onPress={() => setSelectedPeriod(period.id)}
            >
              <Text
                style={[
                  styles.periodText,
                  selectedPeriod === period.id && styles.activePeriodText,
                ]}
              >
                {period.label}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
          }
        >
          {/* Key Metrics */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Key Metrics</Text>
            <View style={styles.metricsGrid}>
              {metrics.map((metric, index) => (
                <GlassCard 
                  key={index} 
                  style={styles.metricCard}
                  onPress={() => console.log(`${metric.title} pressed`)}
                  pressable
                >
                  <View style={styles.metricHeader}>
                    <View style={[styles.metricIconContainer, { backgroundColor: `${metric.color}15` }]}>
                      <metric.icon size={20} color={metric.color} />
                    </View>
                    <View style={[styles.trendIcon, { backgroundColor: metric.trend === 'up' ? '#10B981' : '#EF4444' }]}>
                      {metric.trend === 'up' ? (
                        <TrendingUp size={12} color="#FFFFFF" />
                      ) : (
                        <TrendingDown size={12} color="#FFFFFF" />
                      )}
                    </View>
                  </View>
                  <Text style={styles.metricValue}>{metric.value}</Text>
                  <Text style={styles.metricTitle}>{metric.title}</Text>
                  <Text style={[styles.metricChange, { color: metric.trend === 'up' ? '#10B981' : '#EF4444' }]}>
                    {metric.change} from last {selectedPeriod}
                  </Text>
                </GlassCard>
              ))}
            </View>
          </View>

          {/* Chart */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Performance Overview</Text>
            <GlassCard onPress={() => console.log('Chart pressed')} pressable>
              <SimpleChart 
                data={chartData}
                title="Crop Health Trend"
                color="#10B981"
              />
            </GlassCard>
          </View>

          {/* Insights */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>AI Insights</Text>
            
            <GlassCard 
              style={styles.insightCard}
              onPress={() => console.log('Crop Health insight pressed')}
              pressable
              variant="flat"
            >
              <View style={styles.insightHeader}>
                <View style={[styles.insightIcon, { backgroundColor: '#10B98115' }]}>
                  <TrendingUp size={20} color="#10B981" />
                </View>
                <Text style={styles.insightTitle}>Crop Health Improving</Text>
              </View>
              <Text style={styles.insightText}>
                Your crop health score has increased by 6.2% this week. The recent pest control measures and optimized irrigation schedule are showing positive results.
              </Text>
            </GlassCard>

            <GlassCard 
              style={styles.insightCard}
              onPress={() => console.log('Irrigation insight pressed')}
              pressable
              variant="flat"
            >
              <View style={styles.insightHeader}>
                <View style={[styles.insightIcon, { backgroundColor: '#06B6D415' }]}>
                  <Droplets size={20} color="#06B6D4" />
                </View>
                <Text style={styles.insightTitle}>Irrigation Optimization</Text>
              </View>
              <Text style={styles.insightText}>
                Consider reducing irrigation frequency by 10% in the north field. Current moisture levels are 15% above optimal, which could lead to root problems.
              </Text>
            </GlassCard>

            <GlassCard 
              style={styles.insightCard}
              onPress={() => console.log('Pest Prevention insight pressed')}
              pressable
              variant="flat"
            >
              <View style={styles.insightHeader}>
                <View style={[styles.insightIcon, { backgroundColor: '#F59E0B15' }]}>
                  <Activity size={20} color="#F59E0B" />
                </View>
                <Text style={styles.insightTitle}>Pest Prevention</Text>
              </View>
              <Text style={styles.insightText}>
                Weather conditions next week favor pest activity. Increase monitoring frequency and consider preventive treatments for vulnerable crops.
              </Text>
            </GlassCard>
          </View>

          {/* Bottom Padding for Tab Bar */}
          <View style={styles.bottomPadding} />
        </ScrollView>
      </SafeAreaView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingHorizontal: 20,
    marginTop: 20,
    marginBottom: 20,
  },
  title: {
    fontSize: 28,
    fontFamily: 'Inter-Bold',
    color: '#FFFFFF',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: 'rgba(255, 255, 255, 0.9)',
  },
  calendarButton: {
    padding: 8,
  },
  periodContainer: {
    marginBottom: 16,
  },
  periodContent: {
    paddingHorizontal: 20,
  },
  periodTab: {
    width: 80,
    height: 80,
    borderRadius: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    marginRight: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  activePeriodTab: {
    backgroundColor: '#FFFFFF',
    borderColor: '#FFFFFF',
  },
  periodText: {
    fontSize: 14,
    fontFamily: 'Inter-Medium',
    color: 'rgba(255, 255, 255, 0.9)',
    textAlign: 'center',
  },
  activePeriodText: {
    color: '#059669',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 20,
  },
  section: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 20,
    fontFamily: 'Inter-Bold',
    color: '#FFFFFF',
    marginBottom: 16,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  metricCard: {
    width: (width - 60) / 2,
    marginBottom: 8,
    minHeight: 110,
  },
  metricHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  metricIconContainer: {
    width: 32,
    height: 32,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  trendIcon: {
    width: 20,
    height: 20,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  metricValue: {
    fontSize: 20,
    fontFamily: 'Inter-Bold',
    color: '#1F2937',
    marginBottom: 2,
  },
  metricTitle: {
    fontSize: 13,
    fontFamily: 'Inter-Medium',
    color: '#6B7280',
    marginBottom: 2,
  },
  metricChange: {
    fontSize: 11,
    fontFamily: 'Inter-Regular',
  },
  insightCard: {
    marginBottom: 8,
  },
  insightHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  insightIcon: {
    width: 28,
    height: 28,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  insightTitle: {
    fontSize: 15,
    fontFamily: 'Inter-SemiBold',
    color: '#1F2937',
  },
  insightText: {
    fontSize: 13,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    lineHeight: 18,
  },
  bottomPadding: {
    height: 120,
  },
});