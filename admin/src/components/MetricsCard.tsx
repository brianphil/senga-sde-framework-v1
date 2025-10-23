// src/components/MetricsCard.tsx

import React from "react";
import { LucideIcon } from "lucide-react";

interface MetricsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: "good" | "warning" | "normal";
  loading?: boolean;
}

export default function MetricsCard({
  title,
  value,
  subtitle,
  icon,
  trend = "normal",
  loading = false,
}: MetricsCardProps) {
  const trendColors = {
    good: "text-green-600",
    warning: "text-yellow-600",
    normal: "text-gray-600",
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-2">{title}</p>
          {loading ? (
            <div className="h-8 w-24 bg-gray-200 animate-pulse rounded" />
          ) : (
            <p className="text-3xl font-bold text-gray-900">{value}</p>
          )}
          {subtitle && (
            <p className={`text-sm mt-2 ${trendColors[trend]}`}>{subtitle}</p>
          )}
        </div>
        {icon && (
          <div className="ml-4 text-gray-400">
            {React.cloneElement(icon as React.ReactElement, { size: 24 })}
          </div>
        )}
      </div>
    </div>
  );
}
