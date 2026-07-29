import React, { useEffect, useState } from "react";

export interface NodeData {
  name: string;
  role: "main" | "surrounding";
  alias: string;
  lat: number;
  lon: number;
  elevation_m: number;
  metrics?: {
    temperature_2m?: number;
    relative_humidity_2m?: number;
    wind_speed_10m?: number;
  };
}

export interface StarTopographyMapProps {
  nodes?: NodeData[];
  selectedNode?: string;
  onSelectNode?: (name: string) => void;
  width?: string | number;
  height?: string | number;
}

export const CANONICAL_NODES: NodeData[] = [
  {
    name: "MAIN",
    role: "main",
    alias: "Puncak Pangrango",
    lat: -6.75,
    lon: 107.00,
    elevation_m: 1529.0,
    metrics: { temperature_2m: 18.5, relative_humidity_2m: 85.0, wind_speed_10m: 12.4 },
  },
  {
    name: "UP",
    role: "surrounding",
    alias: "Lereng Utara (Up)",
    lat: -6.50,
    lon: 107.00,
    elevation_m: 162.0,
    metrics: { temperature_2m: 26.2, relative_humidity_2m: 72.0, wind_speed_10m: 8.1 },
  },
  {
    name: "DOWN",
    role: "surrounding",
    alias: "Lereng Selatan (Down)",
    lat: -7.00,
    lon: 107.00,
    elevation_m: 0.0,
    metrics: { temperature_2m: 27.5, relative_humidity_2m: 78.0, wind_speed_10m: 9.3 },
  },
  {
    name: "LEFT",
    role: "surrounding",
    alias: "Lereng Barat (Left)",
    lat: -6.75,
    lon: 106.75,
    elevation_m: 823.0,
    metrics: { temperature_2m: 22.1, relative_humidity_2m: 80.0, wind_speed_10m: 10.5 },
  },
  {
    name: "RIGHT",
    role: "surrounding",
    alias: "Lereng Timur (Right)",
    lat: -6.75,
    lon: 107.25,
    elevation_m: 288.0,
    metrics: { temperature_2m: 25.0, relative_humidity_2m: 74.0, wind_speed_10m: 7.8 },
  },
];

export const STAR_EDGES: Array<[string, string]> = [
  ["MAIN", "UP"],
  ["UP", "MAIN"],
  ["MAIN", "DOWN"],
  ["DOWN", "MAIN"],
  ["MAIN", "LEFT"],
  ["LEFT", "MAIN"],
  ["MAIN", "RIGHT"],
  ["RIGHT", "MAIN"],
];

export const StarTopographyMap: React.FC<StarTopographyMapProps> = ({
  nodes = CANONICAL_NODES,
  selectedNode,
  onSelectNode,
  width = "100%",
  height = "500px",
}) => {
  const [activeNode, setActiveNode] = useState<NodeData | null>(
    nodes.find((n) => n.name === selectedNode) || nodes[0]
  );

  useEffect(() => {
    if (selectedNode) {
      const match = nodes.find((n) => n.name === selectedNode);
      if (match) setActiveNode(match);
    }
  }, [selectedNode, nodes]);

  const handleMarkerClick = (node: NodeData) => {
    setActiveNode(node);
    if (onSelectNode) onSelectNode(node.name);
  };

  const mainNode = nodes.find((n) => n.name === "MAIN") || nodes[0];
  const surroundingNodes = nodes.filter((n) => n.name !== "MAIN");

  return (
    <div
      style={{
        position: "relative",
        width,
        height,
        backgroundColor: "#1a1d24",
        borderRadius: "8px",
        overflow: "hidden",
        fontFamily: "sans-serif",
        color: "#f0f4f8",
        border: "1px solid #2d3748",
      }}
    >
      {/* SVG Canvas for Star Topology & Directed Edges */}
      <svg
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
        }}
      >
        <defs>
          <marker
            id="arrowhead-in"
            markerWidth="8"
            markerHeight="8"
            refX="6"
            refY="4"
            orient="auto"
          >
            <polygon points="0 1, 8 4, 0 7" fill="#3182ce" />
          </marker>
          <marker
            id="arrowhead-out"
            markerWidth="8"
            markerHeight="8"
            refX="6"
            refY="4"
            orient="auto"
          >
            <polygon points="0 1, 8 4, 0 7" fill="#e53e3e" />
          </marker>
        </defs>

        {/* Directed Edges Rendering (Schematic Star Layout Projection) */}
        {surroundingNodes.map((node) => {
          // Layout coords mapping (MAIN at center)
          const posMap: Record<string, { x: string; y: string }> = {
            UP: { x: "50%", y: "18%" },
            DOWN: { x: "50%", y: "82%" },
            LEFT: { x: "18%", y: "50%" },
            RIGHT: { x: "82%", y: "50%" },
          };
          const targetPos = posMap[node.name] || { x: "50%", y: "50%" };

          return (
            <g key={`edge-group-${node.name}`}>
              {/* Inbound Line (Surrounding -> MAIN) */}
              <line
                x1={targetPos.x}
                y1={targetPos.y}
                x2="50%"
                y2="50%"
                stroke="#3182ce"
                strokeWidth="2"
                strokeDasharray="4 2"
                markerEnd="url(#arrowhead-in)"
              />
              {/* Outbound Line (MAIN -> Surrounding) */}
              <line
                x1="50%"
                y1="50%"
                x2={targetPos.x}
                y2={targetPos.y}
                stroke="#e53e3e"
                strokeWidth="1.5"
                strokeOpacity="0.7"
                markerEnd="url(#arrowhead-out)"
              />
            </g>
          );
        })}
      </svg>

      {/* Nodes / Markers Rendering */}
      {nodes.map((node) => {
        const isMain = node.name === "MAIN";
        const isActive = activeNode?.name === node.name;

        const posMap: Record<string, { top: string; left: string }> = {
          MAIN: { top: "50%", left: "50%" },
          UP: { top: "18%", left: "50%" },
          DOWN: { top: "82%", left: "50%" },
          LEFT: { top: "50%", left: "18%" },
          RIGHT: { top: "50%", left: "82%" },
        };

        const pos = posMap[node.name] || { top: "50%", left: "50%" };

        return (
          <div
            key={node.name}
            onClick={() => handleMarkerClick(node)}
            style={{
              position: "absolute",
              top: pos.top,
              left: pos.left,
              transform: "translate(-50%, -50%)",
              cursor: "pointer",
              zIndex: isActive ? 10 : 2,
            }}
          >
            <div
              style={{
                width: isMain ? "24px" : "18px",
                height: isMain ? "24px" : "18px",
                borderRadius: "50%",
                backgroundColor: isMain ? "#dd6b20" : "#3182ce",
                border: isActive ? "3px solid #ffffff" : "2px solid #1a202c",
                boxShadow: isActive
                  ? "0 0 12px rgba(255,255,255,0.8)"
                  : "0 2px 4px rgba(0,0,0,0.5)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "10px",
                fontWeight: "bold",
                color: "#fff",
              }}
            >
              {node.name[0]}
            </div>
            <div
              style={{
                marginTop: "4px",
                fontSize: "11px",
                fontWeight: isMain ? "bold" : "normal",
                color: isMain ? "#f6ad55" : "#cbd5e0",
                textAlign: "center",
                whiteSpace: "nowrap",
                textShadow: "0 1px 2px #000",
              }}
            >
              {node.name} ({node.elevation_m}m)
            </div>
          </div>
        );
      })}

      {/* Interactive Tooltip / Popover Panel */}
      {activeNode && (
        <div
          style={{
            position: "absolute",
            bottom: "16px",
            left: "16px",
            backgroundColor: "rgba(26, 32, 44, 0.92)",
            border: "1px solid #4a5568",
            borderRadius: "6px",
            padding: "12px",
            minWidth: "220px",
            backdropFilter: "blur(4px)",
            boxShadow: "0 4px 6px rgba(0,0,0,0.3)",
            zIndex: 20,
          }}
        >
          <div
            style={{
              fontSize: "14px",
              fontWeight: "bold",
              color: activeNode.role === "main" ? "#ed8936" : "#63b3ed",
              marginBottom: "4px",
            }}
          >
            {activeNode.name} - {activeNode.alias}
          </div>
          <div style={{ fontSize: "11px", color: "#a0aec0", marginBottom: "8px" }}>
            Lat: {activeNode.lat}° | Lon: {activeNode.lon}° | Elev: {activeNode.elevation_m}m
          </div>
          <div style={{ fontSize: "12px", display: "grid", gap: "4px" }}>
            <div>
              <span style={{ color: "#cbd5e0" }}>Suhu: </span>
              <strong>{activeNode.metrics?.temperature_2m ?? "-"} °C</strong>
            </div>
            <div>
              <span style={{ color: "#cbd5e0" }}>Kelembapan: </span>
              <strong>{activeNode.metrics?.relative_humidity_2m ?? "-"} %</strong>
            </div>
            <div>
              <span style={{ color: "#cbd5e0" }}>Angin: </span>
              <strong>{activeNode.metrics?.wind_speed_10m ?? "-"} m/s</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StarTopographyMap;
