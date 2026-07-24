import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { NodeId, TimelineFrame } from '../types';
import { NODES_LIST } from '../data/nodesData';
import { Info, Navigation } from 'lucide-react';

interface MapViewProps {
  selectedNodeId: NodeId;
  onSelectNode: (nodeId: NodeId) => void;
  currentFrame: TimelineFrame;
  radarVisible: boolean;
  nodesVisible: boolean;
  onOpenNodeDetail: (nodeId: NodeId) => void;
}

const GRID_HALF_STEP = 0.125;

function precipitationStyle(rain: number): L.PathOptions {
  if (rain >= 10) return { color: '#b96a4b', fillColor: '#d58a68', fillOpacity: 0.18, weight: 1 };
  if (rain >= 2) return { color: '#bc8a35', fillColor: '#e0bd78', fillOpacity: 0.16, weight: 1 };
  return { color: '#4d897c', fillColor: '#9bc7bb', fillOpacity: 0.14, weight: 1 };
}

const MapViewComponent: React.FC<MapViewProps> = ({
  selectedNodeId,
  onSelectNode,
  currentFrame,
  radarVisible,
  nodesVisible,
  onOpenNodeDetail,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const vectorRendererRef = useRef<L.Canvas | null>(null);
  const markersRef = useRef<Record<string, L.Marker>>({});
  const polyLinesRef = useRef<Record<string, L.Polyline>>({});
  const cellsRef = useRef<Record<string, L.Rectangle>>({});
  const onSelectNodeRef = useRef(onSelectNode);
  const [hoveredNodeId, setHoveredNodeId] = useState<NodeId | null>(null);


  useEffect(() => {
    onSelectNodeRef.current = onSelectNode;
  }, [onSelectNode]);

  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [-6.75, 107.0],
      zoom: 10,
      minZoom: 9,
      maxZoom: 14,
      maxBounds: [[-7.2, 106.45], [-6.3, 107.55]],
      maxBoundsViscosity: 0.9,
      zoomControl: false,
      attributionControl: true,
      preferCanvas: true,
      zoomAnimation: true,
      zoomAnimationThreshold: 4,
      wheelDebounceTime: 35,
      wheelPxPerZoomLevel: 90,
    });

    L.control.zoom({ position: 'topright' }).addTo(map);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap',
      maxZoom: 19,
      subdomains: 'abcd',
      detectRetina: true,
    }).addTo(map);

    // One viewport of overscan prevents ERA5 cells from clipping during a drag.
    vectorRendererRef.current = L.canvas({ padding: 1 });
    mapInstanceRef.current = map;
    const resizeObserver = new ResizeObserver(() => map.invalidateSize());
    resizeObserver.observe(mapContainerRef.current);

    return () => {
      resizeObserver.disconnect();
      (Object.values(polyLinesRef.current) as L.Polyline[]).forEach((line) => line.remove());
      (Object.values(markersRef.current) as L.Marker[]).forEach((marker) => marker.remove());
      (Object.values(cellsRef.current) as L.Rectangle[]).forEach((cell) => cell.remove());
      polyLinesRef.current = {};
      markersRef.current = {};
      cellsRef.current = {};
      vectorRendererRef.current = null;
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (!radarVisible) {
      (Object.values(cellsRef.current) as L.Rectangle[]).forEach((cell) => cell.remove());
      cellsRef.current = {};
      return;
    }

    Object.values(NODES_LIST).forEach((node) => {
      const rain = currentFrame.telemetries[node.id]?.rainIntensityMmH ?? 0;
      const bounds: L.LatLngBoundsExpression = [
        [node.lat - GRID_HALF_STEP, node.lng - GRID_HALF_STEP],
        [node.lat + GRID_HALF_STEP, node.lng + GRID_HALF_STEP],
      ];
      const label = `<strong>${node.id}</strong><br>${rain.toFixed(1)} mm/jam<br><small>ERA5 0.25°</small>`;
      const existingCell = cellsRef.current[node.id];

      if (existingCell) {
        existingCell.setBounds(bounds);
        existingCell.setStyle(precipitationStyle(rain));
        existingCell.setTooltipContent(label);
      } else {
        const cell = L.rectangle(bounds, { ...precipitationStyle(rain), renderer: vectorRendererRef.current ?? undefined }).addTo(map);
        cell.bindTooltip(label, { className: 'era5-cell-tooltip', sticky: true });
        cellsRef.current[node.id] = cell;
      }
    });
  }, [radarVisible, currentFrame]);

  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (!nodesVisible) {
      (Object.values(polyLinesRef.current) as L.Polyline[]).forEach((line) => line.remove());
      polyLinesRef.current = {};
      return;
    }

    (['UP', 'DOWN', 'LEFT', 'RIGHT'] as NodeId[]).forEach((id) => {
      const sourceNode = NODES_LIST[id];
      const mainNode = NODES_LIST.MAIN;
      const weight = currentFrame.telemetries[id]?.spatialWeightToMain ?? 0.25;
      const latLngs: [number, number][] = [[sourceNode.lat, sourceNode.lng], [mainNode.lat, mainNode.lng]];
      const existingLine = polyLinesRef.current[id];

      if (existingLine) {
        existingLine.setLatLngs(latLngs);
        existingLine.setStyle({ weight: Math.max(1, weight * 3), opacity: 0.38 });
        existingLine.setTooltipContent(`Pengaruh area ${id} ke MAIN: ${(weight * 100).toFixed(0)}%`);
      } else {
        const line = L.polyline(latLngs, {
          color: '#315f57',
          weight: Math.max(1, weight * 3),
          opacity: 0.38,
          dashArray: '4, 8',
          renderer: vectorRendererRef.current ?? undefined,
        }).addTo(map);
        line.bindTooltip(`Pengaruh area ${id} ke MAIN: ${(weight * 100).toFixed(0)}%`, {
          permanent: false,
          direction: 'center',
          className: 'era5-edge-tooltip',
        });
        polyLinesRef.current[id] = line;
      }
    });
  }, [nodesVisible, currentFrame]);

  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (!nodesVisible) {
      (Object.values(markersRef.current) as L.Marker[]).forEach((marker) => marker.remove());
      markersRef.current = {};
      return;
    }

    Object.values(NODES_LIST).forEach((node) => {
      const isSelected = selectedNodeId === node.id;
      const rain = currentFrame.telemetries[node.id]?.rainIntensityMmH ?? 0;
      const markerHtml = `
        <div class="era5-node ${isSelected ? 'is-selected' : ''}">
          <span class="era5-node__dot"></span>
          <span class="era5-node__label"><b>${node.id}</b> ${rain.toFixed(1)} mm/jam</span>
        </div>`;
      const icon = L.divIcon({ html: markerHtml, className: 'era5-node-icon', iconSize: [130, 44], iconAnchor: [65, 11] });
      const existingMarker = markersRef.current[node.id];

      if (existingMarker) {
        existingMarker.setIcon(icon);
        existingMarker.setLatLng([node.lat, node.lng]);
        existingMarker.setZIndexOffset(isSelected ? 1000 : 100);
      } else {
        const marker = L.marker([node.lat, node.lng], { icon, zIndexOffset: isSelected ? 1000 : 100 }).addTo(map);
        marker.on('click', () => onSelectNodeRef.current(node.id));
        marker.on('mouseover', () => setHoveredNodeId(node.id));
        marker.on('mouseout', () => setHoveredNodeId(null));
        markersRef.current[node.id] = marker;
      }
    });
  }, [nodesVisible, selectedNodeId, currentFrame]);

  const activeNode = hoveredNodeId ? NODES_LIST[hoveredNodeId] : null;
  const activeTelemetry = hoveredNodeId ? currentFrame.telemetries[hoveredNodeId] : null;

  return (
    <div role="region" aria-label="Peta kondisi cuaca lima area di sekitar Pangrango" className="era5-map relative w-full h-full min-h-[420px] border border-[#c9cec4] bg-[#dfe6dd]">
      <div ref={mapContainerRef} className="w-full h-full" />

      <div className="absolute top-4 left-4 right-4 pointer-events-none z-10 flex justify-between items-start gap-4">
        <div className="era5-map-card">
          <p className="era5-map-card__eyebrow">Lima area pemantauan</p>
          <p className="era5-map-card__title">Kondisi hujan di sekitar puncak.</p>
          <p className="era5-map-card__detail">Warna menunjukkan hujan di tiap area. Perkiraan utama untuk area MAIN.</p>
        </div>
        <div className="era5-legend" aria-label="Legenda hujan per area">
          <span>Contoh hujan per area</span>
          <div><i className="is-light" />&lt; 2</div>
          <div><i className="is-medium" />2-10</div>
          <div><i className="is-heavy" />≥ 10 mm/jam</div>
        </div>
      </div>

      {activeNode && activeTelemetry && (
        <div className="era5-hover-card">
          <div className="flex justify-between gap-4 border-b border-[#d8d4c8] pb-2">
            <div>
              <p className="font-mono-data text-[11px] font-semibold text-[#1f6056]">Area {activeNode.id}</p>
              <p className="text-xs text-[#5f685e]">{activeNode.locationName}</p>
            </div>
            <button onClick={() => onOpenNodeDetail(activeNode.id)} aria-label={`Detail ${activeNode.id}`} className="era5-detail-button" title="Detail simpul">
              <Info className="w-4 h-4" />
            </button>
          </div>
          <div className="grid grid-cols-2 gap-x-5 gap-y-1.5 pt-3 font-mono-data text-xs">
            <span className="text-[#5f685e]">Presipitasi</span><span className="text-right font-semibold text-[#202720]">{activeTelemetry.rainIntensityMmH.toFixed(1)} mm/jam</span>
            <span className="text-[#5f685e]">Angin</span><span className="text-right font-semibold text-[#202720]">{activeTelemetry.windSpeedKmh} km/jam</span>
            <span className="text-[#5f685e]">Kelembapan</span><span className="text-right font-semibold text-[#202720]">{activeTelemetry.humidityPercent}%</span>
            <span className="text-[#5f685e]">Pengaruh ke area utama</span><span className="text-right font-semibold text-[#202720]">{(activeTelemetry.spatialWeightToMain * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}
    </div>
  );
};

export const MapView = React.memo(MapViewComponent);
