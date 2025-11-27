const map = L.map('map', {
  zoomControl: false
}).setView([37.5, -122.3], 14);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

let telemetry = [];
let cursorMarker = null;
let startFinishMarker = null;
let lapLayers = new Map();
let allLaps = [];
let selectedComparisonLap = null;
let currentSample = null;
let visibleLapNumber = null;
let cornerGroup = L.layerGroup().addTo(map);
let heatmapLayers = {};
let speedColoringEnabled = true;
let lapFeaturesData = null;

// Get dataset from URL parameter
const urlParams = new URLSearchParams(window.location.search);
const dataset = urlParams.get('dataset');
const datasetParam = dataset ? `?dataset=${encodeURIComponent(dataset)}` : '';

fetch(`/api/session${datasetParam}`)
  .then(res => res.json())
  .then(session => {
    telemetry = session.telemetry || [];
    lapFeaturesData = session.lap_features;
    drawTrack(session.track);
    renderPartialLaps(session.partial_lap_features);
    renderLapLanes(session.lap_features);
    renderLapControls(session.lap_features);
    renderLapList(session.laps || []);
    allLaps = session.laps || [];
    renderCornerList(session.corners || []);
    // Setup delta after visible lap is determined (in renderLapLanes)
    setupLapDelta();
    setupHeatmapControls(session.heatmap || {});
    setupExportControls(session.laps || []);
    setupSpeedColoringToggle();
    configurePanelToggle();
    initCursor();
    document.getElementById('status').textContent = telemetry.length
      ? 'Hover over the polyline to inspect telemetry.'
      : 'No telemetry samples were generated.';
  })
  .catch(err => {
    console.error('Error loading session', err);
    document.getElementById('status').textContent = 'Failed to load session payload.';
  });

function drawTrack(trackGeoJson) {
  if (!trackGeoJson) return;
  const layer = L.geoJSON(trackGeoJson, {
    style: feature => ({
      color: feature.properties && feature.properties.marker === 'start_finish'
        ? 'transparent'
        : '#9aa7c2',
      weight: feature.properties && feature.properties.marker === 'start_finish' ? 0 : 2,
      opacity: 0.5,
    }),
    pointToLayer: (feature, latlng) => {
      if (feature.properties && feature.properties.marker === 'start_finish') {
        startFinishMarker = L.circleMarker(latlng, {
          radius: 6,
          color: '#000000',
          fillColor: '#000000',
          fillOpacity: 1
        }).bindTooltip('Start / Finish', { permanent: false });
        // Bring to front to ensure it's on top
        startFinishMarker.addTo(map);
        startFinishMarker.bringToFront();
        return startFinishMarker;
      }
      return L.marker(latlng);
    }
  }).addTo(map);

  map.fitBounds(layer.getBounds(), { padding: [40, 40] });
  // Don't attach mouse events to main track - only visible laps should be interactive
  
  // Ensure start/finish marker is on top
  if (startFinishMarker) {
    startFinishMarker.bringToFront();
  }
}

function configurePanelToggle() {
  const buttons = document.querySelectorAll('#panel-toggle button');
  const panels = document.querySelectorAll('.panel-view');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const panelId = btn.dataset.panel;
      buttons.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(panelId).classList.add('active');
    });
  });
  if (buttons.length) {
    buttons[0].classList.add('active');
    document.getElementById(buttons[0].dataset.panel).classList.add('active');
  }
}

function renderPartialLaps(partialLapCollection) {
  if (!partialLapCollection || !partialLapCollection.features) return;
  
  partialLapCollection.features.forEach((feature) => {
    const coords = feature.geometry.coordinates;
    if (!coords || coords.length < 2) return;
    
    const latlngs = coords.map(([lon, lat]) => [lat, lon]);
    const polyline = L.polyline(latlngs, {
      color: '#555555',  // Dark gray
      weight: 3,
      opacity: 0.7,
      interactive: false,  // Non-interactive
    });
    
    polyline.addTo(map);
  });
  
  // Ensure start/finish marker and cursor stay on top
  if (startFinishMarker) {
    startFinishMarker.bringToFront();
  }
  if (cursorMarker) {
    cursorMarker.bringToFront();
  }
}

function renderLapLanes(lapCollection) {
  if (!lapCollection || !lapCollection.features) return;
  
  // Clear existing lap layers
  lapLayers.forEach((entry) => {
    if (map.hasLayer(entry.layer)) {
      map.removeLayer(entry.layer);
    }
  });
  lapLayers.clear();
  
  lapCollection.features.forEach((feature, idx) => {
    const lapNumber = feature.properties?.lap_number ?? idx + 1;
    const baseColor = feature.properties?.stroke || lapColor(idx);
    const strokeWidth = feature.properties?.stroke_width || 4;
    const segments = feature.properties?.segments || [];
    const segmentColors = feature.properties?.segment_colors || [];

    let layer;
    const isFirstLap = idx === 0;
    const willBeVisible = isFirstLap; // Only first lap visible by default
    
    if (segments.length && speedColoringEnabled) {
      // Use speed-colored segments
      const polylines = segments.map((segment, segIdx) => {
        const color = segmentColors[segIdx] || baseColor;
        const latlngs = segment.map(([lon, lat]) => [lat, lon]);
        const line = L.polyline(latlngs, {
          color,
          weight: strokeWidth,
          opacity: 0.95,
        });
        // Only attach mouse events to visible laps
        if (willBeVisible) {
          line.on('mousemove', event => updateTelemetryAt(event.latlng));
        }
        return line;
      });
      layer = L.layerGroup(polylines);
    } else {
      // Use black color when speed coloring is disabled, or if no segments
      const color = speedColoringEnabled ? baseColor : '#93d5db';
      layer = L.geoJSON(feature, {
        style: {
          color: color,
          weight: strokeWidth,
          opacity: 0.95,
        },
      });

      layer.eachLayer(child => {
        if (child instanceof L.Polyline) {
          // Only attach mouse events to visible laps
          if (willBeVisible) {
            child.on('mousemove', event => updateTelemetryAt(event.latlng));
          }
        }
      });
    }
    
    // Only show the first lap by default
    if (isFirstLap) {
      layer.addTo(map);
      visibleLapNumber = lapNumber;
    }

    lapLayers.set(lapNumber, { layer, color: baseColor });
  });
  
  // Ensure start/finish marker and cursor stay on top
  if (startFinishMarker) {
    startFinishMarker.bringToFront();
  }
  if (cursorMarker) {
    cursorMarker.bringToFront();
  }
}

function setupSpeedColoringToggle() {
  const toggle = document.getElementById('speed-coloring-toggle');
  if (!toggle) return;
  
  toggle.addEventListener('change', (event) => {
    speedColoringEnabled = event.target.checked;
    if (lapFeaturesData) {
      const currentVisibleLap = visibleLapNumber; // Store before render
      renderLapLanes(lapFeaturesData);
      // Restore visibility and mouse events for the previously visible lap
      if (currentVisibleLap !== null) {
        const entry = lapLayers.get(currentVisibleLap);
        if (entry) {
          entry.layer.addTo(map);
          attachMouseEventsToLayer(entry.layer);
          visibleLapNumber = currentVisibleLap;
          // Update button state
          const button = document.querySelector(`button[data-lap="${currentVisibleLap}"]`);
          if (button) {
            button.classList.add('active');
          }
        }
      }
    }
  });
  
  buildSpeedLegend();
}

function buildSpeedLegend() {
  const container = document.getElementById('speed-legend-rows');
  if (!container || !telemetry.length) return;
  
  let maxSpeed = 0;
  for (const sample of telemetry) {
    if (sample.speed_mph != null && sample.speed_mph > maxSpeed) {
      maxSpeed = sample.speed_mph;
    }
  }
  
  maxSpeed = Math.ceil(maxSpeed / 10) * 10;
  if (maxSpeed < 10) maxSpeed = 10;
  
  const bands = [
    { pct: 0, color: '#0088ff', label: `< ${Math.round(maxSpeed * 0.2)} mph` },
    { pct: 0.2, color: '#00ff00', label: `${Math.round(maxSpeed * 0.2)} – ${Math.round(maxSpeed * 0.4)} mph` },
    { pct: 0.4, color: '#ffff00', label: `${Math.round(maxSpeed * 0.4)} – ${Math.round(maxSpeed * 0.6)} mph` },
    { pct: 0.6, color: '#ff8800', label: `${Math.round(maxSpeed * 0.6)} – ${Math.round(maxSpeed * 0.8)} mph` },
    { pct: 0.8, color: '#ff0000', label: `> ${Math.round(maxSpeed * 0.8)} mph` },
  ];
  
  container.innerHTML = bands.map(b => `
    <div class="legend-row">
      <span class="legend-swatch" style="background:${b.color}"></span>
      <span>${b.label}</span>
    </div>
  `).join('');
}

function lapColor(idx) {
  const palette = ['#ff2d55', '#28a2ff', '#ffd60a', '#8af27f', '#b28dff', '#ff8a5b'];
  return palette[idx % palette.length];
}

function initCursor() {
  cursorMarker = L.circleMarker([0, 0], {
    radius: 5,
    color: '#ff2d55',
    fillColor: '#ff2d55',
    weight: 2,
    fillOpacity: 0.9,
  }).addTo(map);
  cursorMarker.setStyle({ opacity: 0, fillOpacity: 0 });
  // Bring cursor to front to ensure it's always on top
  cursorMarker.bringToFront();
}

function updateTelemetryAt(latlng) {
  if (!telemetry.length || !cursorMarker || visibleLapNumber === null) return;
  
  // Only allow cursor to attach to samples from the visible lap
  const sample = findNearestSampleInLap(latlng.lat, latlng.lng, visibleLapNumber);
  if (!sample) {
    // Hide cursor if not over visible lap
    cursorMarker.setStyle({ opacity: 0, fillOpacity: 0 });
    return;
  }

  currentSample = sample; // Store current sample for delta updates

  cursorMarker.setLatLng([sample.lat, sample.lon]);
  cursorMarker.setStyle({ opacity: 1, fillOpacity: 0.9 });
  // Ensure cursor stays on top
  cursorMarker.bringToFront();

  document.getElementById('telemetry-time').textContent = formatNumber(sample.elapsed_s, 2);
  document.getElementById('telemetry-speed').textContent = formatNumber(sample.speed_mph, 1);
  document.getElementById('telemetry-long').textContent = formatNumber(sample.long_accel_mps2, 2);
  document.getElementById('telemetry-lat').textContent = formatNumber(sample.lat_accel_mps2, 2);
  // Convert distance from meters to miles (1 meter = 0.000621371 miles)
  const distance_mi = (sample.distance_m || 0) * 0.000621371;
  document.getElementById('telemetry-distance').textContent = formatNumber(distance_mi, 2);
  const imu = [sample.imu_accel_x, sample.imu_accel_y, sample.imu_accel_z]
    .map(v => formatNumber(v, 1))
    .join(' / ');
  document.getElementById('telemetry-imu').textContent = imu;
  updateDeltaReadout(sample);
}

function findNearestSampleInLap(lat, lon, lapNumber) {
  // Find samples that belong to the specified lap
  const lapSamples = telemetry.filter(sample => 
    sample.lap_index === lapNumber && 
    sample.lat != null && 
    sample.lon != null
  );
  
  if (lapSamples.length === 0) return null;
  
  let minDist = Infinity;
  let nearest = null;
  for (const sample of lapSamples) {
    const dist = haversine(lat, lon, sample.lat, sample.lon);
    if (dist < minDist) {
      minDist = dist;
      nearest = sample;
    }
  }
  return nearest;
}

function findNearestSample(lat, lon) {
  let minDist = Infinity;
  let nearest = null;
  for (const sample of telemetry) {
    if (sample.lat == null || sample.lon == null) continue;
    const dist = haversine(lat, lon, sample.lat, sample.lon);
    if (dist < minDist) {
      minDist = dist;
      nearest = sample;
    }
  }
  return nearest;
}

function haversine(lat1, lon1, lat2, lon2) {
  const toRad = deg => deg * Math.PI / 180;
  const R = 6371000;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return Number(value).toFixed(digits);
}

function renderLapList(laps) {
  const container = document.getElementById('lap-list');
  if (!laps.length) {
    container.innerHTML = '<p>No laps detected yet.</p>';
    return;
  }

  container.innerHTML = laps.map(lap => {
    const sectors = (lap.sector_times_s || [])
      .map((time, idx) => `S${idx + 1}: ${formatNumber(time, 2)}s`)
      .join(' · ');
    return `
      <div class="lap-block">
        <div class="lap-entry">
          <strong>Lap ${lap.lap_number}</strong>
          <span>${formatNumber(lap.lap_time_s, 2)} s</span>
        </div>
        <div class="sector-row">${sectors}</div>
      </div>
    `;
  }).join('');
}

function renderLapControls(lapCollection) {
  const container = document.getElementById('lap-buttons');
  container.innerHTML = '';
  if (!lapCollection || !lapCollection.features || !lapCollection.features.length) {
    container.innerHTML = '<p>No lap overlays detected.</p>';
    return;
  }

  lapCollection.features.forEach((feature, idx) => {
    const lapNumber = feature.properties?.lap_number ?? idx + 1;
    const lapTime = feature.properties?.lap_time_s;
    const color = lapLayers.get(lapNumber)?.color || lapColor(idx);

    const button = document.createElement('button');
    // Only mark the first lap as active by default
    button.className = idx === 0 ? 'lap-toggle active' : 'lap-toggle';
    button.dataset.lap = lapNumber;
    button.innerHTML = `
      <span class="lap-swatch" style="background:${color}"></span>
      Lap ${lapNumber}${lapTime ? ` – ${formatNumber(lapTime, 2)}s` : ''}
    `;
    button.addEventListener('click', () => toggleLapVisibility(lapNumber, button));
    container.appendChild(button);
  });
}

function toggleLapVisibility(lapNumber, button) {
  const entry = lapLayers.get(lapNumber);
  if (!entry) return;
  
  // Count how many laps are currently visible
  let visibleCount = 0;
  lapLayers.forEach((otherEntry) => {
    if (map.hasLayer(otherEntry.layer)) {
      visibleCount++;
    }
  });
  
  // If this lap is already visible, prevent hiding if it's the last visible lap
  if (map.hasLayer(entry.layer)) {
    if (visibleCount <= 1) {
      // Don't allow hiding the last visible lap
      return;
    }
    map.removeLayer(entry.layer);
    button.classList.remove('active');
    visibleLapNumber = null;
    // Find the next visible lap
    lapLayers.forEach((otherEntry, otherLapNumber) => {
      if (map.hasLayer(otherEntry.layer)) {
        visibleLapNumber = otherLapNumber;
      }
    });
    // Update delta dropdown and reference when visible lap changes
    if (visibleLapNumber !== null) {
      updateDeltaDropdown();
      updateDeltaReference();
      if (currentSample) {
        updateDeltaReadout(currentSample);
      }
    }
  } else {
    // Hide all other laps first (only one visible at a time)
    lapLayers.forEach((otherEntry, otherLapNumber) => {
      if (otherLapNumber !== lapNumber && map.hasLayer(otherEntry.layer)) {
        map.removeLayer(otherEntry.layer);
        // Remove mouse events from the hidden lap
        removeMouseEventsFromLayer(otherEntry.layer);
        // Update button state for other laps
        const otherButton = document.querySelector(`button[data-lap="${otherLapNumber}"]`);
        if (otherButton) {
          otherButton.classList.remove('active');
        }
      }
    });
    
    // Show the selected lap
    entry.layer.addTo(map);
    button.classList.add('active');
    visibleLapNumber = lapNumber;
    // Attach mouse events to the newly visible lap
    attachMouseEventsToLayer(entry.layer);
    // Update delta dropdown to exclude the newly visible lap
    updateDeltaDropdown();
    // Update reference text
    updateDeltaReference();
    // Update readout if we have a current sample
    if (currentSample) {
      updateDeltaReadout(currentSample);
    }
  }
}

function attachMouseEventsToLayer(layer) {
  if (layer instanceof L.LayerGroup) {
    layer.eachLayer(child => {
      if (child instanceof L.Polyline) {
        child.on('mousemove', event => updateTelemetryAt(event.latlng));
      }
    });
  } else if (layer instanceof L.Polyline) {
    layer.on('mousemove', event => updateTelemetryAt(event.latlng));
  }
}

function removeMouseEventsFromLayer(layer) {
  if (layer instanceof L.LayerGroup) {
    layer.eachLayer(child => {
      if (child instanceof L.Polyline) {
        child.off('mousemove');
      }
    });
  } else if (layer instanceof L.Polyline) {
    layer.off('mousemove');
  }
}

function setupLapDelta() {
  const select = document.getElementById('delta-select');
  // Remove existing event listeners by cloning the select element
  const newSelect = select.cloneNode(true);
  select.parentNode.replaceChild(newSelect, select);
  const selectElement = document.getElementById('delta-select');
  selectElement.innerHTML = '';
  
  if (!allLaps.length || allLaps.length < 2) {
    selectElement.innerHTML = '<option value="">Not enough laps</option>';
    selectedComparisonLap = null;
    document.getElementById('delta-readout').textContent = 'Δ --';
    document.getElementById('delta-reference').textContent = '';
    return;
  }
  
  // Populate dropdown with all laps except the visible one
  updateDeltaDropdown();
  
  // Set default to first available lap (that's not the visible one)
  const availableLaps = allLaps.filter(lap => lap.lap_number !== visibleLapNumber);
  if (availableLaps.length > 0) {
    selectedComparisonLap = availableLaps[0].lap_number;
    selectElement.value = selectedComparisonLap;
    updateDeltaReference();
  }
  
  selectElement.addEventListener('change', event => {
    selectedComparisonLap = Number(event.target.value) || null;
    updateDeltaReference();
    // Immediately update readout if we have a current sample
    if (currentSample) {
      updateDeltaReadout(currentSample);
    } else {
      document.getElementById('delta-readout').textContent = 'Δ 0.00 s';
    }
  });
}

function updateDeltaDropdown() {
  const selectElement = document.getElementById('delta-select');
  selectElement.innerHTML = '';
  
  // Show all laps except the visible one
  const availableLaps = allLaps.filter(lap => lap.lap_number !== visibleLapNumber);
  
  if (availableLaps.length === 0) {
    selectElement.innerHTML = '<option value="">No other laps available</option>';
    return;
  }
  
  availableLaps.forEach(lap => {
    const option = document.createElement('option');
    option.value = lap.lap_number;
    option.textContent = `Lap ${lap.lap_number} (${formatNumber(lap.lap_time_s, 2)}s)`;
    if (lap.lap_number === selectedComparisonLap) {
      option.selected = true;
    }
    selectElement.appendChild(option);
  });
}

function updateDeltaReference() {
  if (!visibleLapNumber || !selectedComparisonLap) {
    document.getElementById('delta-reference').textContent = '';
    return;
  }
  
  const visibleLap = allLaps.find(l => l.lap_number === visibleLapNumber);
  const comparisonLap = allLaps.find(l => l.lap_number === selectedComparisonLap);
  
  if (visibleLap && comparisonLap) {
    document.getElementById('delta-reference').textContent =
      `Comparing Lap ${visibleLapNumber} to Lap ${selectedComparisonLap} (${formatNumber(comparisonLap.lap_time_s, 2)}s)`;
  } else {
    document.getElementById('delta-reference').textContent = '';
  }
}

function updateDeltaReadout(sample) {
  if (!visibleLapNumber || !selectedComparisonLap || !sample) {
    document.getElementById('delta-readout').textContent = 'Δ --';
    return;
  }
  
  const lapDistance = sample.lap_distance_m;
  if (lapDistance == null) {
    document.getElementById('delta-readout').textContent = 'Δ --';
    return;
  }
  
  // Calculate delta by comparing visible lap to selected comparison lap
  const delta = calculateDeltaAtDistance(visibleLapNumber, selectedComparisonLap, lapDistance);
  
  if (delta === null) {
    document.getElementById('delta-readout').textContent = 'Δ --';
    return;
  }
  
  const formatted = delta > 0 ? `+${formatNumber(delta, 2)}` : formatNumber(delta, 2);
  document.getElementById('delta-readout').textContent = `Δ ${formatted} s`;
}

function calculateDeltaAtDistance(lap1Number, lap2Number, distance) {
  // Get samples for both laps
  const lap1 = allLaps.find(l => l.lap_number === lap1Number);
  const lap2 = allLaps.find(l => l.lap_number === lap2Number);
  
  if (!lap1 || !lap2) return null;
  
  // Get telemetry samples for each lap
  const lap1Samples = telemetry.slice(lap1.start_sample_idx, lap1.end_sample_idx + 1)
    .filter(s => s.lap_distance_m != null && s.lap_elapsed_s != null)
    .map(s => ({ dist: parseFloat(s.lap_distance_m), time: parseFloat(s.lap_elapsed_s) }))
    .sort((a, b) => a.dist - b.dist);
  
  const lap2Samples = telemetry.slice(lap2.start_sample_idx, lap2.end_sample_idx + 1)
    .filter(s => s.lap_distance_m != null && s.lap_elapsed_s != null)
    .map(s => ({ dist: parseFloat(s.lap_distance_m), time: parseFloat(s.lap_elapsed_s) }))
    .sort((a, b) => a.dist - b.dist);
  
  if (lap1Samples.length < 2 || lap2Samples.length < 2) return null;
  
  // Interpolate time for lap1 at this distance
  const lap1Time = interpolateTime(lap1Samples, distance);
  if (lap1Time === null) return null;
  
  // Interpolate time for lap2 at this distance
  const lap2Time = interpolateTime(lap2Samples, distance);
  if (lap2Time === null) return null;
  
  // Delta = visible lap time - comparison lap time
  // Positive means visible lap is slower
  return lap1Time - lap2Time;
}

function interpolateTime(samples, distance) {
  // Find the two samples that bracket this distance
  for (let i = 1; i < samples.length; i++) {
    if (distance <= samples[i].dist) {
      const prev = samples[i - 1];
      const curr = samples[i];
      const ratio = (distance - prev.dist) / Math.max(curr.dist - prev.dist, 1e-6);
      return prev.time + ratio * (curr.time - prev.time);
    }
  }
  
  // If distance is beyond the last sample, use last sample's time
  if (distance > samples[samples.length - 1].dist) {
    return samples[samples.length - 1].time;
  }
  
  // If distance is before the first sample, use first sample's time
  if (distance < samples[0].dist) {
    return samples[0].time;
  }
  
  return null;
}

function renderCornerList(corners) {
  const container = document.getElementById('corner-list');
  container.innerHTML = '';
  cornerGroup.clearLayers();
  if (!corners.length) {
    container.innerHTML = '<p>No corners detected yet.</p>';
    return;
  }
  corners.forEach(corner => {
    const entry = document.createElement('div');
    entry.className = 'corner-entry';
    entry.innerHTML = `
      <strong>Corner ${corner.corner_id} · Lap ${corner.lap_number}</strong>
      <span>${formatNumber(corner.min_speed_mph, 1)} mph min · ${formatNumber(corner.max_lat_g, 2)} g</span>
    `;
    entry.addEventListener('click', () => highlightCorner(corner));
    container.appendChild(entry);
  });
}

function highlightCorner(corner) {
  cornerGroup.clearLayers();
  const geometry = corner.geometry || [];
  if (!geometry.length) return;
  const latlngs = geometry.map(([lon, lat]) => [lat, lon]);
  const polyline = L.polyline(latlngs, {
    color: '#ffffff',
    weight: 5,
    dashArray: '8 6',
  }).addTo(cornerGroup);
  const apex = corner.apex;
  if (apex && apex.lat != null && apex.lon != null) {
    L.circleMarker([apex.lat, apex.lon], {
      radius: 6,
      color: '#ffd60a',
      fillColor: '#ffd60a',
      fillOpacity: 1,
    }).addTo(cornerGroup);
  }
  map.fitBounds(polyline.getBounds(), { padding: [50, 50] });
}

function setupHeatmapControls(heatmap) {
  const checkboxes = document.querySelectorAll('#heatmap-panel input[type="checkbox"]');
  checkboxes.forEach(box => {
    box.checked = false;
    box.addEventListener('change', event => {
      const layerName = event.target.dataset.layer;
      toggleHeatmap(layerName, event.target.checked, heatmap[layerName] || []);
    });
  });
}

function toggleHeatmap(name, enabled, points) {
  if (heatmapLayers[name]) {
    map.removeLayer(heatmapLayers[name]);
    delete heatmapLayers[name];
  }
  if (!enabled || !points.length) return;
  const colorMap = {
    brake: '#ff2d55',
    throttle: '#00ffa3',
    lateral: '#ffd60a',
  };
  const markers = points.map(point =>
    L.circleMarker([point.lat, point.lon], {
      radius: 4 + 8 * point.intensity,
      color: colorMap[name] || '#ffffff',
      fillColor: colorMap[name] || '#ffffff',
      fillOpacity: 0.35 + 0.5 * point.intensity,
      weight: 0,
    })
  );
  const layer = L.layerGroup(markers).addTo(map);
  heatmapLayers[name] = layer;
}

function setupExportControls(laps) {
  const lapSelect = document.getElementById('export-lap-select');
  lapSelect.innerHTML = '';
  laps.forEach(lap => {
    const option = document.createElement('option');
    option.value = lap.lap_number;
    option.textContent = `Lap ${lap.lap_number} (${formatNumber(lap.lap_time_s, 1)} s)`;
    lapSelect.appendChild(option);
  });
  if (!laps.length) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No laps available';
    lapSelect.appendChild(option);
  }

  document.getElementById('export-lap-btn').addEventListener('click', () => {
    const lapNumber = lapSelect.value;
    if (!lapNumber) return;
    const exportUrl = dataset ? `/api/export/lap/${lapNumber}?dataset=${encodeURIComponent(dataset)}` : `/api/export/lap/${lapNumber}`;
    window.open(exportUrl, '_blank');
  });
  document.getElementById('export-session-btn').addEventListener('click', () => {
    const exportUrl = dataset ? `/api/export/session?dataset=${encodeURIComponent(dataset)}` : '/api/export/session';
    window.open(exportUrl, '_blank');
  });
}

