// ui.js — client-side behaviors for The Healer's Scribe
document.addEventListener('DOMContentLoaded', function(){
  const form = document.getElementById('processForm');
  const overlay = document.getElementById('processingOverlay');
  if (form && overlay) {
    form.addEventListener('submit', function(e){
      // show overlay when submitting (except when clicking download buttons — they still submit)
      overlay.style.display = 'flex';
    });
  }

  // enhance table: allow CSV export of visible rows
  const exportBtn = document.getElementById('exportVisibleCsv');
  if (exportBtn) {
    exportBtn.addEventListener('click', function(){
      const table = document.querySelector('table');
      if (!table) return;
      const rows = Array.from(table.querySelectorAll('tr')).filter(r => r.style.display !== 'none');
      const csv = rows.map(r => Array.from(r.querySelectorAll('th,td')).map(c => '"' + c.textContent.replace(/"/g, '""') + '"').join(',')).join('\n');
      const blob = new Blob([csv], {type: 'text/csv'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = 'visible_records.csv'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    });
  }
  
  // WOW Feature: Wisdom Timeline
  generateWisdomTimeline();
  // render network after timeline
  renderHealerNetwork();
});

// Sorting helper for records table
function sortTable(colIndex) {
  const table = document.getElementById('recordsTable');
  if (!table) return;
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const currentSort = table.getAttribute('data-sort-col');
  const currentDir = table.getAttribute('data-sort-dir') || 'asc';
  let dir = 'asc';
  if (currentSort == colIndex.toString() && currentDir == 'asc') dir = 'desc';
  rows.sort((a,b) => {
    const A = (a.children[colIndex] && a.children[colIndex].textContent.trim()) || '';
    const B = (b.children[colIndex] && b.children[colIndex].textContent.trim()) || '';
    if (!isNaN(parseFloat(A)) && !isNaN(parseFloat(B))) {
      return dir === 'asc' ? parseFloat(A)-parseFloat(B) : parseFloat(B)-parseFloat(A);
    }
    return dir === 'asc' ? A.localeCompare(B) : B.localeCompare(A);
  });
  // reattach
  rows.forEach(r => tbody.appendChild(r));
  table.setAttribute('data-sort-col', colIndex);
  table.setAttribute('data-sort-dir', dir);
  // update aria-sort on headers
  const headers = table.querySelectorAll('th[role="button"]');
  headers.forEach((h, i) => {
    if (i === colIndex) h.setAttribute('aria-sort', dir === 'asc' ? 'ascending' : 'descending');
    else h.setAttribute('aria-sort', 'none');
  });
}

// attach keyboard handlers to sortable headers
document.addEventListener('DOMContentLoaded', function(){
  const table = document.getElementById('recordsTable');
  if (!table) return;
  const headers = table.querySelectorAll('th[role="button"]');
  headers.forEach((h, idx) => {
    h.addEventListener('keydown', function(e){
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sortTable(idx); }
    });
  });
});

// WOW Feature: Find Similar Cases
function findSimilarCases() {
  const query = document.getElementById('similarQuery').value.trim();
  const resultsDiv = document.getElementById('similarResults');
  
  if (!query) {
    resultsDiv.innerHTML = '<div class="alert alert-warning">Please enter a query</div>';
    return;
  }
  
  resultsDiv.innerHTML = '<div class="text-muted">Searching...</div>';
  
  // Get original text from hidden data
  const resultData = document.getElementById('result-data');
  const originalText = resultData ? resultData.getAttribute('data-original-text') : '';
  
  fetch('/api/similar', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: query, text: originalText || document.querySelector('textarea[name="text"]')?.value || ''})
  })
  .then(res => {
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    return res.json();
  })
  .then(data => {
    if (data.error) {
      resultsDiv.innerHTML = `<div class="alert alert-danger">⚠️ ${data.error}</div>`;
      return;
    }
    
    if (data.similar_cases && data.similar_cases.length > 0) {
      let html = '<div class="list-group">';
      data.similar_cases.forEach((c, i) => {
        const score = (c.similarity_score * 100).toFixed(1);
        html += `
          <div class="list-group-item">
            <div class="d-flex justify-content-between">
              <h6 class="mb-1">${i+1}. ${c.healer || 'Unknown'} - ${c.cure || 'N/A'}</h6>
              <small class="text-muted">${score}% match</small>
            </div>
            <p class="mb-1 small"><strong>Symptom:</strong> ${c.symptom || 'N/A'}</p>
            <p class="mb-1 small"><strong>Outcome:</strong> ${c.outcome || 'N/A'}</p>
            <small class="text-muted">Sentiment: ${c.sentiment || 'neutral'}</small>
          </div>
        `;
      });
      html += '</div>';
      resultsDiv.innerHTML = html;
    } else {
      resultsDiv.innerHTML = '<div class="alert alert-info">🔍 No similar cases found. Try a different query.</div>';
    }
  })
  .catch(err => {
    console.error('Similar cases error:', err);
    resultsDiv.innerHTML = '<div class="alert alert-danger">❌ Search failed. Please try again or check your connection.</div>';
  });
}

// Add keyboard shortcut for Similar Case search (Enter key)
document.addEventListener('DOMContentLoaded', function() {
  const similarQuery = document.getElementById('similarQuery');
  if (similarQuery) {
    similarQuery.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        findSimilarCases();
      }
    });
  }
});

// WOW Feature: Wisdom Timeline
function generateWisdomTimeline() {
  const timelineDiv = document.getElementById('timelineChart');
  if (!timelineDiv) return;
  
  // Get records from table
  const table = document.getElementById('recordsTable');
  if (!table) return;
  
  const rows = Array.from(table.querySelectorAll('tbody tr'));
  if (rows.length === 0) {
    timelineDiv.innerHTML = '<div class="text-muted text-center py-4">No data available</div>';
    return;
  }
  
  // Extract data: simulate chronological order (index = time)
  const cureData = {};
  rows.forEach((row, idx) => {
    const cure = row.children[1]?.textContent.trim() || 'unknown';
    const sentiment = row.children[4]?.textContent.trim().toLowerCase() || 'neutral';
    
    if (!cureData[cure]) {
      cureData[cure] = { positive: [], negative: [], neutral: [] };
    }
    
    // Accumulate counts over time
    if (sentiment === 'positive') cureData[cure].positive.push(idx);
    else if (sentiment === 'negative') cureData[cure].negative.push(idx);
    else cureData[cure].neutral.push(idx);
  });
  
  // Build timeline traces (top 5 cures by total mentions)
  const sortedCures = Object.keys(cureData).sort((a, b) => {
    const totalA = cureData[a].positive.length + cureData[a].negative.length + cureData[a].neutral.length;
    const totalB = cureData[b].positive.length + cureData[b].negative.length + cureData[b].neutral.length;
    return totalB - totalA;
  }).slice(0, 5);
  
  const traces = [];
  sortedCures.forEach(cure => {
    const data = cureData[cure];
    const allIndices = [...data.positive, ...data.negative, ...data.neutral].sort((a,b) => a-b);
    
    // Calculate cumulative effectiveness (positive - negative)
    const cumulativeScore = [];
    let score = 0;
    allIndices.forEach(idx => {
      if (data.positive.includes(idx)) score++;
      else if (data.negative.includes(idx)) score--;
      cumulativeScore.push(score);
    });
    
    traces.push({
      x: allIndices.map(i => `Record ${i+1}`),
      y: cumulativeScore,
      type: 'scatter',
      mode: 'lines+markers',
      name: cure,
      line: { width: 2 }
    });
  });
  
  const layout = {
    title: 'Cure Effectiveness Over Time',
    xaxis: { title: 'Chronological Records' },
    yaxis: { title: 'Cumulative Effectiveness (+ = effective, - = failed)' },
    margin: { l: 50, r: 20, t: 40, b: 80 },
    height: 300,
    showlegend: true,
    legend: { orientation: 'h', y: -0.3 }
  };
  
  Plotly.newPlot(timelineDiv, traces, layout, {responsive: true});
}

  // WOW Feature: Healer–Cure Network (Cytoscape)
  function renderHealerNetwork() {
    const container = document.getElementById('network');
    if (!container || typeof cytoscape === 'undefined') return;

    // try to read structured JSON first (injected by template)
    let records = [];
    const jsonEl = document.getElementById('records-data-json');
    if (jsonEl) {
      try { records = JSON.parse(jsonEl.textContent || jsonEl.innerText || '[]'); } catch(e) { records = []; }
    }

    // fallback: read table rows
    if ((!records || records.length === 0) && document.getElementById('recordsTable')) {
      const rows = Array.from(document.querySelectorAll('#recordsTable tbody tr'));
      records = rows.map(r => ({ healer: r.children[0]?.textContent.trim(), cure: r.children[1]?.textContent.trim(), symptom: r.children[2]?.textContent.trim(), outcome: r.children[3]?.textContent.trim(), sentiment: r.children[4]?.textContent.trim() }));
    }

    if (!records || records.length === 0) {
      container.innerHTML = '<div class="text-muted text-center py-4">No network data available</div>';
      return;
    }

    // Build nodes and edges: healer nodes and cure nodes
    const healerCounts = {};
    const cureCounts = {};
    const edgeMap = {}; // key: healer|cure -> weight

    records.forEach(rec => {
      const h = (rec.healer || 'Unknown').trim() || 'Unknown';
      const c = (rec.cure || 'Unknown').trim() || 'Unknown';
      healerCounts[h] = (healerCounts[h] || 0) + 1;
      cureCounts[c] = (cureCounts[c] || 0) + 1;
      const key = `${h}|||${c}`;
      edgeMap[key] = (edgeMap[key] || 0) + 1;
    });

    const elements = [];
    // add healer nodes
    Object.keys(healerCounts).forEach(h => {
      elements.push({ data: { id: `healer::${h}`, label: h, type: 'healer', count: healerCounts[h] } });
    });
    // add cure nodes
    Object.keys(cureCounts).forEach(c => {
      elements.push({ data: { id: `cure::${c}`, label: c, type: 'cure', count: cureCounts[c] } });
    });
    // add edges
    Object.keys(edgeMap).forEach(k => {
      const [h,c] = k.split('|||');
      elements.push({ data: { id: `e::${h}::${c}`, source: `healer::${h}`, target: `cure::${c}`, weight: edgeMap[k] } });
    });

    // clear container
    container.innerHTML = '';

    const cy = cytoscape({
      container: container,
      elements: elements,
      boxSelectionEnabled: false,
      autounselectify: false,
      style: [
        {
          selector: 'node[type="healer"]',
          style: {
            'background-color': '#7c3aed',
            'label': 'data(label)',
            'color': '#fff',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 'mapData(count, 1, 10, 40, 110)',
            'height': 'mapData(count, 1, 10, 40, 110)',
            'text-wrap': 'wrap',
            'text-max-width': 80,
            'font-weight': '600',
            'overlay-padding': '6px',
            'z-index': 10
          }
        },
        {
          selector: 'node[type="cure"]',
          style: {
            'background-color': '#06b6d4',
            'label': 'data(label)',
            'color': '#042028',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 'mapData(count, 1, 10, 24, 80)',
            'height': 'mapData(count, 1, 10, 24, 80)',
            'text-wrap': 'wrap',
            'text-max-width': 100,
            'font-weight': '500',
            'overlay-padding': '4px'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 'mapData(weight, 1, 10, 1, 6)',
            'line-color': '#b3e5ff',
            'target-arrow-color': '#b3e5ff',
            'curve-style': 'haystack',
            'opacity': 0.9
          }
        },
        { selector: 'node:selected', style: { 'overlay-color': '#fbcfe8', 'overlay-padding': 6, 'overlay-opacity': 0.25 } },
        { selector: ':selected', style: { 'border-width': 2, 'border-color': '#fff' } }
      ],
      layout: { name: 'cose', idealEdgeLength: 80, nodeOverlap: 18, refresh: 20, randomize: true }
    });

  // add simple info box
  let info = document.getElementById('networkInfo');
  if (!info) {
    info = document.createElement('div');
    info.id = 'networkInfo';
    info.className = 'network-info card';
    info.style.position = 'absolute';
    info.style.right = '16px';
    info.style.top = '16px';
    info.style.zIndex = 9999;
    info.style.minWidth = '180px';
    info.style.padding = '8px 12px';
    info.style.display = 'none';
    container.style.position = 'relative';
    container.appendChild(info);
  }

  // Add legend
  let legend = document.getElementById('networkLegend');
  if (!legend) {
    legend = document.createElement('div');
    legend.id = 'networkLegend';
    legend.className = 'network-legend';
    legend.style.position = 'absolute';
    legend.style.left = '16px';
    legend.style.top = '16px';
    legend.style.background = 'rgba(255,255,255,0.95)';
    legend.style.padding = '12px';
    legend.style.borderRadius = '8px';
    legend.style.fontSize = '13px';
    legend.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
    legend.style.zIndex = 9999;
    legend.innerHTML = `
      <div style="font-weight:600; margin-bottom:8px;">Legend</div>
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
        <div style="width:16px; height:16px; border-radius:50%; background:#7c3aed;"></div>
        <span>Healers</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
        <div style="width:16px; height:16px; border-radius:50%; background:#06b6d4;"></div>
        <span>Cures</span>
      </div>
      <div style="margin-top:8px; padding-top:8px; border-top:1px solid #e5e7eb; font-size:11px; color:#6b7280;">
        💡 Node size = frequency<br>
        💡 Edge width = co-occurrence
      </div>
    `;
    container.appendChild(legend);
  }

  cy.on('tap', 'node', function(evt){
    const d = evt.target.data();
    info.style.display = 'block';
    info.innerHTML = `<strong>${d.label}</strong><div style="font-size:13px;color:#444;margin-top:6px">Type: ${d.type}</div><div style="font-size:13px;color:#444">Records: ${d.count || 0}</div>`;
  });
  cy.on('tap', function(evt){ if (evt.target === cy) { info.style.display = 'none'; } });

  // fit to viewport
  cy.fit(40);
}