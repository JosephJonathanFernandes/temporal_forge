document.addEventListener('DOMContentLoaded', function () {
  // Read records embedded as JSON
  const recEl = document.getElementById('result-data');
  const records = recEl ? JSON.parse(recEl.textContent || '[]') : [];

  // Populate healer filter options
  const healerSet = new Set(records.map(r => r.healer).filter(Boolean));
  const healerSelect = document.getElementById('filter_healer');
  if (healerSelect) {
    healerSet.forEach(h => {
      const opt = document.createElement('option'); opt.value = h; opt.textContent = h; healerSelect.appendChild(opt);
    });
  }

  // Table filtering logic
  const table = document.getElementById('recordsTable');
  const filterHealer = document.getElementById('filter_healer');
  const filterSentiment = document.getElementById('filter_sentiment');

  function applyFilters() {
    const h = filterHealer ? filterHealer.value : '';
    const s = filterSentiment ? filterSentiment.value : '';
    if (!table) return;
    const rows = table.querySelectorAll('tbody tr');
    rows.forEach(row => {
      const cells = row.querySelectorAll('td');
      const healer = (cells[0] && cells[0].textContent.trim()) || '';
      const sentiment = (cells[4] && cells[4].textContent.trim()) || '';
      let show = true;
      if (h && healer !== h) show = false;
      if (s && sentiment.toLowerCase() !== s.toLowerCase()) show = false;
      row.style.display = show ? '' : 'none';
    });
  }

  if (filterHealer) filterHealer.addEventListener('change', applyFilters);
  if (filterSentiment) filterSentiment.addEventListener('change', applyFilters);

  // Build a simple Cytoscape network from records (healer -> cure edges)
  const networkEl = document.getElementById('network');
  if (networkEl && typeof cytoscape !== 'undefined' && records.length > 0) {
    const nodes = {};
    const edges = [];
    records.forEach((r, idx) => {
      const healer = r.healer || ('Healer_' + idx);
      const cure = r.cure || ('Cure_' + idx);
      if (!nodes['h:' + healer]) nodes['h:' + healer] = { data: { id: 'h:' + healer, label: healer, type: 'healer' } };
      if (!nodes['c:' + cure]) nodes['c:' + cure] = { data: { id: 'c:' + cure, label: cure, type: 'cure' } };
      edges.push({ data: { id: 'e:' + idx, source: 'h:' + healer, target: 'c:' + cure } });
    });
    const elements = Object.values(nodes).concat(edges);
    // initialize cytoscape
    cytoscape({
      container: networkEl,
      elements: elements,
      style: [
        { selector: 'node[type="healer"]', style: { 'background-color': '#8da0cb', 'label': 'data(label)', 'text-valign': 'center', 'text-halign': 'center' } },
        { selector: 'node[type="cure"]', style: { 'background-color': '#66c2a5', 'label': 'data(label)', 'text-valign': 'center', 'text-halign': 'center' } },
        { selector: 'edge', style: { 'width': 2, 'line-color': '#ccc', 'target-arrow-color': '#ccc', 'target-arrow-shape': 'triangle' } }
      ],
      layout: { name: 'cose', animate: true }
    });
  } else if (networkEl) {
    // leave placeholder text if no data or cytoscape missing
    networkEl.textContent = records.length ? 'Cytoscape not loaded' : 'No network data available yet.';
  }
});
