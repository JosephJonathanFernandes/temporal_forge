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
