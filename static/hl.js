// Tô màu văn bản báo cáo theo kết quả của /api/parse (dùng chung cho trang "Thử phân tích" và "Lịch sử ca")
function hlEsc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function renderHighlight(text, hits) {
  let out = '', pos = 0;
  hits.slice().sort((a, b) => a.start - b.start).forEach(h => {
    if (h.start < pos) return;
    out += hlEsc(text.slice(pos, h.start));
    let cls, tip;
    if (h.negated)        { cls = 'neg'; tip = `Bị phủ định${h.cue ? ' (từ khóa: "' + h.cue + '")' : ''} → loại khỏi ảnh · ${h.en}`; }
    else if (h.uncertain) { cls = 'unc'; tip = `Không chắc chắn${h.cue ? ' (từ khóa: "' + h.cue + '")' : ''} · ${h.en}`; }
    else if (h.supported) { cls = 'ok';  tip = `Được hỗ trợ, sẽ vẽ ảnh · ${h.en}`; }
    else                  { cls = 'un';  tip = `Chưa hỗ trợ sinh ảnh · ${h.en}`; }
    out += `<mark class="hl ${cls}" title="${hlEsc(tip)}">${hlEsc(text.slice(h.start, h.end))}</mark>`;
    pos = h.end;
  });
  return out + hlEsc(text.slice(pos));
}
const HL_LEGEND = '<div class="legend"><span><mark class="hl ok">bệnh được hỗ trợ</mark></span><span><mark class="hl un">bệnh chưa hỗ trợ</mark></span><span><mark class="hl unc">không chắc chắn</mark></span><span><mark class="hl neg">bị phủ định</mark></span></div>';