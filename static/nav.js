(function () {
  const pages = [["/about", "Giới thiệu"], ["/", "Phân tích và sinh ảnh"], ["/parse", "Thử phân tích"], ["/cases", "Lịch sử ca"], ["/compare", "So sánh mô hình"], ["/evaluation", "Kết quả đánh giá"], ["/rate", "Chấm điểm"], ["/stats", "Thống kê"], ["/dictionary", "Từ điển bệnh"], ["/export", "Xuất kết quả"]];
  const path = location.pathname.replace(/\/$/, "") || "/";
  const el = document.getElementById("nav");
  if (!el) return;
  el.innerHTML = '<div class="in">' + pages.map(([h, t]) =>
    `<a href="${h}" class="${h === path ? "on" : ""}">${t}</a>`).join("") + "</div>";
})();