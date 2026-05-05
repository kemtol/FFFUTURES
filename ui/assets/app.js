const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

let chart;
let candleSeries;
let pnlChart;
let pnlGreenSeries;
let pnlRedSeries;
let allCandles = [];
let allTrades = [];
let filteredTrades = [];
let selectedTradeNo = null;
let currentTimeframe = "5m";
let currentStrategy = "st_dema_adx_cci";

function setStatus(text) {
  $("#status").text(text);
}

function telegramApiUrl() {
  const host = window.location.hostname || "127.0.0.1";
  if (window.location.port === "8080") return "/api/telegram-signal";
  return `${window.location.protocol}//${host}:8080/api/telegram-signal`;
}

function setLoading(isLoading, text = "Loading...") {
  $("#loadingText").text(text);
  $("#loadingOverlay").toggleClass("show", isLoading).attr("aria-hidden", isLoading ? "false" : "true");
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function initChart() {
  const el = $("#chart").get(0);
  chart = LightweightCharts.createChart(el, {
    layout: { background: { color: "#101214" }, textColor: "#cfd5df" },
    grid: { vertLines: { color: "transparent" }, horzLines: { color: "transparent" } },
    rightPriceScale: { borderColor: "#303640" },
    timeScale: { borderColor: "#303640", timeVisible: true, secondsVisible: false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: "#2ebd85",
    downColor: "#f05d5e",
    borderUpColor: "#2ebd85",
    borderDownColor: "#f05d5e",
    wickUpColor: "#2ebd85",
    wickDownColor: "#f05d5e",
  });
  $(window).on("resize", function () {
    chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    resizePnlChart();
  });
}

function initPnlChart() {
  const el = $("#pnlChart").get(0);
  pnlChart = LightweightCharts.createChart(el, {
    layout: { background: { color: "#101214" }, textColor: "#cfd5df" },
    grid: { vertLines: { color: "transparent" }, horzLines: { color: "transparent" } },
    rightPriceScale: { borderColor: "#303640" },
    handleScroll: false,
    handleScale: false,
    timeScale: {
      borderColor: "#303640",
      timeVisible: true,
      secondsVisible: false,
      fixLeftEdge: true,
      fixRightEdge: true,
      lockVisibleTimeRangeOnResize: true,
      rightOffset: 0,
      barSpacing: 6,
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });
  pnlGreenSeries = pnlChart.addAreaSeries({
    lineColor: "#2ebd85",
    topColor: "rgba(46, 189, 133, .28)",
    bottomColor: "rgba(46, 189, 133, .02)",
    lineWidth: 2,
    priceFormat: { type: "price", precision: 0, minMove: 1 },
  });
  pnlGreenSeries.createPriceLine({
    price: 0,
    color: "#6b7078",
    lineWidth: 1,
    lineStyle: 1,
    axisLabelVisible: false,
  });
  pnlRedSeries = pnlChart.addLineSeries({
    color: "#f05d5e",
    lineWidth: 2,
    priceFormat: { type: "price", precision: 0, minMove: 1 },
  });
}

function resizePnlChart() {
  if (!pnlChart) return;
  const el = $("#pnlChart").get(0);
  pnlChart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
}

function getFilteredTrades() {
  const range = getActiveDateRange();
  const side = $("#sideFilter").val();
  const sessions = sessionVals();
  const result = $("#resultFilter").val();
  const gapMin = parseFloat($("#gapMin").val());
  const gapMax = parseFloat($("#gapMax").val());
  const adxMin = parseFloat($("#adxMin").val());
  const adxMax = parseFloat($("#adxMax").val());
  const cciMin = parseFloat($("#cciMin").val());
  const cciMax = parseFloat($("#cciMax").val());
  const chopMin = parseFloat($("#chopMin").val());
  const chopMax = parseFloat($("#chopMax").val());
  const demaSlopeMin = parseFloat($("#demaSlopeMin").val());
  const demaDistMin = parseFloat($("#demaDistMin").val());
  const demaDistMax = parseFloat($("#demaDistMax").val());

  return allTrades.filter(function (t) {
    if (range.startTime !== null && t.entry_time < range.startTime) return false;
    if (range.endTime !== null && t.entry_time > range.endTime) return false;
    if (side !== "All" && t.side !== side) return false;
    if (sessions.length && !sessions.includes(t.session)) return false;
    if (result === "Win" && !t.is_win) return false;
    if (result === "Loss" && t.is_win) return false;
    var gp = t.entry_gap_pts || 0;
    if (!Number.isNaN(gapMin) && gp < gapMin) return false;
    if (!Number.isNaN(gapMax) && gp > gapMax) return false;
    if (!Number.isNaN(adxMin) && t.entry_adx < adxMin) return false;
    if (!Number.isNaN(adxMax) && t.entry_adx > adxMax) return false;
    var cc = t.entry_cci || 0;
    if (!Number.isNaN(cciMin) && cc < cciMin) return false;
    if (!Number.isNaN(cciMax) && cc > cciMax) return false;
    var ch = t.entry_chop || 0;
    if (!Number.isNaN(chopMin) && ch < chopMin) return false;
    if (!Number.isNaN(chopMax) && ch > chopMax) return false;
    var ds = t.entry_dema_slope || 0;
    if (!Number.isNaN(demaSlopeMin) && ds < demaSlopeMin) return false;
    var dd = t.dema_distance_atr || 0;
    if (!Number.isNaN(demaDistMin) && dd < demaDistMin) return false;
    if (!Number.isNaN(demaDistMax) && dd > demaDistMax) return false;
    return true;
  });
}

function getActiveDateRange() {
  const preset = $("#rangePreset").val();
  const startDate = $("#dateStart").val();
  const endDate = $("#dateEnd").val();
  const maxEntryTime = allTrades.length ? Math.max(...allTrades.map((t) => t.entry_time)) : null;
  let startTime = null;
  let endTime = null;

  if (preset === "7D" && maxEntryTime !== null) {
    endTime = endOfUtcDay(maxEntryTime);
    startTime = startOfUtcDay(endTime - 6 * 86400);
  } else if (preset === "30D" && maxEntryTime !== null) {
    endTime = endOfUtcDay(maxEntryTime);
    startTime = startOfUtcDay(endTime - 29 * 86400);
  } else if (preset === "90D" && maxEntryTime !== null) {
    endTime = endOfUtcDay(maxEntryTime);
    startTime = startOfUtcDay(endTime - 89 * 86400);
  } else if (preset === "Custom") {
    if (startDate) startTime = Date.parse(`${startDate}T00:00:00Z`) / 1000;
    if (endDate) endTime = Date.parse(`${endDate}T23:59:59Z`) / 1000;
  }

  return { preset, startTime, endTime };
}

function startOfUtcDay(epochSec) {
  const d = new Date(epochSec * 1000);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), 0, 0, 0) / 1000;
}

function endOfUtcDay(epochSec) {
  return startOfUtcDay(epochSec) + 86399;
}

function formatDateRangeLabel(range, trades) {
  if (!trades.length) return "No trades in selected range";

  const minTs = Math.min(...trades.map((t) => t.entry_time));
  const maxTs = Math.max(...trades.map((t) => t.entry_time));
  const start = new Date(minTs * 1000).toISOString().slice(0, 10);
  const end = new Date(maxTs * 1000).toISOString().slice(0, 10);
  const label = range.preset === "All" ? "All data" : $("#rangePreset option:selected").text();
  return `${label}: ${start} to ${end}`;
}

function renderStats(trades) {
  const total = trades.length;
  const wins = trades.filter((t) => t.is_win).length;
  const pnl = trades.reduce((s, t) => s + t.pnl_usd, 0);
  const avgR = total ? trades.reduce((s, t) => s + (t.r_multiple ?? 0), 0) / total : 0;
  const grossProfit = trades.filter((t) => t.pnl_usd > 0).reduce((s, t) => s + t.pnl_usd, 0);
  const grossLoss = Math.abs(trades.filter((t) => t.pnl_usd < 0).reduce((s, t) => s + t.pnl_usd, 0));
  const winRate = total ? (wins / total) * 100 : 0;
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);

  $("#statTrades").text(total.toLocaleString());
  setMetric("#statWin", total ? `${winRate.toFixed(1)}%` : "0%", winRate >= 50, winRate > 0 && winRate < 50);
  setMetric("#statPnl", money.format(pnl), pnl > 0, pnl < 0);
  setMetric("#statPf", Number.isFinite(profitFactor) ? profitFactor.toFixed(2) : "Inf", profitFactor > 1, profitFactor < 1);
  setMetric("#statR", avgR.toFixed(2), avgR > 0, avgR < 0);
}

function renderPnlCurve(trades) {
  if (!pnlGreenSeries || !pnlRedSeries) return;
  const rows = trades.slice().sort((a, b) => a.exit_time - b.exit_time);
  let cumulative = 0;
  const green = [];
  const red = [];
  rows.forEach((t) => {
    cumulative += t.pnl_usd;
    const ts = t.exit_time;
    const val = Math.round(cumulative);
    green.push({ time: ts, value: Math.max(0, val) });
    red.push({ time: ts, value: Math.min(0, val) });
  });
  pnlGreenSeries.setData(green);
  pnlRedSeries.setData(red);
  pnlChart.timeScale().fitContent();
  pnlChart.timeScale().applyOptions({ rightOffset: 0 });
}

function setMetric(selector, text, isPositive, isNegative) {
  const $el = $(selector);
  $el.text(text).removeClass("positive negative neutral");
  if (isPositive) {
    $el.addClass("positive");
  } else if (isNegative) {
    $el.addClass("negative");
  } else {
    $el.addClass("neutral");
  }
}

function renderMarkers(trades) {
  const markers = [];
  for (const t of trades) {
    markers.push({
      time: markerTime(t.entry_time),
      position: t.side === "Long" ? "belowBar" : "aboveBar",
      color: t.side === "Long" ? "#2ebd85" : "#f05d5e",
      shape: t.side === "Long" ? "arrowUp" : "arrowDown",
      text: `${t.trade_no} ${t.side}`,
    });
    markers.push({
      time: markerTime(t.exit_time),
      position: t.side === "Long" ? "aboveBar" : "belowBar",
      color: t.is_win ? "#2ebd85" : "#d7a84f",
      shape: "circle",
      text: `${t.trade_no} exit`,
    });
  }
  candleSeries.setMarkers(markers);
}

function markerTime(epochSec) {
  if (currentTimeframe === "15m") return Math.floor(epochSec / 900) * 900;
  if (currentTimeframe === "5m") return Math.floor(epochSec / 300) * 300;
  return epochSec;
}

function updateTableHeader() {
  const isFVG = currentStrategy === "fvg_scalper";
  $(".col-dynamic").eq(0).text(isFVG ? "Gap" : "CCI");
  $(".col-dynamic").eq(1).text(isFVG ? "CHOP" : "");
} 

function renderTable(trades) {
  const $body = $("#tradeRows");
  $body.empty();

  const isFVG = currentStrategy === "fvg_scalper";
  updateTableHeader();

  const rows = trades.slice().sort((a, b) => b.entry_time - a.entry_time);
  for (const t of rows) {
    const sideClass = t.side === "Long" ? "long" : "short";
    const pnlClass = t.pnl_usd >= 0 ? "win" : "loss";
    const rClass = t.r_multiple >= 0 ? "win" : "loss";
    const selectedClass = t.trade_no === selectedTradeNo ? "selected" : "";
    const rText = t.r_multiple?.toFixed(2) ?? "";

    const dyn1 = isFVG
      ? (t.entry_gap_pts?.toFixed(1) ?? "—")
      : (t.entry_cci?.toFixed(0) ?? "—");
    const dyn2 = isFVG
      ? (t.entry_chop?.toFixed(1) ?? "—")
      : (t.entry_gap_pts?.toFixed(1) ?? "—");

    const $row = $(`
      <tr class="${selectedClass}" data-trade-no="${t.trade_no}">
        <td>${t.trade_no}</td>
        <td>${t.entry_ts}</td>
        <td class="${sideClass}">${t.side}</td>
        <td class="text-end">${fmt.format(t.entry_price)}</td>
        <td class="text-end ${pnlClass}">${money.format(t.pnl_usd)}</td>
        <td class="text-end ${rClass}">${rText}</td>
        <td class="text-end">${t.entry_adx?.toFixed(1) ?? "—"}</td>
        <td class="text-end">${dyn1}</td>
        <td class="text-end">${dyn2}</td>
        <td>${t.session ?? "—"}</td>
        <td>${t.exit_ts} <span class="pill">${t.exit_reason}</span></td>
      </tr>
    `);
    $body.append($row);
  }
}

function renderDetail(t) {
  const $el = $("#tradeDetail");
  if (!t) {
    $el.html("<span>Select a row to inspect the setup.</span>");
    return;
  }

  const isFVG = currentStrategy === "fvg_scalper";
  const extraFields = isFVG
    ? `<div>ADX: <b>${t.entry_adx?.toFixed(2) ?? "n/a"}</b> | CHOP: <b>${t.entry_chop?.toFixed(1) ?? "n/a"}</b></div>
       <div>Gap: <b>${t.entry_gap_pts?.toFixed(1) ?? "n/a"} pts</b> | DEMA: <b>${t.entry_dema?.toFixed(1) ?? "n/a"}</b></div>
       <div>DEMA slope: <b>${t.entry_dema_slope?.toFixed(4) ?? "n/a"}</b> | DEMA dist ATR: <b>${t.dema_distance_atr?.toFixed(2) ?? "n/a"}</b></div>`
    : `<div>ADX: <b>${t.entry_adx?.toFixed(2) ?? "n/a"}</b> | CCI: <b>${t.entry_cci?.toFixed(2) ?? "n/a"}</b></div>
       <div>DEMA dist ATR: <b>${t.dema_distance_atr?.toFixed(2) ?? "n/a"}</b> | ST dist ATR: <b>${t.st_distance_atr?.toFixed(2) ?? "n/a"}</b></div>`;

  $el.html(`
    <div><b>#${t.trade_no}</b> <span class="${t.side === "Long" ? "long" : "short"}">${t.side}</span> <span class="pill">${t.session}</span></div>
    <div>Entry: <b>${t.entry_ts}</b> at <b>${fmt.format(t.entry_price)}</b></div>
    <div>Exit: <b>${t.exit_ts}</b> at <b>${fmt.format(t.exit_price)}</b> <span class="pill">${t.exit_reason}</span></div>
    <div>PnL: <b class="${t.pnl_usd >= 0 ? "win" : "loss"}">${money.format(t.pnl_usd)}</b> | R: <b>${t.r_multiple?.toFixed(2) ?? "n/a"}</b></div>
    <div>MFE: <b>${money.format(t.mfe_usd)}</b> | MAE: <b>${money.format(t.mae_usd)}</b></div>
    ${extraFields}
    <div>Duration: <b>${t.duration_min} min</b> | Hit 1R: <b>${t.hit_1r ? "yes" : "no"}</b> | Hit 2R: <b>${t.hit_2r ? "yes" : "no"}</b></div>
  `);
}

function getSignalTrade() {
  if (selectedTradeNo !== null) {
    const selected = allTrades.find((t) => t.trade_no === selectedTradeNo);
    if (selected) return selected;
  }
  const rows = filteredTrades.length ? filteredTrades : allTrades;
  return rows.slice().sort((a, b) => b.entry_time - a.entry_time)[0] || null;
}

function buildTelegramSignalPayload(t) {
  return {
    strategy: "Super Structure",
    timeframe: currentTimeframe,
    trade_no: t.trade_no,
    side: t.side,
    session: t.session,
    entry_ts: t.entry_ts,
    entry_time: t.entry_time,
    entry_price: t.entry_price,
    exit_ts: t.exit_ts,
    exit_time: t.exit_time,
    exit_price: t.exit_price,
    exit_reason: t.exit_reason,
    pnl_usd: t.pnl_usd,
    r_multiple: t.r_multiple,
    entry_adx: t.entry_adx,
    entry_cci: t.entry_cci,
  };
}

async function sendTelegramSignal() {
  const trade = getSignalTrade();
  const $btn = $("#telegramSignalBtn");
  if (!trade) {
    setStatus("Telegram signal failed: no trade is available");
    return;
  }

  const originalText = $btn.text();
  $btn.prop("disabled", true).addClass("active").text("Sending...");
  try {
    await $.ajax({
      url: telegramApiUrl(),
      method: "POST",
      contentType: "application/json",
      dataType: "json",
      data: JSON.stringify(buildTelegramSignalPayload(trade)),
      timeout: 10000,
    });
    setStatus(`Telegram signal sent for ${currentTimeframe} trade #${trade.trade_no}`);
  } catch (err) {
    const message = err?.responseJSON?.error || err?.statusText || "request failed";
    setStatus(`Telegram signal failed: ${message}`);
    console.error(err);
  } finally {
    $btn.prop("disabled", false).removeClass("active").text(originalText);
  }
}

function selectTrade(tradeNo) {
  selectedTradeNo = tradeNo;
  const t = allTrades.find((x) => x.trade_no === tradeNo);
  renderDetail(t);
  renderTable(filteredTrades);

  if (t) {
    chart.timeScale().setVisibleRange({
      from: t.entry_time - 60 * 60 * 4,
      to: t.exit_time + 60 * 60 * 4,
    });
  }
}

function applyFilters() {
  const activeRange = getActiveDateRange();
  filteredTrades = getFilteredTrades();
  renderStats(filteredTrades);
  renderMarkers(filteredTrades);
  renderPnlCurve(filteredTrades);
  renderTable(filteredTrades);
  setStatus(`${formatDateRangeLabel(activeRange, filteredTrades)} | ${filteredTrades.length.toLocaleString()} visible trades`);
}

async function loadData() {
  const strategy = currentStrategy;
  const tf = currentTimeframe;
  console.log("[loadData]", strategy, tf, candleUrl(strategy, tf));
  setLoading(true, `Loading ${tf} strategy...`);
  allTrades = [];
  allCandles = [];
  filteredTrades = [];
  selectedTradeNo = null;
  renderDetail(null);

  const [candlesData, tradesData] = await Promise.all([
    $.ajax({ url: candleUrl(currentStrategy, currentTimeframe), cache: false, dataType: "json" }),
    $.ajax({ url: tradeUrl(currentStrategy, currentTimeframe), cache: false, dataType: "json" }),
    delay(250),
  ]);

  allCandles = candlesData.candles;
  allTrades = tradesData.trades;
  candleSeries.setData(allCandles);
  chart.timeScale().fitContent();
  applyFilters();
  setLoading(false);
  setStrategyStatus();
}

function candleUrl(strategy, timeframe) {
  return `data/candles_${strategy}_${timeframe}.json`;
}

function tradeUrl(strategy, timeframe) {
  return `data/trade_events_${strategy}_${timeframe}.json`;
}

async function loadTimeframe(timeframe) {
  currentTimeframe = timeframe;
  selectedTradeNo = null;
  renderDetail(null);
  setStatus(`Loading ${timeframe} strategy data...`);
  setLoading(true, `Loading ${timeframe} strategy...`);
  const [candlesData, tradesData] = await Promise.all([
    $.ajax({ url: candleUrl(currentStrategy, timeframe), cache: false, dataType: "json" }),
    $.ajax({ url: tradeUrl(currentStrategy, timeframe), cache: false, dataType: "json" }),
    delay(250),
  ]);
  allCandles = candlesData.candles;
  allTrades = tradesData.trades;
  candleSeries.setData(allCandles);
  chart.timeScale().fitContent();
  applyFilters();
  setLoading(false);
  setStrategyStatus();
}

function sessionVals() {
  const checked = [];
  $("#sessionFilter input:checked").each(function () { checked.push($(this).val()); });
  return checked;
}

function setSessionVals(vals) {
  if (!vals || !vals.length) {
    $("#sessionFilter input").prop("checked", true);
  } else {
    $("#sessionFilter input").prop("checked", false);
    vals.forEach(function (v) {
      $(`#sessionFilter input[value="${v}"]`).prop("checked", true);
    });
  }
}

function setStrategyStatus() {
  const name = $("#strategySelect option:selected").text();
  const total = allTrades.length;
  const pnl = allTrades.reduce((s, t) => s + t.pnl_usd, 0);
  const wins = allTrades.filter(t => t.is_win).length;
  const wr = total ? (wins / total * 100).toFixed(1) : "0";
  setStatus(`${name} | ${currentTimeframe} | ${total} trades | PnL: ${money.format(pnl)} | WR: ${wr}%`);
}

function readUrlParams() {
  const p = new URLSearchParams(window.location.search);
  if (!p.toString()) return false;
  console.log("[URL] Loading params:", p.toString());
  if (p.has("strategy")) {
    currentStrategy = p.get("strategy");
    $("#strategySelect").val(currentStrategy);
    console.log("[URL] Strategy:", currentStrategy);
  }
  if (p.has("timeframe")) {
    currentTimeframe = p.get("timeframe");
    $("#timeframeSelect").val(currentTimeframe);
  }
  if (p.has("date_from")) {
    $("#dateStart").val(p.get("date_from"));
    $("#rangePreset").val("Custom");
  }
  if (p.has("date_to")) {
    $("#dateEnd").val(p.get("date_to"));
    $("#rangePreset").val("Custom");
  }
  if (p.has("range")) $("#rangePreset").val(p.get("range"));
  if (p.has("side")) $("#sideFilter").val(p.get("side"));
  if (p.has("session")) {
    const s = p.get("session");
    if (s) setSessionVals(s.split(","));
  } else {
    setSessionVals(null);
  }
  return true;
}

function updateUrl() {
  const range = getActiveDateRange();
  const p = new URLSearchParams();
  p.set("strategy", currentStrategy);
  p.set("timeframe", currentTimeframe);
  if (range.preset === "Custom") {
    if ($("#dateStart").val()) p.set("date_from", $("#dateStart").val());
    if ($("#dateEnd").val()) p.set("date_to", $("#dateEnd").val());
  } else {
    p.set("range", range.preset);
  }
  const side = $("#sideFilter").val();
  if (side !== "All") p.set("side", side);
  const sessions = sessionVals();
  const allSessions = currentStrategy === "fvg_scalper"
    ? ["Asia", "London", "NY", "Other"]
    : ["Tokyo", "London", "US", "Other"];
  if (sessions.length && sessions.length < allSessions.length) {
    p.set("session", sessions.join(","));
  }
  const url = `${window.location.pathname}?${p.toString()}`;
  if (url !== window.location.search) {
    window.history.replaceState(null, "", url);
  }
}

function applyFilters() {
  const activeRange = getActiveDateRange();
  filteredTrades = getFilteredTrades();
  renderStats(filteredTrades);
  renderMarkers(filteredTrades);
  renderPnlCurve(filteredTrades);
  renderTable(filteredTrades);
  setStrategyStatus();
  updateUrl();
}

function updateSessionFilter() {
  const $sel = $("#sessionFilter");
  $sel.empty();
  const sessions = currentStrategy === "fvg_scalper"
    ? ["Asia", "London", "NY", "Other"]
    : ["Tokyo", "London", "US", "Other"];
  sessions.forEach(function (s) {
    $sel.append(`<label class="form-check form-check-inline"><input class="form-check-input" type="checkbox" value="${s}">${s}</label>`);
  });
  $("#sessionFilter input").on("change", applyFilters);
  setSessionVals(null);  // all checked by default
}

function applyBestPreset() {
  if (currentStrategy === "fvg_scalper") {
    $("#gapMin").val("1.0");
    $("#adxMin").val("12");
    $("#chopMax").val("62");
    $("#demaSlopeMin").val("0.03");
    setSessionVals(["Asia", "London"]);
  } else {
    $("#adxMin").val("25");
    setSessionVals(null);
  }
}

function resetFilters() {
  $("#rangePreset").val("All");
  $("#dateStart").val("");
  $("#dateEnd").val("");
  $("#sideFilter").val("All");
  setSessionVals(null);
  $("#resultFilter").val("All");
  $("#gapMin").val("");
  $("#gapMax").val("");
  $("#adxMin").val("");
  $("#adxMax").val("");
  $("#cciMin").val("");
  $("#cciMax").val("");
  $("#chopMin").val("");
  $("#chopMax").val("");
  $("#demaSlopeMin").val("");
  $("#demaDistMin").val("");
  $("#demaDistMax").val("");
  selectedTradeNo = null;
  renderDetail(null);
  applyFilters();
  chart.timeScale().fitContent();
}

function showProductView(view) {
  $(".product-tab").removeClass("active");
  $(`.product-tab[data-view="${view}"]`).addClass("active");
  $(".main-view").removeClass("active");
  $(`.main-view[data-view="${view}"]`).addClass("active");
  $(".sidebar-view").removeClass("active");
  $(`.sidebar-view[data-sidebar="${view}"]`).addClass("active");
  $(".app-shell").toggleClass("full-main", view !== "strategy");

  if (view === "strategy") {
    setStatus("Super Structure strategy explorer");
    setTimeout(function () {
      const el = $("#chart").get(0);
      chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
      resizePnlChart();
      applyFilters();
    }, 0);
    return;
  }

  const labels = {
    dashboard: "Product map preview",
    setups: "Find Setups preview: ORB ML research layer",
    evaluation: "Pass Evaluation preview: Topstep simulation layer",
    live: "Go Live preview: webhook, listener, execution, monitoring",
    data: "Data preview: source health and continuity",
  };
  const titles = {
    dashboard: ["Dashboard", "Product flow from setup discovery to live execution"],
    setups: ["Find Setups", "ORB ML research and setup discovery layer"],
    evaluation: ["Pass Evaluation", "Topstep feasibility and risk simulation layer"],
    live: ["Go Live", "Signal listener, risk gate, and execution layer"],
    data: ["Data", "Source health, continuity, and candle quality layer"],
  };
  if (titles[view]) setStatus(titles[view][1]);
  setStatus(labels[view] || "Preview");
}

$(function () {
  initChart();
  initPnlChart();
  $("select,input").not("#timeframeSelect,#strategySelect").on("input change", applyFilters);
  $("#timeframeSelect").on("change", function () {
    loadTimeframe($(this).val()).catch(function (err) {
      setLoading(false);
      setStatus(`Load failed: ${err.message}`);
      console.error(err);
    });
  });
  $("#strategySelect").on("change", function () {
    currentStrategy = $(this).val();
    if (currentStrategy === "orb_v2") return;
    $("#rangePreset").val("All");
    resetFilters();
    updateSessionFilter();
    applyBestPreset();
    updateUrl();
    setLoading(true);
    loadData().catch(function (err) {
      setLoading(false);
      setStatus(`Load failed: ${err.message}`);
      console.error(err);
    });
  });
  $("#resetFilters").on("click", resetFilters);
  $("#bestPresetBtn").on("click", function () {
    applyBestPreset();
    applyFilters();
  });
  $("#telegramSignalBtn").on("click", function () {
    sendTelegramSignal();
  });
  $(".strategy-tab").on("click", function () {
    const panel = $(this).data("strategy-panel");
    $(".strategy-tab").removeClass("active");
    $(this).addClass("active");
    $(".strategy-panel").removeClass("active");
    $(`#${panel}`).addClass("active");
    setTimeout(function () {
      const chartEl = $("#chart").get(0);
      chart.applyOptions({ width: chartEl.clientWidth, height: chartEl.clientHeight });
      resizePnlChart();
      if (panel === "pnlPanel") {
        pnlChart.timeScale().fitContent();
        pnlChart.timeScale().applyOptions({ rightOffset: 0 });
      }
    }, 0);
  });
  $(".product-tab").on("click", function () {
    showProductView($(this).data("view"));
  });
  $("#tradeRows").on("click", "tr", function () {
    selectTrade(Number($(this).data("trade-no")));
  });

  readUrlParams();

  loadData().catch(function (err) {
    setLoading(false);
    setStatus(`Load failed: ${err.message}`);
    console.error(err);
  });
  updateSessionFilter();
});
