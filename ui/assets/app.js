const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

let chart;
let candleSeries;
let pnlChart;
let pnlSeries;
let allCandles = [];
let allTrades = [];
let filteredTrades = [];
let selectedTradeNo = null;
let currentTimeframe = "5m";

function setStatus(text) {
  $("#status").text(text);
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
  pnlSeries = pnlChart.addAreaSeries({
    lineColor: "#6aa1ff",
    topColor: "rgba(106, 161, 255, .28)",
    bottomColor: "rgba(106, 161, 255, .02)",
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
  const session = $("#sessionFilter").val();
  const result = $("#resultFilter").val();
  const adxMin = parseFloat($("#adxMin").val());
  const adxMax = parseFloat($("#adxMax").val());
  const cciMin = parseFloat($("#cciMin").val());
  const cciMax = parseFloat($("#cciMax").val());

  return allTrades.filter(function (t) {
    if (range.startTime !== null && t.entry_time < range.startTime) return false;
    if (range.endTime !== null && t.entry_time > range.endTime) return false;
    if (side !== "All" && t.side !== side) return false;
    if (session !== "All" && t.session !== session) return false;
    if (result === "Win" && !t.is_win) return false;
    if (result === "Loss" && t.is_win) return false;
    if (!Number.isNaN(adxMin) && t.entry_adx < adxMin) return false;
    if (!Number.isNaN(adxMax) && t.entry_adx > adxMax) return false;
    if (!Number.isNaN(cciMin) && t.entry_cci < cciMin) return false;
    if (!Number.isNaN(cciMax) && t.entry_cci > cciMax) return false;
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
  if (!pnlSeries) return;
  const rows = trades.slice().sort((a, b) => a.exit_time - b.exit_time);
  let cumulative = 0;
  const data = rows.map((t) => {
    cumulative += t.pnl_usd;
    return {
      time: t.exit_time,
      value: Math.round(cumulative),
    };
  });
  pnlSeries.setData(data);
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

function renderTable(trades) {
  const $body = $("#tradeRows");
  $body.empty();

  const rows = trades.slice().sort((a, b) => b.entry_time - a.entry_time);
  for (const t of rows) {
    const sideClass = t.side === "Long" ? "long" : "short";
    const pnlClass = t.pnl_usd >= 0 ? "win" : "loss";
    const rClass = t.r_multiple >= 0 ? "win" : "loss";
    const selectedClass = t.trade_no === selectedTradeNo ? "selected" : "";
    const rText = t.r_multiple?.toFixed(2) ?? "";

    const $row = $(`
      <tr class="${selectedClass}" data-trade-no="${t.trade_no}">
        <td>${t.trade_no}</td>
        <td>${t.entry_ts}</td>
        <td class="${sideClass}">${t.side}</td>
        <td class="text-end">${fmt.format(t.entry_price)}</td>
        <td class="text-end ${pnlClass}">${money.format(t.pnl_usd)}</td>
        <td class="text-end ${rClass}">${rText}</td>
        <td class="text-end">${t.entry_adx.toFixed(1)}</td>
        <td class="text-end">${t.entry_cci.toFixed(0)}</td>
        <td>${t.session}</td>
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

  $el.html(`
    <div><b>#${t.trade_no}</b> <span class="${t.side === "Long" ? "long" : "short"}">${t.side}</span> <span class="pill">${t.session}</span></div>
    <div>Entry: <b>${t.entry_ts}</b> at <b>${fmt.format(t.entry_price)}</b></div>
    <div>Exit: <b>${t.exit_ts}</b> at <b>${fmt.format(t.exit_price)}</b> <span class="pill">${t.exit_reason}</span></div>
    <div>PnL: <b class="${t.pnl_usd >= 0 ? "win" : "loss"}">${money.format(t.pnl_usd)}</b> | R: <b>${t.r_multiple?.toFixed(2) ?? "n/a"}</b></div>
    <div>MFE: <b>${money.format(t.mfe_usd)}</b> | MAE: <b>${money.format(t.mae_usd)}</b></div>
    <div>ADX: <b>${t.entry_adx.toFixed(2)}</b> | CCI: <b>${t.entry_cci.toFixed(2)}</b></div>
    <div>DEMA dist ATR: <b>${t.dema_distance_atr?.toFixed(2) ?? "n/a"}</b> | ST dist ATR: <b>${t.st_distance_atr?.toFixed(2) ?? "n/a"}</b></div>
    <div>Duration: <b>${t.duration_min} min</b> | Hit 1R: <b>${t.hit_1r ? "yes" : "no"}</b> | Hit 2R: <b>${t.hit_2r ? "yes" : "no"}</b></div>
  `);
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
  setLoading(true, `Loading ${currentTimeframe} strategy...`);
  const [candlesData, tradesData] = await Promise.all([
    $.ajax({ url: candleUrl(currentTimeframe), cache: false, dataType: "json" }),
    $.ajax({ url: tradeUrl(currentTimeframe), cache: false, dataType: "json" }),
    delay(250),
  ]);

  allCandles = candlesData.candles;
  allTrades = tradesData.trades;
  candleSeries.setData(allCandles);
  chart.timeScale().fitContent();
  applyFilters();
  setLoading(false);
}

function candleUrl(timeframe) {
  return `data/candles_${timeframe}.json`;
}

function tradeUrl(timeframe) {
  return `data/trade_events_${timeframe}.json`;
}

async function loadTimeframe(timeframe) {
  currentTimeframe = timeframe;
  selectedTradeNo = null;
  renderDetail(null);
  setStatus(`Loading ${timeframe} strategy data...`);
  setLoading(true, `Loading ${timeframe} strategy...`);
  const [candlesData, tradesData] = await Promise.all([
    $.ajax({ url: candleUrl(timeframe), cache: false, dataType: "json" }),
    $.ajax({ url: tradeUrl(timeframe), cache: false, dataType: "json" }),
    delay(250),
  ]);
  allCandles = candlesData.candles;
  allTrades = tradesData.trades;
  candleSeries.setData(allCandles);
  chart.timeScale().fitContent();
  applyFilters();
  setLoading(false);
}

function resetFilters() {
  $("#rangePreset").val("All");
  $("#dateStart").val("");
  $("#dateEnd").val("");
  $("#sideFilter").val("All");
  $("#sessionFilter").val("All");
  $("#resultFilter").val("All");
  $("#adxMin").val("");
  $("#adxMax").val("");
  $("#cciMin").val("");
  $("#cciMax").val("");
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
    setViewCopy("Test Strategy", "ST + DEMA + ADX + CCI strategy explorer");
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
  if (titles[view]) setViewCopy(titles[view][0], titles[view][1]);
  setStatus(labels[view] || "Preview");
}

function setViewCopy(title, subtitle) {
  $("#viewTitle").text(title);
  $("#viewSubtitle").text(subtitle);
}

$(function () {
  initChart();
  initPnlChart();
  $("select,input").not("#timeframeSelect").on("input change", applyFilters);
  $("#timeframeSelect").on("change", function () {
    loadTimeframe($(this).val()).catch(function (err) {
      setLoading(false);
      setStatus(`Load failed: ${err.message}`);
      console.error(err);
    });
  });
  $("#resetFilters").on("click", resetFilters);
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

  loadData().catch(function (err) {
    setLoading(false);
    setStatus(`Load failed: ${err.message}`);
    console.error(err);
  });
});
