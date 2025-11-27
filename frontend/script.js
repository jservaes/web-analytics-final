const API_BASE = "http://127.0.0.1:8000";

const tickerInput = document.getElementById("ticker-input");
const loadBtn = document.getElementById("load-btn");
const statusSpan = document.getElementById("status");

const ret1yEl = document.getElementById("ret-1y");
const ret3yEl = document.getElementById("ret-3y");
const ret5yEl = document.getElementById("ret-5y");
const rev1yEl = document.getElementById("rev-1y");

let priceChart = null;
let fundChart = null;

loadBtn.addEventListener("click", () => {
  const ticker = tickerInput.value.trim();
  if (!ticker) return;
  loadCompany(ticker);
});

// Fetch data from FastAPI
async function loadCompany(ticker) {
  statusSpan.textContent = "Loading...";
  try {
    const url = `${API_BASE}/api/company-summary?ticker=${encodeURIComponent(
      ticker
    )}`;
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }
    const data = await res.json();
    statusSpan.textContent = `${data.name} (${data.ticker})`;

    updateSummaryCards(data);
    updateCharts(data);
  } catch (err) {
    console.error(err);
    statusSpan.textContent = "Error loading company";
  }
}

function formatPct(x) {
  if (x === null || x === undefined) return "–";
  return (x * 100).toFixed(1) + "%";
}

function formatBillions(x) {
  if (x === null || x === undefined) return "–";
  return (x / 1e9).toFixed(1) + " B";
}

function updateSummaryCards(data) {
  const returns = data.predictions.returns || {};
  const fund1y = data.predictions.fundamentals_1y || {};
  const vals = fund1y.values || {};

  ret1yEl.textContent = formatPct(returns["1y"]);
  ret3yEl.textContent = formatPct(returns["3y"]);
  ret5yEl.textContent = formatPct(returns["5y"]);
  rev1yEl.textContent = formatBillions(vals["revenue"]);
}

function updateCharts(data) {
  const years = data.history.year;
  const price = data.history.mean_price;
  const revenue = data.history.revenue;
  const netIncome = data.history.net_income;
  const ocf = data.history.operating_cash_flow;

  const ctxPrice = document.getElementById("priceChart").getContext("2d");
  const ctxFund = document.getElementById("fundChart").getContext("2d");

  if (priceChart) priceChart.destroy();
  if (fundChart) fundChart.destroy();

  // Price history chart (line)
  priceChart = new Chart(ctxPrice, {
    type: "line",
    data: {
      labels: years,
      datasets: [
        {
          label: "Mean price",
          data: price,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "Year" } },
        y: { title: { display: true, text: "Price (USD)" } },
      },
    },
  });

  // Fundamentals chart (line, multiple series)
  fundChart = new Chart(ctxFund, {
    type: "line",
    data: {
      labels: years,
      datasets: [
        {
          label: "Revenue",
          data: revenue,
        },
        {
          label: "Net income",
          data: netIncome,
        },
        {
          label: "Operating cash flow",
          data: ocf,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "Year" } },
        y: {
          title: { display: true, text: "USD" },
          ticks: {
            callback: (v) => (v / 1e9).toFixed(1) + " B",
          },
        },
      },
    },
  });
}
