//+------------------------------------------------------------------+
//|                                               ONNXTrader.mq5     |
//|                    Forex RL Trader — MT5 ONNX deployment         |
//|                                                                  |
//| Action space : Discrete(1 + 2*N_TIERS)                          |
//|   0               = HOLD                                        |
//|   1 + tier*2      = BUY  lot_tiers[tier]  (toggle open/close)  |
//|   2 + tier*2      = SELL lot_tiers[tier]  (toggle open/close)  |
//|                                                                  |
//| Observation (must match Python env exactly):                     |
//|   price_lags log-returns  [1,2,4,8,24]                          |
//|   RSI(14) normalised to [-1,1]                                  |
//|   ATR(14) / close                                               |
//|   EMA(8)/EMA(21) - 1  clipped [-0.1, 0.1]                      |
//|   Bollinger %B(20,2) normalised [-1,1]                          |
//|   Momentum log-returns [5,20,50]                                |
//|   Session: hour_sin, hour_cos, dow_sin, dow_cos                 |
//|   Position slots: n_tiers*2 × [dir*lot, upnl_norm, bars_norm]  |
//|   balance_norm, equity_norm                                     |
//+------------------------------------------------------------------+
#property copyright "Forex RL Trader"
#property version   "3.00"
#property tester_file "model.onnx"

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Inputs                                                            |
//+------------------------------------------------------------------+
input group "=== Model ==="
input string InpModelPath      = "model.onnx";

input group "=== Lot Tiers (must match training config) ==="
input double InpLot0           = 0.1;    // Tier 0 lot size  (0 to disable)
input double InpLot1           = 0.2;    // Tier 1 lot size  (0 to disable)
input double InpLot2           = 0.5;    // Tier 2 lot size  (0 to disable)

input group "=== Trading ==="
input int    InpMagicNumber    = 234567;
input int    InpMaxSpreadPips  = 30;
input double InpInitialBalance = 1000.0; // Must match training initial_balance

input group "=== Timeframe ==="
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1;

//+------------------------------------------------------------------+
//| Constants matching Python preprocessor defaults                  |
//+------------------------------------------------------------------+
int    PRICE_LAGS[]    = {1, 2, 4, 8, 24};   // must match obs_config.price_lags
int    MOM_PERIODS[]   = {5, 20, 50};         // must match obs_config.indicators.momentum.periods
int    RSI_PERIOD      = 14;
int    ATR_PERIOD      = 14;
int    EMA_FAST        = 8;
int    EMA_SLOW        = 21;
int    BOLL_PERIOD     = 20;
double BOLL_STDDEV     = 2.0;

//+------------------------------------------------------------------+
//| Logging helper                                                   |
//+------------------------------------------------------------------+
void LogInfo(string msg)  { Print("[INFO]  ", msg); }
void LogWarn(string msg)  { Print("[WARN]  ", msg); }
void LogError(string msg) { Print("[ERROR] ", msg); }

//+------------------------------------------------------------------+
//| Globals                                                           |
//+------------------------------------------------------------------+
long   g_onnx_handle = INVALID_HANDLE;
CTrade g_trade;

// Built from inputs at OnInit
double g_lot_tiers[];   // active lot tiers
int    g_n_tiers  = 0;
int    g_n_actions = 0;
int    g_n_slots   = 0;
int    g_obs_dim   = 0;

// Bars needed for the longest lag / indicator
int    g_bars_needed = 0;

// Per-slot open tracking: ticket, open_step (bar index at open)
ulong  g_slot_tickets[];
int    g_slot_open_bar[];   // absolute bar index when position was opened
int    g_total_bars = 0;    // incremented each new bar


//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
    // Build lot tiers from inputs (skip zeros)
    double raw_tiers[3] = {InpLot0, InpLot1, InpLot2};
    g_n_tiers = 0;
    for(int i = 0; i < 3; i++)
        if(raw_tiers[i] > 1e-9) g_n_tiers++;

    if(g_n_tiers == 0) { Print("ERROR: all lot tiers are 0."); return INIT_FAILED; }

    ArrayResize(g_lot_tiers, g_n_tiers);
    int t = 0;
    for(int i = 0; i < 3; i++)
        if(raw_tiers[i] > 1e-9) g_lot_tiers[t++] = raw_tiers[i];

    g_n_actions = 1 + 2 * g_n_tiers;
    g_n_slots   = g_n_tiers * 2;

    // obs_dim mirrors obs_dim_from_config() in preprocessor.py:
    //   price_lags + rsi + atr + ema_ratio + bollinger + momentum + session(4) + slots*3 + account(2)
    int n_price_lags = ArraySize(PRICE_LAGS);
    int n_mom_periods = ArraySize(MOM_PERIODS);
    g_obs_dim = n_price_lags + 1 + 1 + 1 + 1 + n_mom_periods + 4 + g_n_slots * 3 + 2;

    // Bars needed: max(price_lags) + max(indicator lookback)
    // max lag=24, EMA slow warmup ~3*slow=63, Bollinger=20, momentum max=50
    g_bars_needed = MathMax(PRICE_LAGS[4], MathMax(EMA_SLOW * 3, MOM_PERIODS[2])) + 5;

    ArrayResize(g_slot_tickets,  g_n_slots);
    ArrayResize(g_slot_open_bar, g_n_slots);
    ArrayInitialize(g_slot_tickets,  0);
    ArrayInitialize(g_slot_open_bar, 0);

    g_trade.SetExpertMagicNumber(InpMagicNumber);
    g_trade.SetDeviationInPoints(50);
    g_trade.SetTypeFilling(ORDER_FILLING_FOK);
    g_trade.SetAsyncMode(false);

    //--- 1. Check file exists
    LogInfo("Checking model file: " + InpModelPath);
    
    if(!FileIsExist(InpModelPath))
    {
        LogError("Model file NOT found: " + InpModelPath);
        LogError("Place the file at: [DataFolder]\\MQL5\\Files\\" + InpModelPath);
        LogError("Open Data Folder via: File -> Open Data Folder in MT5");
        return INIT_FAILED;
    }
    
    LogInfo("✓ Model file found.");

    //--- 2. Load ONNX model (CPU only for compatibility)
    LogInfo("Attempting OnnxCreate (CPU only)...");
    g_onnx_handle = OnnxCreate(InpModelPath, ONNX_USE_CPU_ONLY);

    if(g_onnx_handle == INVALID_HANDLE)
    {
        int err = GetLastError();
        LogError("OnnxCreate failed. Error code: " + IntegerToString(err));

        switch(err)
        {
            case 5019: LogError("ERR_ONNX_CANNOT_CREATE — file not found, corrupt, or unsupported opset."); break;
            case 5020: LogError("ERR_ONNX_CANNOT_EXECUTE — runtime execution failure."); break;
            case 5021: LogError("ERR_ONNX_INCORRECT_INPUT_SHAPE — input shape mismatch."); break;
            case 5022: LogError("ERR_ONNX_INCORRECT_OUTPUT_SHAPE — output shape mismatch."); break;
            default:   LogError("Unknown ONNX error."); break;
        }

        LogError("Possible causes:");
        LogError("  1. ONNX opset > 17 (MT5 supports up to opset 17)");
        LogError("  2. File is corrupt or not a valid ONNX file");
        LogError("  3. Model uses unsupported ops");
        return INIT_FAILED;
    }
    
    LogInfo("✓ OnnxCreate succeeded. Handle=" + IntegerToString(g_onnx_handle));

    //--- 3. Set input/output shapes
    ulong in_shape[]  = {1, (ulong)g_obs_dim};
    ulong out_shape[] = {1, (ulong)g_n_actions};
    
    LogInfo("Setting input shape: [1, " + IntegerToString(g_obs_dim) + "]");
    if(!OnnxSetInputShape(g_onnx_handle, 0, in_shape))
    {
        LogError("OnnxSetInputShape failed. Error: " + IntegerToString(GetLastError()));
        OnnxRelease(g_onnx_handle);
        return INIT_FAILED;
    }
    
    LogInfo("Setting output shape: [1, " + IntegerToString(g_n_actions) + "]");
    if(!OnnxSetOutputShape(g_onnx_handle, 0, out_shape))
    {
        LogError("OnnxSetOutputShape failed. Error: " + IntegerToString(GetLastError()));
        OnnxRelease(g_onnx_handle);
        return INIT_FAILED;
    }

    LogInfo("=== ONNXTrader initialised ===");
    LogInfo("ModelPath     = " + InpModelPath);
    LogInfo("InitialBalance= " + DoubleToString(InpInitialBalance, 2));
    LogInfo("MaxPositions  = " + IntegerToString(g_n_slots));
    LogInfo("OBS_DIM       = " + IntegerToString(g_obs_dim));
    LogInfo("Timeframe     = " + IntegerToString(InpTimeframe));
    LogInfo("Lot tiers     = " + IntegerToString(g_n_tiers));
    LogInfo("Actions       = " + IntegerToString(g_n_actions));
    LogInfo("Bars needed   = " + IntegerToString(g_bars_needed));
    
    Print(StringFormat("ONNXTrader ready. tiers=%d  n_actions=%d  obs_dim=%d",
                       g_n_tiers, g_n_actions, g_obs_dim));
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(g_onnx_handle != INVALID_HANDLE)
        OnnxRelease(g_onnx_handle);
}


//+------------------------------------------------------------------+
//| OnTick — act once per closed bar                                 |
//+------------------------------------------------------------------+
void OnTick()
{
    static datetime last_bar = 0;
    datetime cur_bar = (datetime)SeriesInfoInteger(_Symbol, InpTimeframe, SERIES_LASTBAR_DATE);
    if(cur_bar == last_bar) return;
    last_bar = cur_bar;
    g_total_bars++;

    double spread_pts = (SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                       - SymbolInfoDouble(_Symbol, SYMBOL_BID))
                       / SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    if(spread_pts > InpMaxSpreadPips)
    {
        Print("Spread too high (", DoubleToString(spread_pts, 1), "). Skipping.");
        return;
    }

    float obs_buf[];
    if(!BuildObservation(obs_buf)) { Print("BuildObservation failed."); return; }

    float logits[];
    ArrayResize(logits, g_n_actions);
    float obs_copy[];
    ArrayResize(obs_copy, g_obs_dim);
    ArrayCopy(obs_copy, obs_buf);

    if(!OnnxRun(g_onnx_handle, ONNX_DEFAULT, obs_copy, logits))
    {
        Print("OnnxRun failed (", GetLastError(), ")");
        return;
    }

    // Argmax over all n_actions logits
    int action = 0;
    float best = logits[0];
    for(int i = 1; i < g_n_actions; i++)
        if(logits[i] > best) { best = logits[i]; action = i; }

    Print(StringFormat("Action=%d  %s", action, ActionName(action)));
    ExecuteAction(action);
}


//+------------------------------------------------------------------+
//| Execute action — toggle semantics                                |
//+------------------------------------------------------------------+
void ExecuteAction(int action)
{
    if(action == 0) { Print("HOLD."); return; }

    // Decode tier and direction from action index
    int tier      = (action - 1) / 2;
    bool is_buy   = ((action - 1) % 2 == 0);
    double lot    = g_lot_tiers[tier];
    ENUM_POSITION_TYPE pos_type = is_buy ? POSITION_TYPE_BUY : POSITION_TYPE_SELL;

    // Toggle: close if matching position exists, else open
    ulong match_ticket = FindOldestMatchingPosition(pos_type, lot);

    if(match_ticket != 0)
    {
        // Close
        if(g_trade.PositionClose(match_ticket))
            Print(StringFormat("CLOSE %s lot=%.2f ticket=%d",
                               is_buy ? "LONG" : "SHORT", lot, (int)match_ticket));
        else
            Print("CLOSE failed: ", g_trade.ResultRetcodeDescription());
    }
    else
    {
        // Open
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        bool ok;
        if(is_buy)
            ok = g_trade.Buy(lot, _Symbol, ask, 0, 0, "RL-BUY");
        else
            ok = g_trade.Sell(lot, _Symbol, bid, 0, 0, "RL-SELL");

        if(ok)
        {
            // Record open bar for bars_open tracking
            ulong new_ticket = g_trade.ResultOrder();
            RegisterSlot(new_ticket, g_total_bars);
            Print(StringFormat("OPEN %s lot=%.2f ticket=%d",
                               is_buy ? "BUY" : "SELL", lot, (int)new_ticket));
        }
        else
            Print("OPEN failed: ", g_trade.ResultRetcodeDescription());
    }
}

//+------------------------------------------------------------------+
//| Find oldest open position matching type + lot                    |
//+------------------------------------------------------------------+
ulong FindOldestMatchingPosition(ENUM_POSITION_TYPE pos_type, double lot)
{
    ulong    best_ticket = 0;
    datetime best_time   = D'3000.01.01';
    int      total       = PositionsTotal();

    for(int i = 0; i < total; i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetString(POSITION_SYMBOL)                    != _Symbol)        continue;
        if((int)PositionGetInteger(POSITION_MAGIC)               != InpMagicNumber) continue;
        if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != pos_type)       continue;
        if(MathAbs(PositionGetDouble(POSITION_VOLUME) - lot)     > 1e-9)            continue;

        datetime t = (datetime)PositionGetInteger(POSITION_TIME);
        if(t < best_time) { best_time = t; best_ticket = ticket; }
    }
    return best_ticket;
}

//+------------------------------------------------------------------+
//| Slot registry for bars_open tracking                             |
//+------------------------------------------------------------------+
void RegisterSlot(ulong ticket, int open_bar)
{
    for(int i = 0; i < g_n_slots; i++)
    {
        if(g_slot_tickets[i] == 0)
        {
            g_slot_tickets[i]  = ticket;
            g_slot_open_bar[i] = open_bar;
            return;
        }
    }
}

int BarsOpenForTicket(ulong ticket)
{
    for(int i = 0; i < g_n_slots; i++)
        if(g_slot_tickets[i] == ticket)
            return g_total_bars - g_slot_open_bar[i];
    return 0;
}

void PruneClosedSlots()
{
    for(int i = 0; i < g_n_slots; i++)
    {
        if(g_slot_tickets[i] == 0) continue;
        if(!PositionSelectByTicket(g_slot_tickets[i]))
        {
            g_slot_tickets[i]  = 0;
            g_slot_open_bar[i] = 0;
        }
    }
}


//+------------------------------------------------------------------+
//| Build observation vector — mirrors TradingEnv._observation()     |
//+------------------------------------------------------------------+
bool BuildObservation(float &obs[])
{
    ArrayResize(obs, g_obs_dim);
    ArrayInitialize(obs, 0.0f);

    int n_bars = g_bars_needed + 10;
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    if(CopyRates(_Symbol, InpTimeframe, 0, n_bars, rates) < n_bars)
    {
        Print("Not enough bars (need ", n_bars, ")");
        return false;
    }

    // rates[0] = most recent closed bar, rates[k] = k bars ago
    // close array for indicator computation (oldest first for EMA)
    double close[];
    ArrayResize(close, n_bars);
    for(int i = 0; i < n_bars; i++)
        close[i] = rates[n_bars - 1 - i].close;   // index 0 = oldest

    int fill = 0;

    // ------------------------------------------------------------------
    // 1. Sparse lagged close log-returns
    //    Python: close_log_returns[t - lag]  where t = current step
    //    In MQL5: rates[lag].close / rates[lag+1].close
    // ------------------------------------------------------------------
    int n_lags = ArraySize(PRICE_LAGS);
    for(int li = 0; li < n_lags; li++)
    {
        int lag = PRICE_LAGS[li];
        double prev = rates[lag].close;
        double cur  = rates[lag - 1].close;   // one bar newer
        if(prev <= 0.0) prev = 1.0;
        obs[fill++] = (float)MathLog(cur / prev);
    }

    // ------------------------------------------------------------------
    // 2. RSI(14) normalised to [-1,1]: (rsi/50) - 1
    // ------------------------------------------------------------------
    obs[fill++] = (float)ComputeRSI(close, n_bars, RSI_PERIOD);

    // ------------------------------------------------------------------
    // 3. ATR(14) / close
    // ------------------------------------------------------------------
    obs[fill++] = (float)ComputeATR(rates, n_bars, ATR_PERIOD);

    // ------------------------------------------------------------------
    // 4. EMA(8)/EMA(21) - 1, clipped [-0.1, 0.1]
    // ------------------------------------------------------------------
    obs[fill++] = (float)ComputeEMARatio(close, n_bars, EMA_FAST, EMA_SLOW);

    // ------------------------------------------------------------------
    // 5. Bollinger %B(20,2) normalised to [-1,1]: (pct_b - 0.5)*2
    // ------------------------------------------------------------------
    obs[fill++] = (float)ComputeBollinger(close, n_bars, BOLL_PERIOD, BOLL_STDDEV);

    // ------------------------------------------------------------------
    // 6. Momentum: log(close[t] / close[t-p]) for p in [5,20,50]
    // ------------------------------------------------------------------
    int n_mom = ArraySize(MOM_PERIODS);
    for(int mi = 0; mi < n_mom; mi++)
    {
        int p = MOM_PERIODS[mi];
        double c0 = rates[0].close;
        double cp = rates[p].close;
        obs[fill++] = (cp > 0.0) ? (float)MathLog(c0 / cp) : 0.0f;
    }

    // ------------------------------------------------------------------
    // 7. Session features: hour_sin, hour_cos, dow_sin, dow_cos
    // ------------------------------------------------------------------
    MqlDateTime dt;
    TimeToStruct(rates[0].time, dt);
    double hour = (double)dt.hour;
    double dow  = (double)dt.day_of_week;   // 0=Sun..6=Sat; Python uses 0=Mon..4=Fri
    obs[fill++] = (float)MathSin(2.0 * M_PI * hour / 24.0);
    obs[fill++] = (float)MathCos(2.0 * M_PI * hour / 24.0);
    obs[fill++] = (float)MathSin(2.0 * M_PI * dow  / 5.0);
    obs[fill++] = (float)MathCos(2.0 * M_PI * dow  / 5.0);

    // ------------------------------------------------------------------
    // 8. Position slots: [direction*lot_size, upnl_norm, bars_open_norm]
    //    Sorted by ticket (ascending) to match Python's sorted-by-ticket order.
    // ------------------------------------------------------------------
    PruneClosedSlots();

    // Collect live positions for this symbol/magic, sorted by ticket
    ulong  live_tickets[];
    double live_dir[];
    double live_lot[];
    double live_entry[];
    int    live_bars[];
    int    n_live = 0;

    int total = PositionsTotal();
    for(int i = 0; i < total; i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetString(POSITION_SYMBOL)      != _Symbol)        continue;
        if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;

        ArrayResize(live_tickets, n_live + 1);
        ArrayResize(live_dir,     n_live + 1);
        ArrayResize(live_lot,     n_live + 1);
        ArrayResize(live_entry,   n_live + 1);
        ArrayResize(live_bars,    n_live + 1);

        live_tickets[n_live] = ticket;
        live_dir[n_live]     = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1.0 : -1.0;
        live_lot[n_live]     = PositionGetDouble(POSITION_VOLUME);
        live_entry[n_live]   = PositionGetDouble(POSITION_PRICE_OPEN);
        live_bars[n_live]    = BarsOpenForTicket(ticket);
        n_live++;
    }

    // Sort by ticket ascending (insertion sort — n_live is small)
    for(int i = 1; i < n_live; i++)
    {
        ulong kt = live_tickets[i]; double kd = live_dir[i];
        double kl = live_lot[i]; double ke = live_entry[i]; int kb = live_bars[i];
        int j = i - 1;
        while(j >= 0 && live_tickets[j] > kt)
        {
            live_tickets[j+1] = live_tickets[j]; live_dir[j+1]  = live_dir[j];
            live_lot[j+1]     = live_lot[j];     live_entry[j+1] = live_entry[j];
            live_bars[j+1]    = live_bars[j];    j--;
        }
        live_tickets[j+1] = kt; live_dir[j+1] = kd;
        live_lot[j+1] = kl; live_entry[j+1] = ke; live_bars[j+1] = kb;
    }

    double cur_price = rates[0].close;
    int    n_total_bars_approx = g_total_bars > 0 ? g_total_bars : 1;

    for(int s = 0; s < g_n_slots; s++)
    {
        if(s < n_live)
        {
            double dir   = live_dir[s];
            double lot   = live_lot[s];
            double entry = live_entry[s];
            double upnl  = (cur_price - entry) * dir * lot * 100000.0 / InpInitialBalance;
            double bars_norm = (double)live_bars[s] / (double)n_total_bars_approx;

            obs[fill++] = (float)(dir * lot);
            obs[fill++] = (float)MathMax(-1.0, MathMin(1.0, upnl));
            obs[fill++] = (float)bars_norm;
        }
        else
        {
            obs[fill++] = 0.0f;
            obs[fill++] = 0.0f;
            obs[fill++] = 0.0f;
        }
    }

    // ------------------------------------------------------------------
    // 9. Account state
    // ------------------------------------------------------------------
    obs[fill++] = (float)(AccountInfoDouble(ACCOUNT_BALANCE) / InpInitialBalance);
    obs[fill++] = (float)(AccountInfoDouble(ACCOUNT_EQUITY)  / InpInitialBalance);

    return true;
}


//+------------------------------------------------------------------+
//| Indicator helpers — match Python preprocessor exactly            |
//+------------------------------------------------------------------+

// EMA of a series (oldest-first array), returns value at last index
double EMA(const double &series[], int n, int period)
{
    double alpha = 2.0 / (period + 1.0);
    double ema   = series[0];
    for(int i = 1; i < n; i++)
        ema = alpha * series[i] + (1.0 - alpha) * ema;
    return ema;
}

// EMA array (oldest-first), fills result[] of length n
void EMAArray(const double &series[], int n, int period, double &result[])
{
    ArrayResize(result, n);
    double alpha = 2.0 / (period + 1.0);
    result[0] = series[0];
    for(int i = 1; i < n; i++)
        result[i] = alpha * series[i] + (1.0 - alpha) * result[i-1];
}

// RSI(period) normalised to [-1,1]: (rsi/50)-1
// close[] is oldest-first, length n; returns value at last bar
double ComputeRSI(const double &close[], int n, int period)
{
    if(n < period + 1) return 0.0;

    // Wilder EMA of gains/losses
    double alpha = 1.0 / (double)period;
    double avg_gain = 0.0, avg_loss = 0.0;

    // Seed with first period
    for(int i = 1; i <= period; i++)
    {
        double d = close[i] - close[i-1];
        if(d > 0) avg_gain += d; else avg_loss -= d;
    }
    avg_gain /= period;
    avg_loss /= period;

    for(int i = period + 1; i < n; i++)
    {
        double d = close[i] - close[i-1];
        double g = (d > 0) ? d : 0.0;
        double l = (d < 0) ? -d : 0.0;
        avg_gain = alpha * g + (1.0 - alpha) * avg_gain;
        avg_loss = alpha * l + (1.0 - alpha) * avg_loss;
    }

    double rs  = (avg_loss > 1e-10) ? avg_gain / avg_loss : 100.0;
    double rsi = 100.0 - 100.0 / (1.0 + rs);
    return rsi / 50.0 - 1.0;
}

// ATR(period) / close — rates[] is newest-first (as returned by CopyRates)
double ComputeATR(const MqlRates &rates[], int n, int period)
{
    if(n < period + 1) return 0.0;

    double alpha = 1.0 / (double)period;
    // Seed
    double atr = 0.0;
    for(int i = n - 2; i >= n - 1 - period; i--)
    {
        double tr = MathMax(rates[i].high - rates[i].low,
                   MathMax(MathAbs(rates[i].high - rates[i+1].close),
                           MathAbs(rates[i].low  - rates[i+1].close)));
        atr += tr;
    }
    atr /= period;

    // Wilder smooth to most recent bar
    for(int i = n - 1 - period - 1; i >= 0; i--)
    {
        double tr = MathMax(rates[i].high - rates[i].low,
                   MathMax(MathAbs(rates[i].high - rates[i+1].close),
                           MathAbs(rates[i].low  - rates[i+1].close)));
        atr = alpha * tr + (1.0 - alpha) * atr;
    }

    double c = rates[0].close;
    return (c > 0.0) ? atr / c : 0.0;
}

// EMA(fast)/EMA(slow) - 1, clipped [-0.1, 0.1]
double ComputeEMARatio(const double &close[], int n, int fast, int slow)
{
    double ema_f = EMA(close, n, fast);
    double ema_s = EMA(close, n, slow);
    if(ema_s <= 0.0) return 0.0;
    double ratio = ema_f / ema_s - 1.0;
    return MathMax(-0.1, MathMin(0.1, ratio));
}

// Bollinger %B(period, std_dev) normalised to [-1,1]: (pct_b - 0.5)*2
// close[] oldest-first; returns value at last bar
double ComputeBollinger(const double &close[], int n, int period, double std_dev)
{
    if(n < period) return 0.0;

    // Rolling mean and std of last `period` values
    double sum = 0.0, sq = 0.0;
    for(int i = n - period; i < n; i++) sum += close[i];
    double mean = sum / period;
    for(int i = n - period; i < n; i++) sq += MathPow(close[i] - mean, 2);
    double std = MathSqrt(sq / period);

    double upper = mean + std_dev * std;
    double lower = mean - std_dev * std;
    double band  = upper - lower;
    double pct_b = (band > 1e-10) ? (close[n-1] - lower) / band : 0.5;
    double norm  = (pct_b - 0.5) * 2.0;
    return MathMax(-1.5, MathMin(1.5, norm));
}

//+------------------------------------------------------------------+
//| Utility                                                           |
//+------------------------------------------------------------------+
string ActionName(int action)
{
    if(action == 0) return "HOLD";
    int tier    = (action - 1) / 2;
    bool is_buy = ((action - 1) % 2 == 0);
    return StringFormat("%s lot=%.2f (tier %d)",
                        is_buy ? "BUY" : "SELL", g_lot_tiers[tier], tier);
}
//+------------------------------------------------------------------+
