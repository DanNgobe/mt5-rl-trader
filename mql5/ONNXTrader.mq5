//+------------------------------------------------------------------+
//|                                               ONNXTrader.mq5     |
//|                    Forex RL Trader — MT5 ONNX deployment         |
//|                                                                  |
//| Matches Python environment:                                      |
//|   Action space : MultiDiscrete([5, 3])                          |
//|     axis-0 direction : 0=HOLD  1=BUY  2=SELL                   |
//|                        3=CLOSE_LONG  4=CLOSE_SHORT              |
//|     axis-1 lot_tier  : 0=0.01  1=0.02  2=0.05                  |
//|   Observation  : [ohlcv_window | position_slots | account]      |
//|   Preprocessing: log returns on OHLC, z-score on volume         |
//|   Hedging mode : multiple positions per symbol                  |
//+------------------------------------------------------------------+
#property copyright "Forex RL Trader"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Inputs                                                            |
//+------------------------------------------------------------------+
input group "=== Model ==="
input string InpModelPath      = "Models\\model.onnx";
input int    InpWindowSize     = 10;       // Must match training
input int    InpMaxPositions   = 3;        // Must match training
input int    InpNFeatures      = 5;        // OHLCV = 5

input group "=== Trading ==="
input int    InpMagicNumber    = 234567;
input int    InpMaxSpreadPips  = 30;
input double InpInitialBalance = 10000.0;  // For account state normalisation

input group "=== Timeframe ==="
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1;

//+------------------------------------------------------------------+
//| Constants                                                         |
//+------------------------------------------------------------------+
// Lot tiers — must match LOT_TIERS in simulator.py
double LOT_TIERS[3] = {0.01, 0.02, 0.05};

// Observation dimension (set in OnInit)
int OBS_DIM = 0;

//+------------------------------------------------------------------+
//| Globals                                                           |
//+------------------------------------------------------------------+
long    g_onnx_handle = INVALID_HANDLE;
CTrade  g_trade;

double  g_vol_mean = 0.0;
double  g_vol_std  = 1.0;

//+------------------------------------------------------------------+
//| Position slot tracking                                            |
//+------------------------------------------------------------------+
struct PositionSlot
{
    bool   filled;
    int    direction;    // 1 = LONG, -1 = SHORT
    double lot_size;
    double entry_price;
    ulong  ticket;
};

PositionSlot g_slots[];

//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
    OBS_DIM = InpWindowSize * InpNFeatures + InpMaxPositions * 5 + 2;

    g_trade.SetExpertMagicNumber(InpMagicNumber);
    g_trade.SetDeviationInPoints(50);
    g_trade.SetTypeFilling(ORDER_FILLING_FOK);
    g_trade.SetAsyncMode(false);

    ArrayResize(g_slots, InpMaxPositions);
    ClearSlots();

    // OnnxCreate paths are relative to MQL5\Files\ for live trading,
    // but the Strategy Tester uses a different sandboxed folder.
    // ONNX_COMMON_FOLDER resolves to the shared Common\Files\ folder,
    // which works identically in both live and the Strategy Tester.
    //
    // Copy model.onnx to:
    //   [MT5 common data folder]\Files\Models\model.onnx
    // Find via: MT5 -> File -> Open Common Data Folder
    string common_path = TerminalInfoString(TERMINAL_COMMONDATA_PATH)
                         + "\\Files\\" + InpModelPath;
    Print("Resolved model path: ", common_path);
    g_onnx_handle = OnnxCreate(InpModelPath, ONNX_COMMON_FOLDER);
    if(g_onnx_handle == INVALID_HANDLE)
    {
        Print("ERROR: OnnxCreate failed (", GetLastError(),
              ") - ensure file exists at: ", common_path);
        return INIT_FAILED;
    }

    // MT5 ONNX runtime requires explicit shape when batch dim is dynamic
    ulong in_shape[]  = {1, (ulong)OBS_DIM};
    // output shape is now [1, 8]: 5 direction logits + 3 lot logits
    ulong out_shape[] = {1, 8};
    if(!OnnxSetInputShape(g_onnx_handle, 0, in_shape))
    {
        Print("ERROR: OnnxSetInputShape failed (", GetLastError(), ")");
        OnnxRelease(g_onnx_handle);
        return INIT_FAILED;
    }
    if(!OnnxSetOutputShape(g_onnx_handle, 0, out_shape))
    {
        Print("ERROR: OnnxSetOutputShape failed (", GetLastError(), ")");
        OnnxRelease(g_onnx_handle);
        return INIT_FAILED;
    }

    Print("ONNX loaded OK. obs_dim=", OBS_DIM, "  output=[1,8]");

    CalibrateVolumeStats();
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
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
    // Act only on the first tick of each new closed bar
    static datetime last_bar = 0;
    datetime cur_bar = (datetime)SeriesInfoInteger(_Symbol, InpTimeframe, SERIES_LASTBAR_DATE);
    if(cur_bar == last_bar) return;
    last_bar = cur_bar;

    double point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double spread = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / point;
    if(spread > InpMaxSpreadPips)
    {
        Print("Spread too high (", DoubleToString(spread, 1), "). Skipping.");
        return;
    }

    SyncSlots();

    float obs_buf[];
    if(!BuildObservation(obs_buf)) { Print("BuildObservation failed."); return; }

    float logits[];
    if(!RunInference(obs_buf, logits)) { Print("RunInference failed."); return; }

    int    dir_idx  = ArgMax(logits, 0, 5);
    int    lot_idx  = ArgMax(logits, 5, 3);
    double lot_size = LOT_TIERS[lot_idx];

    Print(StringFormat(
        "Action: %s  lot=%.2f  logits=[%.3f,%.3f,%.3f,%.3f,%.3f|%.3f,%.3f,%.3f]",
        DirectionName(dir_idx), lot_size,
        logits[0], logits[1], logits[2], logits[3], logits[4],
        logits[5], logits[6], logits[7]));

    ExecuteAction(dir_idx, lot_size);
}

//+------------------------------------------------------------------+
//| Build observation vector                                          |
//+------------------------------------------------------------------+
bool BuildObservation(float &obs_buf[])
{
    ArrayResize(obs_buf, OBS_DIM);
    ArrayInitialize(obs_buf, 0.0f);

    int n = InpWindowSize + 2;
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    if(CopyRates(_Symbol, InpTimeframe, 0, n, rates) < n)
    {
        Print("Not enough bars.");
        return false;
    }

    // --- 1. OHLCV log returns + z-scored volume (oldest candle first) ---
    int fill = 0;
    for(int i = InpWindowSize - 1; i >= 0; i--)
    {
        double prev_close = rates[i + 1].close;
        if(prev_close <= 0.0) prev_close = 1.0;

        obs_buf[fill++] = (float)MathLog(rates[i].open  / prev_close);
        obs_buf[fill++] = (float)MathLog(rates[i].high  / prev_close);
        obs_buf[fill++] = (float)MathLog(rates[i].low   / prev_close);
        obs_buf[fill++] = (float)MathLog(rates[i].close / prev_close);

        double vol_z = (g_vol_std > 1e-8)
                       ? ((double)rates[i].tick_volume - g_vol_mean) / g_vol_std
                       : 0.0;
        obs_buf[fill++] = (float)vol_z;
    }

    // --- 2. Position slots ---
    double cur_price = rates[0].close;
    for(int s = 0; s < InpMaxPositions; s++)
    {
        if(!g_slots[s].filled) { fill += 5; continue; }

        double entry       = g_slots[s].entry_price;
        double lots        = g_slots[s].lot_size;
        int    dir         = g_slots[s].direction;
        double price_delta = (entry - cur_price) / cur_price;
        double pnl         = (cur_price - entry) * (double)dir * lots * 100000.0;

        obs_buf[fill++] = 1.0f;
        obs_buf[fill++] = (float)dir;
        obs_buf[fill++] = (float)lots;
        obs_buf[fill++] = (float)price_delta;
        obs_buf[fill++] = (float)pnl;
    }

    // --- 3. Account state ---
    obs_buf[fill++] = (float)(AccountInfoDouble(ACCOUNT_BALANCE) / InpInitialBalance);
    obs_buf[fill++] = (float)(AccountInfoDouble(ACCOUNT_EQUITY)  / InpInitialBalance);

    return true;
}

//+------------------------------------------------------------------+
//| Run ONNX inference                                                |
//+------------------------------------------------------------------+
bool RunInference(const float &obs_buf[], float &logits[])
{
    ArrayResize(logits, 8);

    float obs_copy[];
    ArrayResize(obs_copy, OBS_DIM);
    ArrayCopy(obs_copy, obs_buf);

    if(!OnnxRun(g_onnx_handle, ONNX_DEFAULT, obs_copy, logits))
    {
        Print("OnnxRun failed (", GetLastError(), ")");
        return false;
    }
    return true;
}

//+------------------------------------------------------------------+
//| Execute action                                                    |
//+------------------------------------------------------------------+
void ExecuteAction(int dir_idx, double lot_size)
{
    int    open_count = CountOpenPositions();
    double ask        = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid        = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    if(dir_idx == 0) { Print("HOLD."); return; }

    if(dir_idx == 1)  // BUY
    {
        if(open_count >= InpMaxPositions)
            { Print("BUY ignored: cap reached."); return; }
        if(g_trade.Buy(lot_size, _Symbol, ask, 0, 0, "RL-BUY"))
            Print(StringFormat("BUY  lot=%.2f  ask=%.5f", lot_size, ask));
        else
            Print("BUY failed: ", g_trade.ResultRetcodeDescription());
        return;
    }

    if(dir_idx == 2)  // SELL
    {
        if(open_count >= InpMaxPositions)
            { Print("SELL ignored: cap reached."); return; }
        if(g_trade.Sell(lot_size, _Symbol, bid, 0, 0, "RL-SELL"))
            Print(StringFormat("SELL lot=%.2f  bid=%.5f", lot_size, bid));
        else
            Print("SELL failed: ", g_trade.ResultRetcodeDescription());
        return;
    }

    if(dir_idx == 3)  // CLOSE_LONG
        CloseOldestMatchingPosition(lot_size, POSITION_TYPE_BUY);

    if(dir_idx == 4)  // CLOSE_SHORT
        CloseOldestMatchingPosition(lot_size, POSITION_TYPE_SELL);
}

//+------------------------------------------------------------------+
//| Close oldest position matching lot_size (FIFO)                   |
//+------------------------------------------------------------------+
void CloseOldestMatchingPosition(double lot_size, ENUM_POSITION_TYPE pos_type)
{
    ulong    best_ticket = 0;
    datetime best_time   = D'3000.01.01';
    int      total       = PositionsTotal();

    for(int i = 0; i < total; i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetString(POSITION_SYMBOL)              != _Symbol)        continue;
        if((int)PositionGetInteger(POSITION_MAGIC)         != InpMagicNumber) continue;
        if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != pos_type) continue;
        if(MathAbs(PositionGetDouble(POSITION_VOLUME) - lot_size) > 1e-9)     continue;

        datetime t = (datetime)PositionGetInteger(POSITION_TIME);
        if(t < best_time) { best_time = t; best_ticket = ticket; }
    }

    string type_str = (pos_type == POSITION_TYPE_BUY) ? "LONG" : "SHORT";
    if(best_ticket == 0)
        { Print(StringFormat("CLOSE_%s %.2f: no match.", type_str, lot_size)); return; }

    if(g_trade.PositionClose(best_ticket))
        Print(StringFormat("CLOSE_%s ticket=%d lot=%.2f", type_str, (int)best_ticket, lot_size));
    else
        Print("CLOSE failed: ", g_trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| Sync g_slots with live MT5 positions                             |
//+------------------------------------------------------------------+
void SyncSlots()
{
    ClearSlots();
    int slot = 0, total = PositionsTotal();

    for(int i = 0; i < total && slot < InpMaxPositions; i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetString(POSITION_SYMBOL)      != _Symbol)        continue;
        if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;

        g_slots[slot].filled      = true;
        g_slots[slot].ticket      = ticket;
        g_slots[slot].lot_size    = PositionGetDouble(POSITION_VOLUME);
        g_slots[slot].entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
        g_slots[slot].direction   = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
        slot++;
    }
}

//+------------------------------------------------------------------+
//| Calibrate volume z-score from recent history                     |
//+------------------------------------------------------------------+
void CalibrateVolumeStats()
{
    long vol_arr[];
    ArraySetAsSeries(vol_arr, true);
    int copied = CopyTickVolume(_Symbol, InpTimeframe, 0, 500, vol_arr);

    if(copied < 10)
    {
        Print("Volume calibration: insufficient bars. Using defaults.");
        g_vol_mean = 0.0; g_vol_std = 1.0;
        return;
    }

    double s = 0.0, sq = 0.0;
    for(int i = 0; i < copied; i++) s += (double)vol_arr[i];
    g_vol_mean = s / copied;
    for(int i = 0; i < copied; i++) sq += MathPow((double)vol_arr[i] - g_vol_mean, 2);
    g_vol_std = MathSqrt(sq / copied);
    if(g_vol_std < 1e-8) g_vol_std = 1.0;

    Print(StringFormat("Volume: mean=%.1f std=%.1f (%d bars)", g_vol_mean, g_vol_std, copied));
}

//+------------------------------------------------------------------+
//| Helpers                                                           |
//+------------------------------------------------------------------+
void ClearSlots()
{
    for(int i = 0; i < InpMaxPositions; i++)
        { g_slots[i].filled=false; g_slots[i].direction=0;
          g_slots[i].lot_size=0; g_slots[i].entry_price=0; g_slots[i].ticket=0; }
}

int CountOpenPositions()
{
    int c = 0, total = PositionsTotal();
    for(int i = 0; i < total; i++)
    {
        ulong t = PositionGetTicket(i);
        if(t == 0) continue;
        if(PositionGetString(POSITION_SYMBOL)      == _Symbol &&
           (int)PositionGetInteger(POSITION_MAGIC) == InpMagicNumber) c++;
    }
    return c;
}

int ArgMax(const float &arr[], int start, int length)
{
    int best = start; float bv = arr[start];
    for(int i = start+1; i < start+length; i++)
        if(arr[i] > bv) { bv = arr[i]; best = i; }
    return best - start;
}

string DirectionName(int idx)
{
    if(idx==0) return "HOLD";
    if(idx==1) return "BUY";
    if(idx==2) return "SELL";
    if(idx==3) return "CLOSE_LONG";
    if(idx==4) return "CLOSE_SHORT";
    return "UNKNOWN";
}
//+------------------------------------------------------------------+